pub mod ffmpeg;
pub mod ffprobe;
pub mod whisper_xxl;
// `prkt` is gone: Parakeet inference moved in-process to `crate::asr`, so
// there is no external transcription binary to invoke any more.
pub mod alass;
pub mod locate;
pub mod subtitle_edit;
pub mod python;

use crate::error::ToolError;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

/// A prepared tool invocation.
pub struct ToolInvocation {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub working_dir: Option<PathBuf>,
    pub env_overrides: Vec<(String, String)>,
    pub timeout: Option<Duration>,
}

/// Result of running a tool to completion.
pub struct ToolOutput {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Line-by-line output from a running tool, sent via channel.
#[derive(Debug, Clone)]
pub enum ToolLine {
    Stdout(String),
    Stderr(String),
    Done { exit_code: Option<i32> },
}

/// Run a tool to completion, capturing all output.
pub async fn run_tool(inv: &ToolInvocation) -> Result<ToolOutput, ToolError> {
    let tool_name = inv
        .program
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut cmd = Command::new(&inv.program);
    cmd.args(&inv.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if let Some(ref wd) = inv.working_dir {
        cmd.current_dir(wd);
    }
    for (key, val) in &inv.env_overrides {
        cmd.env(key, val);
    }

    let child = cmd.spawn().map_err(|e| ToolError::Spawn {
        tool: tool_name.clone(),
        source: e,
    })?;

    let output = if let Some(timeout) = inv.timeout {
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(result) => result.map_err(|e| ToolError::Spawn {
                tool: tool_name.clone(),
                source: e,
            })?,
            Err(_) => {
                return Err(ToolError::Timeout {
                    tool: tool_name,
                    timeout_secs: timeout.as_secs(),
                });
            }
        }
    } else {
        child.wait_with_output().await.map_err(|e| ToolError::Spawn {
            tool: tool_name.clone(),
            source: e,
        })?
    };

    Ok(ToolOutput {
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Run a tool at BELOW_NORMAL Windows priority so its CPU work yields to
/// foreground apps. Used for Phase-1 audio decode of large files: same total
/// CPU work, but the desktop / browser / TUI stay responsive while it runs.
/// On non-Windows this is identical to `run_tool`.
#[cfg(target_os = "windows")]
pub async fn run_tool_lowprio(inv: &ToolInvocation) -> Result<ToolOutput, ToolError> {
    use std::os::windows::process::CommandExt;

    let tool_name = inv
        .program
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let mut cmd = Command::new(&inv.program);
    cmd.args(&inv.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if let Some(ref wd) = inv.working_dir {
        cmd.current_dir(wd);
    }
    for (key, val) in &inv.env_overrides {
        cmd.env(key, val);
    }

    // BELOW_NORMAL_PRIORITY_CLASS = 0x00004000.
    // Process keeps full CPU when system is idle; yields under contention.
    cmd.creation_flags(0x00004000);

    let child = cmd.spawn().map_err(|e| ToolError::Spawn {
        tool: tool_name.clone(),
        source: e,
    })?;

    let output = if let Some(timeout) = inv.timeout {
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(result) => result.map_err(|e| ToolError::Spawn {
                tool: tool_name.clone(),
                source: e,
            })?,
            Err(_) => {
                return Err(ToolError::Timeout {
                    tool: tool_name,
                    timeout_secs: timeout.as_secs(),
                });
            }
        }
    } else {
        child.wait_with_output().await.map_err(|e| ToolError::Spawn {
            tool: tool_name.clone(),
            source: e,
        })?
    };

    Ok(ToolOutput {
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

#[cfg(not(target_os = "windows"))]
pub async fn run_tool_lowprio(inv: &ToolInvocation) -> Result<ToolOutput, ToolError> {
    run_tool(inv).await
}

/// Watch a single output line for faster-whisper-xxl's "finished" markers
/// and pre-emptively kill the process tree when one is seen. This prevents
/// the PyInstaller + CUDA DLL unload crash at process exit from firing
/// Windows Error Reporting ("...has stopped working" dialog). By the time
/// any of these markers appear, whisper's transcription output is already
/// flushed to disk, so a hard kill loses nothing. Windows-only: the exit
/// crash it races against is a Windows DLL-unload problem, so elsewhere the
/// process is simply left to exit on its own.
fn maybe_early_kill_whisper(
    line: &str,
    is_flaky: bool,
    flag: &AtomicBool,
    pid: Option<u32>,
) {
    if !cfg!(target_os = "windows") || !is_flaky || flag.load(Ordering::SeqCst) {
        return;
    }
    // Any of these markers means whisper has finished writing outputs.
    // "Operation finished in:" is the last line before the cleanup crash.
    // The other two are emitted earlier, giving us even more head-start
    // against the crash race.
    let finished = line.contains("Operation finished in:")
        || line.contains("Subtitles are saved to")
        || line.contains("Subtitles written to");
    if !finished {
        return;
    }
    if let Some(pid) = pid {
        kill_process_tree(pid);
        flag.store(true, Ordering::SeqCst);
    }
}

/// Run a tool with real-time line-by-line output streaming via channel.
pub async fn run_tool_streaming(
    inv: &ToolInvocation,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<Option<i32>, ToolError> {
    let tool_name = inv
        .program
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // On Unix the child leads its own process group, so `kill_process_tree`
    // can signal the whole group (helpers the tool forks included) without
    // touching fwxl-burn itself. `process_group` lives on std's Command, so
    // it is set there before handing the command to tokio.
    #[cfg(unix)]
    let mut cmd = {
        use std::os::unix::process::CommandExt;
        let mut std_cmd = std::process::Command::new(&inv.program);
        std_cmd.process_group(0);
        Command::from(std_cmd)
    };
    #[cfg(not(unix))]
    let mut cmd = Command::new(&inv.program);
    cmd.args(&inv.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if let Some(ref wd) = inv.working_dir {
        cmd.current_dir(wd);
    }
    for (key, val) in &inv.env_overrides {
        cmd.env(key, val);
    }

    // On Windows, create a new process group so we can kill the entire tree,
    // AND give the child its own hidden console (CREATE_NO_WINDOW).
    //
    // Without the hidden console, faster-whisper-xxl shares OUR console and
    // repaints the window title with its tqdm progress ("52% | 213/406 ...")
    // several times a second, clobbering the "[i/N] file | mode/model | pct"
    // title the TUI maintains. The child's stdout/stderr are pipes either
    // way (we stream them into the log), and the tree is torn down with
    // `taskkill /T`, not console control events, so nothing else changes.
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }

    let mut child = cmd.spawn().map_err(|e| ToolError::Spawn {
        tool: tool_name.clone(),
        source: e,
    })?;

    let child_pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let tx_out = tx.clone();
    let tx_err = tx.clone();

    // Identify whisper-family tools once up front. Both the stream tasks
    // (for marker-based early kill) and the pipe-EOF fallback below consult
    // this flag.
    let is_flaky_tool: bool = {
        let name_lower = tool_name.to_lowercase();
        name_lower.contains("faster-whisper") || name_lower.contains("whisper-xxl")
    };
    // Shared across stdout task, stderr task, and post-wait block so all
    // three agree on whether we've already killed the tree.
    let killed_early_flag: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

    // Stream stdout lines
    let flag_stdout = Arc::clone(&killed_early_flag);
    let pid_stdout = child_pid;
    let flaky_stdout = is_flaky_tool;
    let stdout_handle = tokio::spawn(async move {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                maybe_early_kill_whisper(&line, flaky_stdout, &flag_stdout, pid_stdout);
                let _ = tx_out.send(ToolLine::Stdout(line));
            }
        }
    });

    // Stream stderr lines
    let flag_stderr = Arc::clone(&killed_early_flag);
    let pid_stderr = child_pid;
    let flaky_stderr = is_flaky_tool;
    let stderr_handle = tokio::spawn(async move {
        if let Some(stderr) = stderr {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                maybe_early_kill_whisper(&line, flaky_stderr, &flag_stderr, pid_stderr);
                let _ = tx_err.send(ToolLine::Stderr(line));
            }
        }
    });

    // Wait for BOTH pipes to close before waiting on exit. If the marker-
    // based kill above fired, the pipes close very quickly as taskkill
    // tears down the child. Otherwise this waits for whisper to either
    // exit cleanly or crash.
    let _ = stdout_handle.await;
    let _ = stderr_handle.await;

    // WINDOWS "STOPPED WORKING" WORKAROUND
    // ---------------------------------------------------------------------
    // faster-whisper-xxl is a PyInstaller bundle that crashes during its
    // own process exit (CUDA DLL unload ordering race, numpy/torch atexit
    // handlers) AFTER all transcription outputs are already on disk. That
    // crash fires Windows Error Reporting and pops the "Standalone
    // Faster-Whisper-XXL Pro has stopped working" dialog even though the
    // SRT / JSON / VTT files have been written.
    //
    // PRIMARY FIX: the stream tasks above watch for whisper's
    // "Operation finished in:" / "Subtitles are saved to" markers and call
    // `kill_process_tree` the instant one appears, winning the race against
    // the cleanup crash (TerminateProcess does not trigger WER).
    //
    // FALLBACK: if neither marker was seen (renamed output, crashed earlier,
    // non-standard build) we still kill at pipe EOF for whisper-family tools.
    // Same intent, slightly higher chance of losing the race with the
    // already-running crash.
    #[cfg(target_os = "windows")]
    let killed_early = {
        if killed_early_flag.load(Ordering::SeqCst) {
            true
        } else if is_flaky_tool {
            if let Some(pid) = child_pid {
                kill_process_tree(pid);
            }
            true
        } else {
            false
        }
    };
    #[cfg(not(target_os = "windows"))]
    let killed_early = killed_early_flag.load(Ordering::SeqCst);

    let status = child.wait().await.map_err(|e| ToolError::Spawn {
        tool: tool_name.clone(),
        source: e,
    })?;

    // Kill any orphaned child processes (CUDA workers, Python subprocesses)
    // -- no-op if we already killed the tree above.
    if !killed_early {
        if let Some(pid) = child_pid {
            kill_process_tree(pid);
        }
    }

    // If we killed the child pre-emptively and it would otherwise have
    // exited 0, report 0 to callers so the "exit code != 0" warning path
    // isn't falsely triggered by our own termination.
    let code = if killed_early {
        Some(0)
    } else {
        status.code()
    };
    let _ = tx.send(ToolLine::Done { exit_code: code });

    Ok(code)
}

/// Kill a process and all its children on Windows using taskkill /T.
/// This is more reliable than kill_on_drop alone because faster-whisper-xxl
/// spawns child processes (Python, CUDA workers) that survive TerminateProcess.
#[cfg(target_os = "windows")]
pub fn kill_process_tree(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

/// Kill a process and everything it forked on Unix by signalling its process
/// group. `run_tool_streaming` makes each child the leader of its own group,
/// so the group id equals the child's pid and a negative pid addresses the
/// whole group. A group that has already exited makes `kill` fail harmlessly.
///
/// The `--` is load-bearing. This runs the external `kill` binary, not a
/// shell builtin, and procps-ng's `kill -KILL -1234` parses `-1234` as a
/// second option: it exits 0 having signalled nothing (verified on procps-ng
/// 4.0.4). `--` ends option parsing so the negative pid is read as a target.
#[cfg(unix)]
pub fn kill_process_tree(pid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

#[cfg(not(any(unix, target_os = "windows")))]
pub fn kill_process_tree(_pid: u32) {}

/// Validate that an output artifact exists and is non-empty.
pub fn validate_output(path: &Path, tool: &str, description: &str) -> Result<(), ToolError> {
    let meta = std::fs::metadata(path).map_err(|_| ToolError::MissingOutput {
        tool: tool.to_string(),
        description: description.to_string(),
        path: path.to_path_buf(),
    })?;
    if meta.len() == 0 {
        return Err(ToolError::EmptyOutput {
            tool: tool.to_string(),
            description: description.to_string(),
            path: path.to_path_buf(),
        });
    }
    Ok(())
}
