use crate::error::ToolError;
use crate::tools::{ToolInvocation, ToolLine, run_tool, run_tool_streaming};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;

/// Name of the Python interpreter to launch (resolved through PATH).
///
/// Windows installs expose `python`; there, `python3` is usually the
/// Microsoft Store stub that opens the store instead of running anything.
/// Linux and macOS are the reverse: `python3` is the reliable name and a bare
/// `python` is often absent.
fn find_python() -> PathBuf {
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

/// Run the translation Python script.
pub async fn run_translation(
    script_path: &Path,
    input_srt: &Path,
    target_lang: &str,
    mkv_path: Option<&Path>,
    ffmpeg_path: &Path,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<Option<i32>, ToolError> {
    let mut args: Vec<OsString> = vec![
        OsString::from(script_path),
        OsString::from(input_srt),
        OsString::from("--to"),
        OsString::from(target_lang),
        OsString::from("--ffmpeg"),
        OsString::from(ffmpeg_path),
    ];

    if let Some(mkv) = mkv_path {
        args.push(OsString::from("--mkv"));
        args.push(OsString::from(mkv));
    }

    let inv = ToolInvocation {
        program: find_python(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(3600)),
    };

    run_tool_streaming(&inv, tx).await
}

/// Run the PGS/VOBSUB OCR Python script.
pub async fn run_pgs_ocr(
    script_path: &Path,
    media_path: &Path,
    stream_index: u32,
    output_srt: &Path,
    ffmpeg_path: &Path,
    lang: &str,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<Option<i32>, ToolError> {
    let args: Vec<OsString> = vec![
        OsString::from(script_path),
        OsString::from(media_path),
        OsString::from("--stream"),
        OsString::from(stream_index.to_string()),
        OsString::from("--output"),
        OsString::from(output_srt),
        OsString::from("--ffmpeg"),
        OsString::from(ffmpeg_path),
        OsString::from("--lang"),
        OsString::from(lang),
    ];

    let inv = ToolInvocation {
        program: find_python(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(3600)),
    };

    run_tool_streaming(&inv, tx).await
}
