//! Pre-flight: make sure the input is actually decodable before a transcription
//! run is started against it.
//!
//! An interrupted recording (`moov` never written) is byte-perfect audio that no
//! decoder will open. Handed straight to whisper it burns a full run and writes
//! nothing at all -- the failure mode that produced empty `.fwxl-recovery`
//! directories. Repairing here turns that into a few seconds of work followed by
//! a normal transcription.
//!
//! The original file is never modified: `fwxl-audiofix` opens it read-only and
//! writes a separate repaired copy, which becomes the input for this run only.

use crate::error::PipelineError;
use crate::pipeline::{PipelineContext, PipelineEvent};
use crate::probe;
use crate::tools::{run_tool, ToolInvocation};
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

/// The subset of `fwxl-audiofix --json` we act on.
#[derive(serde::Deserialize, Debug)]
struct FixReport {
    ok: bool,
    status: String,
    output: Option<String>,
    #[serde(default)]
    class: String,
    #[serde(default)]
    duration_seconds: f64,
    #[serde(default)]
    missing_bytes: Option<u64>,
    #[serde(default)]
    message: String,
}

/// Probe `ctx.media`; if it is unreadable, repair it and repoint `ctx` at the
/// repaired copy.
///
/// Returns `Ok(())` when the pipeline may proceed. Returns `Err` when the input
/// cannot be made readable -- which is a far better outcome than a transcription
/// run that was never going to produce anything.
pub async fn ensure_readable(
    ctx: &mut PipelineContext,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<(), PipelineError> {
    let status = match probe::probe_media_file(&ctx.config, &mut ctx.media).await {
        Ok(s) => s,
        // ffprobe itself could not be run. Not something repair can fix, and not
        // evidence the file is bad, so let the run continue as before.
        Err(e) => {
            let _ = event_tx.send(PipelineEvent::Warning(format!(
                "Could not probe input ({e}); continuing without a pre-flight check"
            )));
            return Ok(());
        }
    };

    let Some(reason) = status.reason() else {
        return Ok(());
    };

    let _ = event_tx.send(PipelineEvent::Warning(format!(
        "Input container is unreadable: {reason}"
    )));

    if !ctx.config.audio_repair.enabled {
        return Err(PipelineError::SrtParse(format!(
            "{} cannot be read by ffmpeg ({reason}), and automatic repair is disabled",
            ctx.media.path.display()
        )));
    }

    let Some(tool) = resolve_tool(ctx) else {
        return Err(PipelineError::SrtParse(format!(
            "{} cannot be read by ffmpeg ({reason}). fwxl-audiofix was not found, so it \
             cannot be repaired automatically -- set tools.audiofix in the config.",
            ctx.media.path.display()
        )));
    };

    let out_path = repaired_path(ctx);
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: "Repairing damaged container".to_string(),
        phase_num: 0,
        total_phases: 2,
    });
    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Repair: {} -> {}",
        tool.display(),
        out_path.display()
    )));

    let inv = ToolInvocation {
        program: tool,
        args: vec![
            OsString::from("--json"),
            OsString::from("--force"),
            OsString::from("-i"),
            OsString::from(&ctx.media.path),
            OsString::from("-o"),
            OsString::from(&out_path),
            OsString::from("--ffprobe"),
            OsString::from(&ctx.config.tools.ffprobe),
            OsString::from("--ffmpeg"),
            OsString::from(&ctx.config.tools.ffmpeg),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        // A full-decode verification pass over a multi-hour recording is the
        // slowest part; this bound is generous rather than tight.
        timeout: Some(Duration::from_secs(3600)),
    };

    let out = run_tool(&inv).await.map_err(|e| PipelineError::ToolFailed {
        phase: "container repair".to_string(),
        source: Box::new(e),
    })?;

    let report: FixReport = serde_json::from_str(out.stdout.trim()).map_err(|e| {
        PipelineError::SrtParse(format!(
            "could not understand fwxl-audiofix output ({e}); stderr: {}",
            out.stderr.trim()
        ))
    })?;

    if !report.ok || report.status != "repaired" {
        return Err(PipelineError::SrtParse(format!(
            "repair of {} did not succeed: {}",
            ctx.media.path.display(),
            report.message
        )));
    }

    let Some(repaired) = report.output.map(PathBuf::from) else {
        return Err(PipelineError::SrtParse(
            "fwxl-audiofix reported success but named no output file".to_string(),
        ));
    };

    // Truncation is a property of the source, not a defect of the repair. Say so
    // plainly so a short transcript is never a surprise.
    if let Some(missing) = report.missing_bytes {
        let _ = event_tx.send(PipelineEvent::Warning(format!(
            "{} bytes were never written to this recording; {:.0} minutes recovered, \
             the rest is unrecoverable",
            missing,
            report.duration_seconds / 60.0
        )));
    }

    let _ = event_tx.send(PipelineEvent::Warning(format!(
        "Repaired ({}): {:.0} minutes now decodable. Original left untouched.",
        report.class,
        report.duration_seconds / 60.0
    )));

    // Re-point this run at the repaired copy and re-probe so duration and stream
    // info reflect what will actually be transcribed.
    ctx.media.path = repaired;
    match probe::probe_media_file(&ctx.config, &mut ctx.media).await {
        Ok(s) if s.is_readable() => Ok(()),
        Ok(s) => Err(PipelineError::SrtParse(format!(
            "the repaired file is still unreadable: {}",
            s.reason().unwrap_or("unknown reason")
        ))),
        Err(e) => Err(PipelineError::ToolFailed {
            phase: "probing repaired file".to_string(),
            source: Box::new(e),
        }),
    }
}

/// Configured path, else a sibling of our own executable.
fn resolve_tool(ctx: &PipelineContext) -> Option<PathBuf> {
    if let Some(p) = &ctx.config.tools.audiofix {
        return p.is_file().then(|| p.clone());
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let name = crate::tools::locate::exe_name("fwxl-audiofix");
    for cand in [dir.join(&name), dir.join("fwxl-audiofix").join(&name)] {
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Where the repaired copy goes. Never the source's own path.
fn repaired_path(ctx: &PipelineContext) -> PathBuf {
    let stem = ctx.media.stem.clone();
    let base = match ctx.config.audio_repair.out_subdir.as_deref() {
        Some(sub) if !sub.is_empty() => ctx.media.dir.join(sub),
        _ => ctx.media.dir.clone(),
    };
    base.join(format!("{stem}.repaired.m4a"))
}
