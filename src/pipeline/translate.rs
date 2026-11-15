use crate::error::PipelineError;
use crate::pipeline::{PipelineContext, PipelineEvent};
use crate::tools::python;
use crate::tools::ToolLine;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Run the translation pipeline.
pub async fn run_translate(
    ctx: &PipelineContext,
    input_srt: &std::path::Path,
    target_lang: &str,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PathBuf, PipelineError> {
    let script_path = ctx
        .config
        .translation
        .script_path
        .as_ref()
        .ok_or_else(|| PipelineError::ModeUnavailable {
            mode: "Translate".to_string(),
            reason: "Translation script path not configured".to_string(),
        })?;

    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!("Translating to {target_lang}"),
        phase_num: 1,
        total_phases: 1,
    });

    let (tool_tx, mut tool_rx) = mpsc::unbounded_channel::<ToolLine>();
    let event_tx2 = event_tx.clone();
    let forward = tokio::spawn(async move {
        while let Some(line) = tool_rx.recv().await {
            match line {
                ToolLine::Stdout(l) => {
                    let _ = event_tx2.send(PipelineEvent::StdoutLine(l));
                }
                ToolLine::Stderr(l) => {
                    let _ = event_tx2.send(PipelineEvent::StderrLine(l));
                }
                ToolLine::Done { .. } => break,
            }
        }
    });

    let mkv_path = if ctx.media.is_video() {
        Some(ctx.media.path.as_path())
    } else {
        None
    };

    python::run_translation(script_path, input_srt, target_lang, mkv_path, &ctx.config.tools.ffmpeg, tool_tx)
        .await
        .map_err(|e| PipelineError::ToolFailed {
            phase: "translation".to_string(),
            source: Box::new(e),
        })?;

    let _ = forward.await;

    // The Python script generates the output SRT alongside the input
    let output_srt = input_srt.with_extension(format!("{target_lang}.srt"));

    Ok(output_srt)
}
