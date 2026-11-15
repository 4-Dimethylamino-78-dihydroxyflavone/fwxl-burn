use crate::error::PipelineError;
use crate::pipeline::{PipelineContext, PipelineEvent};
use crate::tools::python;
use crate::tools::ToolLine;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Run bitmap subtitle OCR (PGS/VOBSUB) via Python script.
pub async fn run_bitmap_ocr(
    ctx: &PipelineContext,
    stream_index: u32,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PathBuf, PipelineError> {
    let script_path = ctx
        .config
        .ocr
        .script_path
        .as_ref()
        .ok_or_else(|| PipelineError::ModeUnavailable {
            mode: "OCR".to_string(),
            reason: "PGS OCR script path not configured".to_string(),
        })?;

    let output_srt = ctx.srt_dir.join(format!("{}.ocr.srt", ctx.media.stem));

    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: "Running bitmap subtitle OCR".to_string(),
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

    python::run_pgs_ocr(
        script_path,
        &ctx.media.path,
        stream_index,
        &output_srt,
        &ctx.config.tools.ffmpeg,
        &ctx.config.ocr.lang,
        tool_tx,
    )
    .await
    .map_err(|e| PipelineError::ToolFailed {
        phase: "bitmap OCR".to_string(),
        source: Box::new(e),
    })?;

    let _ = forward.await;

    Ok(output_srt)
}
