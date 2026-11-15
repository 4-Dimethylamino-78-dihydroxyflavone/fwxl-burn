use crate::error::PipelineError;
use crate::pipeline::{PipelineContext, PipelineEvent, PipelineResult, TranscriptionMode};
use crate::pipeline::whisper::run_whisper_pipeline;
use tokio::sync::mpsc;

/// Diagnostic mode: run multiple transcription modes sequentially.
/// Sequences: U -> W -> Q -> P -> D
pub async fn run_diagnostic(
    ctx: &PipelineContext,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<Vec<PipelineResult>, PipelineError> {
    let modes = [
        TranscriptionMode::Unfiltered,
        TranscriptionMode::TestWhisper,
        TranscriptionMode::Quality,
        // Parakeet modes would be added here once implemented
    ];

    let mut results = Vec::new();

    for (i, &mode) in modes.iter().enumerate() {
        let _ = event_tx.send(PipelineEvent::PhaseStarted {
            name: format!("Diagnostic: running mode {} ({}/{})", mode.mode_tag(), i + 1, modes.len()),
            phase_num: (i + 1) as u8,
            total_phases: modes.len() as u8,
        });

        // Create a modified context with the diagnostic mode
        let diag_ctx = PipelineContext {
            config: ctx.config.clone(),
            media: ctx.media.clone(),
            mode,
            selected_model: ctx.selected_model.clone(),
            srt_dir: ctx.srt_dir.clone(),
            final_out_dir: ctx.final_out_dir.clone(),
            timestamp: ctx.timestamp.clone(),
            is_batch: false,
            cancel: ctx.cancel.clone(),
        };

        match run_whisper_pipeline(&diag_ctx, event_tx.clone()).await {
            Ok(result) => results.push(result),
            Err(e) => {
                let _ = event_tx.send(PipelineEvent::Warning(format!(
                    "Mode {} failed: {e}",
                    mode.mode_tag()
                )));
            }
        }
    }

    Ok(results)
}
