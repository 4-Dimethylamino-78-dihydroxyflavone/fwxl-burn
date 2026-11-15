use crate::asr::{GpuOptions, ModelFiles, ParakeetSession};
use crate::config::ParakeetModelDef;
use crate::error::PipelineError;
use crate::gpu;
use crate::pipeline::chunked::{self, ChunkTranscriber, TOTAL_PHASES};
use crate::pipeline::{PipelineContext, PipelineEvent, PipelineResult};
use std::time::Instant;
use tokio::sync::mpsc;

/// Run the Parakeet transcription pipeline entirely in-process on the GPU.
///
/// Phases 1-3 and 5-6 are the shared chunked-ASR machinery in
/// [`crate::pipeline::chunked`]; this function contributes only Phase 4 --
/// loading the Parakeet TDT stack once and streaming chunks through it.
pub async fn run_parakeet_pipeline(
    ctx: &PipelineContext,
    model: &ParakeetModelDef,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PipelineResult, PipelineError> {
    let chunked::Prepared {
        source,
        spans,
        cleanup_dir,
    } = chunked::prepare(ctx, &event_tx).await?;

    // ─── Phase 4: in-process GPU inference, model loaded once ─────────────
    let gpu_info = gpu::probe_nvidia_gpu();
    let vram_note = match &gpu_info {
        Some(info) => format!("free VRAM {}/{} MiB", info.free_mib, info.total_mib),
        None => "GPU probe unavailable".to_string(),
    };
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Transcribing {} segments in-process (model={}, precision={:?}, CUDA device {}, {vram_note})",
            spans.len(),
            model.name,
            ctx.config.prkt.precision,
            ctx.config.prkt.device_id,
        ),
        phase_num: 4,
        total_phases: TOTAL_PHASES,
    });

    let config = ctx.config.clone();
    let model_cloned = model.clone();
    let chunk_count = spans.len();

    let chunk_texts = chunked::transcribe_segments_in_process(
        move |event_tx| -> Result<Box<dyn ChunkTranscriber>, _> {
            let precision = config.prkt.precision;

            // Resolve weights: a local directory is read directly with no
            // network access; otherwise fall back to fetching the HF repo.
            let files = match model_cloned.model_dir.as_ref() {
                Some(dir) => {
                    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                        "Parakeet: loading local model from {}",
                        dir.display()
                    )));
                    ModelFiles::from_dir(dir, precision)?
                }
                None => {
                    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                        "Parakeet: resolving HF repo {} (cached after first run)",
                        model_cloned.hf_repo
                    )));
                    ModelFiles::from_hf(
                        &model_cloned.hf_repo,
                        config.prkt.model_cache_dir.as_deref(),
                        precision,
                    )?
                }
            };

            let load_start = Instant::now();
            let session = ParakeetSession::load(
                &files,
                GpuOptions {
                    device_id: config.prkt.device_id,
                    conv_search: config.prkt.conv_algo_search,
                },
            )?;
            let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                "Parakeet: model resident on CUDA device {} in {:.1}s (vocab {}), \
                 loaded once for all {chunk_count} chunks",
                config.prkt.device_id,
                load_start.elapsed().as_secs_f64(),
                session.vocab_size(),
            )));
            Ok(Box::new(session))
        },
        source,
        spans.clone(),
        &event_tx,
    )
    .await?;

    let srt_path = ctx.srt_dir.join(format!(
        "{}.{}.P.{}_{}.srt",
        ctx.media.stem, ctx.config.naming.parakeet_prefix, model.name, model.lang_code
    ));

    chunked::finalize_srt(
        &spans,
        cleanup_dir.as_deref(),
        &chunk_texts,
        srt_path,
        &model.lang_code,
        "P",
        &event_tx,
    )
}
