use crate::error::PipelineError;
use crate::media::SubStreamType;
use crate::pipeline::{PipelineContext, PipelineEvent, PipelineResult};
use crate::probe::embedded;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Native mode: extract the best-ranked embedded TEXT subtitle from the
/// media file to an SRT in `srt_dir`. Returns a PipelineResult populated
/// with the extracted SRT so the SrtPicker auto-selects it. If the file
/// has no text subtitle streams (or only PGS/VOBSUB bitmap streams), this
/// surfaces a clear error -- bitmap subtitles need OCR, not extraction.
pub async fn run_native_pipeline(
    ctx: &PipelineContext,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PipelineResult, PipelineError> {
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: "Extracting embedded subtitle".to_string(),
        phase_num: 1,
        total_phases: 1,
    });

    let text_streams: Vec<_> = ctx
        .media
        .embedded_subs
        .iter()
        .filter(|s| s.stream_type == SubStreamType::Text)
        .cloned()
        .collect();

    if text_streams.is_empty() {
        // Distinguish "no subs at all" vs "only bitmap subs" so the user
        // knows whether to switch modes (Whisper transcribe) or run OCR.
        let bitmap_count = ctx
            .media
            .embedded_subs
            .iter()
            .filter(|s| matches!(s.stream_type, SubStreamType::Pgs | SubStreamType::Vobsub))
            .count();
        let msg = if bitmap_count > 0 {
            format!(
                "Found {bitmap_count} bitmap (PGS/VOBSUB) subtitle stream(s) -- Native mode \
                 needs TEXT subs. Use the OCR mode or transcribe with Whisper instead."
            )
        } else {
            "No embedded subtitle streams found in this file".to_string()
        };
        let _ = event_tx.send(PipelineEvent::Error(msg));
        return Err(PipelineError::NoSrtOutput);
    }

    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Found {} text subtitle stream(s)",
        text_streams.len()
    )));
    for s in &text_streams {
        let title = s.title.as_deref().unwrap_or("");
        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "  stream {} lang={} codec={} score={} {}",
            s.index,
            s.lang,
            s.codec,
            s.score,
            if title.is_empty() {
                String::new()
            } else {
                format!("title=\"{title}\"")
            }
        )));
    }

    let extracted: PathBuf = embedded::extract_best_subtitle(
        &ctx.config,
        &ctx.media.path,
        &text_streams,
        &ctx.srt_dir,
        &ctx.media.stem,
    )
    .await
    .map_err(|e| PipelineError::ToolFailed {
        phase: "subtitle extraction".to_string(),
        source: Box::new(e),
    })?
    .ok_or(PipelineError::NoSrtOutput)?;

    // Pre-flight verify: an extraction can succeed (exit 0) but produce a
    // 0-byte file when the chosen stream is malformed. Surface that here so
    // the SrtPicker doesn't auto-select an empty file.
    let v = crate::srt::verify_srt(&extracted);
    if !v.valid {
        let _ = event_tx.send(PipelineEvent::Error(format!(
            "Extracted SRT failed verification: {} ({})",
            extracted.display(),
            v.message
        )));
        return Err(PipelineError::SrtParse(v.message));
    }

    // Pick the extracted stream's language tag for the result. We picked the
    // highest-scoring stream; that's what extract_best_subtitle did too.
    let best = text_streams
        .iter()
        .max_by_key(|s| s.score)
        .expect("non-empty checked above");
    let lang = best.lang.clone();

    let _ = event_tx.send(PipelineEvent::PhaseCompleted {
        name: format!(
            "Extracted embedded subtitle ({}): {} ({})",
            lang,
            extracted.display(),
            v.message
        ),
        duration: std::time::Duration::ZERO,
    });

    Ok(PipelineResult {
        srt_file: Some(extracted),
        hardcoded_video: None,
        lang_code: lang,
        mode_tag: "N".to_string(),
        warnings: Vec::new(),
    })
}
