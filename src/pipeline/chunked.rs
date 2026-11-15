//! Shared machinery for chunked in-process ASR pipelines.
//!
//! Parakeet (P) and Omnilingual (O) run the same six-phase shape; only Phase 4
//! -- which model is loaded and how audio becomes text -- differs:
//!
//!   1. ffmpeg single-decode source -> master.wav (NVDEC for video sources)
//!   2. segmentation analysis (Silero VAD on GPU, or ffmpeg silencedetect)
//!   3. segment planning / slicing
//!   4. load the model ONCE, stream every segment through it  <- engine-specific
//!   5. build SRT entries (one per segment)
//!   6. write SRT
//!
//! Two segmentation strategies exist behind [`prepare`]:
//!
//! - **VAD** (default): Silero VAD runs in-process on the GPU over the master
//!   WAV held in memory. Segments are real utterances, so each SRT entry is
//!   one utterance with tight timestamps instead of a ~20 s multi-sentence
//!   string. No per-segment WAV files are written at all -- slices are fed to
//!   the engine straight from RAM, and silence never reaches the GPU.
//! - **Silence** (legacy): ffmpeg silencedetect picks ~`target_chunk_sec`
//!   windows and the master WAV is re-split into chunk files with `-c copy`.

use crate::asr::parakeet::load_chunk_16k_mono;
use crate::asr::vad::{VadParams, VadSession};
use crate::asr::GpuOptions;
use crate::chunk;
use crate::config::Segmentation;
use crate::error::{AsrError, PipelineError};
use crate::pipeline::{PipelineContext, PipelineEvent, PipelineResult};
use crate::srt;
use crate::tools::ffmpeg;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Total phases displayed for every chunked ASR pipeline.
pub const TOTAL_PHASES: u8 = 6;

/// Anything that turns one 16 kHz mono segment into text. Implemented by each
/// in-process ASR session; `Send` because inference runs on a blocking thread.
pub trait ChunkTranscriber: Send {
    fn transcribe_chunk(&mut self, samples: &[f32]) -> Result<String, AsrError>;
}

impl ChunkTranscriber for crate::asr::ParakeetSession {
    fn transcribe_chunk(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        self.transcribe(samples)
    }
}

impl ChunkTranscriber for crate::asr::omnilingual::OmnilingualSession {
    fn transcribe_chunk(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        self.transcribe(samples)
    }
}

/// Where segment audio comes from during Phase 4.
pub enum SegmentSource {
    /// One WAV file per segment (silence path).
    Files(Vec<PathBuf>),
    /// The whole master recording in RAM; segments are index ranges (VAD path).
    Memory(Vec<f32>),
}

/// Result of phases 1-3.
pub struct Prepared {
    pub source: SegmentSource,
    /// Segment time spans in seconds; SRT entry i covers spans[i].
    pub spans: Vec<(f64, f64)>,
    /// Directory to sweep after a successful SRT write (silence path only).
    pub cleanup_dir: Option<PathBuf>,
}

/// Phases 1-3, dispatching on the configured segmentation strategy.
pub async fn prepare(
    ctx: &PipelineContext,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<Prepared, PipelineError> {
    match ctx.config.prkt.segmentation {
        Segmentation::Vad => prepare_vad(ctx, event_tx).await,
        Segmentation::Silence => prepare_silence(ctx, event_tx).await,
    }
}

/// Decode the source to a 16 kHz mono master WAV (Phase 1). Shared by both
/// strategies.
async fn decode_master_wav(
    ctx: &PipelineContext,
    chunk_dir: &std::path::Path,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PathBuf, PipelineError> {
    // NVDEC handles H.264/HEVC/AV1 if hwaccel="cuda" and the source is video;
    // ffmpeg ignores -hwaccel on audio-only sources so the flag is harmless.
    let hwaccel = if ctx.media.is_video() {
        ctx.config.prkt.hwaccel.as_deref()
    } else {
        None
    };
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Decoding source -> master WAV (16kHz mono, hwaccel={})",
            hwaccel.unwrap_or("(none)"),
        ),
        phase_num: 1,
        total_phases: TOTAL_PHASES,
    });

    let master_wav = chunk_dir.join(format!("{}_master.wav", ctx.media.stem));
    ffmpeg::resample_to_master_wav(&ctx.config, &ctx.media.path, &master_wav, hwaccel)
        .await
        .map_err(|e| PipelineError::ToolFailed {
            phase: "master WAV decode".to_string(),
            source: Box::new(e),
        })?;

    let master_size = std::fs::metadata(&master_wav).map(|m| m.len()).unwrap_or(0);
    let _ = event_tx.send(PipelineEvent::Progress {
        percent: 10.0,
        message: format!(
            "Master WAV ready ({:.1} MB)",
            master_size as f64 / 1_048_576.0
        ),
    });
    Ok(master_wav)
}

/// VAD strategy: master WAV -> RAM -> Silero on GPU -> utterance spans.
async fn prepare_vad(
    ctx: &PipelineContext,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<Prepared, PipelineError> {
    let chunk_dir = ctx.srt_dir.join(format!("prkt_chunks_{}", ctx.media.stem));
    std::fs::create_dir_all(&chunk_dir)
        .map_err(|e| PipelineError::ChunkingFailed(format!("Cannot create chunk dir: {e}")))?;

    let master_wav = decode_master_wav(ctx, &chunk_dir, event_tx).await?;

    // Pull the whole recording into RAM (16 kHz mono f32 = ~230 MB/hour) and
    // drop the on-disk copies immediately -- the VAD path never writes
    // per-segment files.
    let samples = load_chunk_16k_mono(&master_wav).map_err(PipelineError::Asr)?;
    let _ = std::fs::remove_file(&master_wav);
    let _ = std::fs::remove_dir_all(&chunk_dir);
    let audio_secs = samples.len() as f64 / 16_000.0;

    let cfg = &ctx.config.prkt;
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Silero VAD on GPU (threshold={}, min_silence={}ms, batch={} stripes)",
            cfg.vad_threshold, cfg.vad_min_silence_ms, cfg.vad_batch,
        ),
        phase_num: 2,
        total_phases: TOTAL_PHASES,
    });

    // Resolve the VAD model: local file if present, else the HF mirror.
    let vad_model = if ctx.config.prkt.vad_model.is_file() {
        ctx.config.prkt.vad_model.clone()
    } else {
        let repo = ctx.config.prkt.vad_hf_repo.clone();
        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "VAD model not found at {}; fetching {repo} (cached after first run)",
            ctx.config.prkt.vad_model.display()
        )));
        let cache = ctx.config.prkt.model_cache_dir.clone();
        let api = match cache {
            Some(dir) => hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(dir)
                .build(),
            None => hf_hub::api::sync::Api::new(),
        }
        .map_err(|e| {
            PipelineError::Asr(AsrError::Download {
                repo: repo.clone(),
                reason: e.to_string(),
            })
        })?;
        api.model(repo.clone())
            .get("silero_vad.onnx")
            .map_err(|e| {
                PipelineError::Asr(AsrError::Download {
                    repo,
                    reason: e.to_string(),
                })
            })?
    };

    let params = VadParams {
        threshold: cfg.vad_threshold,
        min_speech_ms: cfg.vad_min_speech_ms,
        min_silence_ms: cfg.vad_min_silence_ms,
        speech_pad_ms: cfg.vad_speech_pad_ms,
        max_segment_sec: cfg.vad_max_segment_sec,
    };
    let gpu = GpuOptions {
        device_id: cfg.device_id,
        conv_search: cfg.conv_algo_search,
    };
    let batch = cfg.vad_batch as usize;

    // VAD inference is synchronous; keep it off the tokio worker.
    let vad_start = Instant::now();
    let (samples, spans) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<f32>, Vec<(f64, f64)>), AsrError> {
            let mut session = VadSession::load(&vad_model, gpu)?;
            let spans = session.segment(&samples, batch, params)?;
            Ok((samples, spans))
        },
    )
    .await
    .map_err(|join_err| PipelineError::ChunkingFailed(format!("VAD task panicked: {join_err}")))?
    .map_err(PipelineError::Asr)?;

    let speech_secs: f64 = spans.iter().map(|(s, e)| e - s).sum();
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Planning {} utterance segments ({speech_secs:.0}s speech of {audio_secs:.0}s audio, VAD took {:.1}s)",
            spans.len(),
            vad_start.elapsed().as_secs_f64(),
        ),
        phase_num: 3,
        total_phases: TOTAL_PHASES,
    });
    let _ = event_tx.send(PipelineEvent::Progress {
        percent: 35.0,
        message: format!("{} utterances found", spans.len()),
    });

    Ok(Prepared {
        source: SegmentSource::Memory(samples),
        spans,
        cleanup_dir: None,
    })
}

/// Legacy silence strategy: ffmpeg silencedetect windows + `-c copy` re-split.
async fn prepare_silence(
    ctx: &PipelineContext,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<Prepared, PipelineError> {
    let duration = ctx
        .media
        .duration
        .map(|d| d.as_secs_f64())
        .unwrap_or(999999.0);

    let target_chunk = ctx.config.prkt.target_chunk_sec as f64;
    let tolerance = ctx.config.prkt.chunk_tolerance_sec as f64;
    let silence_db = ctx.config.prkt.silence_db;
    let silence_min = ctx.config.prkt.silence_min_sec;

    let chunk_dir = ctx.srt_dir.join(format!("prkt_chunks_{}", ctx.media.stem));
    std::fs::create_dir_all(&chunk_dir)
        .map_err(|e| PipelineError::ChunkingFailed(format!("Cannot create chunk dir: {e}")))?;

    let master_wav = decode_master_wav(ctx, &chunk_dir, event_tx).await?;

    // ─── Phase 2: silence detection on master.wav ─────────────────────────
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Detecting silence on master WAV (noise={silence_db}dB, min={silence_min}s)"
        ),
        phase_num: 2,
        total_phases: TOTAL_PHASES,
    });

    let silence_stderr = ffmpeg::detect_silence(&ctx.config, &master_wav, silence_db, silence_min)
        .await
        .map_err(|e| PipelineError::ToolFailed {
            phase: "silence detection".to_string(),
            source: Box::new(e),
        })?;
    let silences = chunk::parse_silence_detect(&silence_stderr);

    let min_chunk = (target_chunk / 4.0).max(3.0);
    let split_points =
        chunk::compute_split_points(&silences, duration, target_chunk, tolerance, min_chunk);
    let total_chunks = split_points.len() + 1;
    let _ = event_tx.send(PipelineEvent::Progress {
        percent: 25.0,
        message: format!(
            "Found {} silence gaps; planning {total_chunks} chunks (~{target_chunk:.0}s each)",
            silences.len(),
        ),
    });

    let mut offsets = vec![0.0f64];
    offsets.extend_from_slice(&split_points);
    offsets.push(duration);

    // ─── Phase 3: split master.wav at silence points (one ffmpeg call) ────
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!("Slicing master WAV into {total_chunks} chunks (-c copy)"),
        phase_num: 3,
        total_phases: TOTAL_PHASES,
    });
    let wav_chunks = ffmpeg::split_master_wav(
        &ctx.config,
        &master_wav,
        &split_points,
        &chunk_dir,
        &ctx.media.stem,
    )
    .await
    .map_err(|e| PipelineError::ToolFailed {
        phase: "WAV segment split".to_string(),
        source: Box::new(e),
    })?;
    let _ = event_tx.send(PipelineEvent::Progress {
        percent: 35.0,
        message: format!("Split into {} chunks", wav_chunks.len()),
    });

    // Master WAV is no longer needed once chunks exist.
    let _ = std::fs::remove_file(&master_wav);

    let spans: Vec<(f64, f64)> = offsets.windows(2).map(|w| (w[0], w[1])).collect();

    Ok(Prepared {
        source: SegmentSource::Files(wav_chunks),
        spans,
        cleanup_dir: Some(chunk_dir),
    })
}

/// Phase 4 driver: load an engine once on a blocking thread, then stream
/// every segment through it in order.
///
/// ONNX Runtime sessions are not `Sync` and inference needs `&mut self`, so
/// segments run sequentially. One resident model saturates the GPU far better
/// than N processes that each spent most of their life loading weights. The
/// blocking thread keeps synchronous inference off the tokio worker driving
/// the TUI.
///
/// `loader` builds the session and may emit its own progress lines; an
/// [`AsrError::CudaUnavailable`] from any segment aborts the run -- there is
/// no CPU to limp along on, and a silent half-transcript would be worse than
/// an error that names the problem.
pub async fn transcribe_segments_in_process<L>(
    loader: L,
    source: SegmentSource,
    spans: Vec<(f64, f64)>,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<Vec<String>, PipelineError>
where
    L: FnOnce(&mpsc::UnboundedSender<PipelineEvent>) -> Result<Box<dyn ChunkTranscriber>, AsrError>
        + Send
        + 'static,
{
    let event_tx = event_tx.clone();

    let joined = tokio::task::spawn_blocking(move || -> Result<Vec<String>, AsrError> {
        let mut engine = loader(&event_tx)?;

        let total = spans.len();
        let mut texts: Vec<String> = vec![String::new(); total];
        let run_start = Instant::now();
        let mut audio_secs_done = 0.0f64;

        for (i, &(start, end)) in spans.iter().enumerate() {
            let seg_secs = (end - start).max(0.0);

            // A single unreadable segment should cost one subtitle entry, not
            // the entire run.
            let owned: Vec<f32>;
            let samples: &[f32] = match &source {
                SegmentSource::Files(wavs) => match wavs.get(i).map(|w| load_chunk_16k_mono(w)) {
                    Some(Ok(s)) => {
                        owned = s;
                        &owned
                    }
                    Some(Err(e)) => {
                        let _ = event_tx.send(PipelineEvent::Warning(format!(
                            "Segment {}/{total} unreadable: {e}",
                            i + 1
                        )));
                        continue;
                    }
                    None => continue,
                },
                SegmentSource::Memory(all) => {
                    let a = ((start * 16_000.0) as usize).min(all.len());
                    let b = ((end * 16_000.0) as usize).clamp(a, all.len());
                    &all[a..b]
                }
            };

            match engine.transcribe_chunk(samples) {
                Ok(text) => {
                    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                        "  Segment {}/{total}: {} chars ({start:.1}s..{end:.1}s)",
                        i + 1,
                        text.chars().count(),
                    )));
                    texts[i] = text;
                }
                Err(e @ AsrError::CudaUnavailable { .. }) => {
                    // The GPU went away mid-run; stop rather than emit a
                    // silent half-transcript.
                    return Err(e);
                }
                Err(e) => {
                    let _ = event_tx.send(PipelineEvent::Warning(format!(
                        "Segment {}/{total} failed: {e}",
                        i + 1
                    )));
                }
            }

            audio_secs_done += seg_secs;
            let elapsed = run_start.elapsed().as_secs_f64().max(0.001);
            let pct = 35.0 + 50.0 * ((i + 1) as f32 / total.max(1) as f32);
            let _ = event_tx.send(PipelineEvent::Progress {
                percent: pct,
                message: format!(
                    "Transcribed {}/{total} ({:.1}x realtime)",
                    i + 1,
                    audio_secs_done / elapsed
                ),
            });
        }

        let elapsed = run_start.elapsed().as_secs_f64().max(0.001);
        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "ASR: {total} segments / {audio_secs_done:.0}s audio in {elapsed:.1}s \
             ({:.1}x realtime)",
            audio_secs_done / elapsed
        )));

        Ok(texts)
    })
    .await;

    match joined {
        Ok(Ok(texts)) => Ok(texts),
        Ok(Err(e)) => Err(PipelineError::Asr(e)),
        Err(join_err) => Err(PipelineError::ChunkingFailed(format!(
            "Transcription task panicked: {join_err}"
        ))),
    }
}

/// Phases 5-6: assemble segment texts into SRT entries and write the file.
///
/// One entry per non-empty segment; entry bounds are the (padded) segment
/// bounds. Any cleanup dir is removed only after the SRT is verified
/// non-empty.
pub fn finalize_srt(
    spans: &[(f64, f64)],
    cleanup_dir: Option<&std::path::Path>,
    texts: &[String],
    srt_path: PathBuf,
    lang_code: &str,
    mode_tag: &str,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PipelineResult, PipelineError> {
    // ─── Phase 5: build SRT entries from segment texts ────────────────────
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: "Building SRT from segment texts".to_string(),
        phase_num: 5,
        total_phases: TOTAL_PHASES,
    });

    let mut entries: Vec<srt::SrtEntry> = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let t = text.trim();
        if t.is_empty() {
            continue;
        }
        let (start, end) = spans[i];
        entries.push(srt::SrtEntry {
            index: 0,
            start: Duration::from_secs_f64(start.max(0.0)),
            end: Duration::from_secs_f64(end.max(start)),
            text: t.to_string(),
        });
    }
    for (i, e) in entries.iter_mut().enumerate() {
        e.index = (i + 1) as u32;
    }

    // ─── Phase 6: write SRT ───────────────────────────────────────────────
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: "Writing final SRT".to_string(),
        phase_num: 6,
        total_phases: TOTAL_PHASES,
    });

    if entries.is_empty() {
        let _ = event_tx.send(PipelineEvent::Warning(
            "No speech detected across all segments".to_string(),
        ));
        let fallback = vec![srt::SrtEntry {
            index: 1,
            start: Duration::from_secs(0),
            end: Duration::from_secs(1),
            text: "[Silence/Error]".to_string(),
        }];
        srt::write_srt_file(&fallback, &srt_path)
            .map_err(|e| PipelineError::SrtParse(format!("Failed to write SRT: {e}")))?;
    } else {
        srt::write_srt_file(&entries, &srt_path)
            .map_err(|e| PipelineError::SrtParse(format!("Failed to write SRT: {e}")))?;
    }

    let srt_size = std::fs::metadata(&srt_path).map(|m| m.len()).unwrap_or(0);
    if srt_size == 0 {
        return Err(PipelineError::NoSrtOutput);
    }

    if let Some(dir) = cleanup_dir {
        let _ = std::fs::remove_dir_all(dir);
    }

    let _ = event_tx.send(PipelineEvent::PhaseCompleted {
        name: format!(
            "{mode_tag}: {} entries from {} segments, SRT {srt_size} bytes",
            entries.len(),
            spans.len(),
        ),
        duration: Duration::ZERO,
    });

    Ok(PipelineResult {
        srt_file: Some(srt_path),
        hardcoded_video: None,
        lang_code: lang_code.to_string(),
        mode_tag: mode_tag.to_string(),
        warnings: if entries.is_empty() {
            vec!["No speech detected".to_string()]
        } else {
            Vec::new()
        },
    })
}
