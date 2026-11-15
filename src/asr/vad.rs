//! Silero VAD (v5 ONNX schema), GPU-resident, for utterance segmentation.
//!
//! Replaces ffmpeg `silencedetect` chunking: instead of arbitrary ~20 s
//! windows split at dB-silence, the pipeline gets real speech-utterance
//! boundaries, so each SRT entry is one utterance instead of a long
//! multi-sentence string.
//!
//! ## Why striped batching
//!
//! Silero is a tiny stateful model consuming 512-sample (32 ms) windows
//! sequentially -- naively that is ~30,000 GPU round-trips for 15 minutes of
//! audio, and per-call latency would dominate by orders of magnitude. Audio
//! is therefore split into `batch` parallel stripes processed as one batched
//! call per window index with independent LSTM states (`state` is
//! `[2, B, 128]`). Sequential call count drops by the batch factor.
//!
//! Stripe seams would reset LSTM state cold, which can smear a boundary, so
//! each stripe (except the first) re-processes `WARMUP_WINDOWS` of the
//! preceding stripe's audio to warm its state; probabilities from warmup
//! windows are discarded.
//!
//! Schema (verified against the snakers4/deepghs v5 exports and the local
//! `silero_v5/files/silero_vad.onnx`):
//!   inputs:  `input` [B, 512] f32, `state` [2, B, 128] f32, `sr` i64
//!   outputs: `output` [B, 1] f32 (speech probability), `stateN`

use super::GpuOptions;
use crate::error::AsrError;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// Silero v5 window size at 16 kHz.
const WINDOW: usize = 512;
/// 16 kHz sample rate, the only rate the chunker produces.
const SR: i64 = 16_000;
/// Windows of preceding audio re-processed to warm a stripe's LSTM state
/// (31 windows = ~1 s).
const WARMUP_WINDOWS: usize = 31;

/// Post-processing knobs, all in the audio's own timebase.
#[derive(Debug, Clone, Copy)]
pub struct VadParams {
    /// Speech-start probability threshold.
    pub threshold: f32,
    /// Segments shorter than this are dropped (ms).
    pub min_speech_ms: u64,
    /// Silence must persist this long to close a segment (ms).
    pub min_silence_ms: u64,
    /// Padding added to both ends of every segment (ms).
    pub speech_pad_ms: u64,
    /// Segments longer than this are hard-split into equal parts (s).
    pub max_segment_sec: f64,
}

pub struct VadSession {
    session: Session,
}

impl VadSession {
    /// Load the VAD graph onto the given CUDA device -- same GPU-only,
    /// fail-loud contract as the ASR engines.
    pub fn load(model: &Path, opts: GpuOptions) -> Result<Self, AsrError> {
        super::assert_cuda_available(opts.device_id)?;
        let session = super::build_cuda_session(model, "silero VAD", opts)?;
        Ok(Self { session })
    }

    /// Compute per-window speech probabilities with striped batching.
    ///
    /// Returns one probability per 512-sample window of `samples`.
    fn window_probs(&mut self, samples: &[f32], batch: usize) -> Result<Vec<f32>, AsrError> {
        let total_windows = samples.len().div_ceil(WINDOW);
        if total_windows == 0 {
            return Ok(Vec::new());
        }

        let batch = batch.clamp(1, 512).min(total_windows);
        // Windows per stripe, last stripe may run short.
        let stripe_len = total_windows.div_ceil(batch);
        let stripes: Vec<usize> = (0..batch).map(|b| b * stripe_len).collect();

        let mut probs = vec![0.0f32; total_windows];
        let mut state = Array3::<f32>::zeros((2, batch, 128));

        // Warmup: step index runs from -WARMUP_WINDOWS..stripe_len; negative
        // steps feed the tail of the previous stripe and their outputs are
        // discarded. Stripe 0 has no predecessor, so it feeds zeros (which is
        // also what a cold silero stream sees).
        let warmup = WARMUP_WINDOWS as isize;
        for step in -warmup..stripe_len as isize {
            let mut input = Array2::<f32>::zeros((batch, WINDOW));
            let mut any_live = false;
            for (b, &stripe_start) in stripes.iter().enumerate() {
                let w = stripe_start as isize + step;
                if w < 0 || w as usize >= total_windows {
                    continue;
                }
                // Warmup windows before stripe 0 fall out via w < 0; warmup
                // windows of later stripes overlap the previous stripe's tail.
                if step >= 0 || stripe_start > 0 {
                    let start = w as usize * WINDOW;
                    let end = (start + WINDOW).min(samples.len());
                    input
                        .row_mut(b)
                        .as_slice_mut()
                        .expect("row is contiguous")[..end - start]
                        .copy_from_slice(&samples[start..end]);
                    any_live = true;
                }
            }
            if !any_live {
                continue;
            }

            let outputs = self
                .session
                .run(ort::inputs![
                    Tensor::from_array(input).map_err(|e| infer_err("vad input", e))?,
                    Tensor::from_array(state.clone()).map_err(|e| infer_err("vad state", e))?,
                    Tensor::from_array(Array1::from_vec(vec![SR]))
                        .map_err(|e| infer_err("vad sr", e))?,
                ])
                .map_err(|e| infer_err("vad forward", e))?;

            let (_, out) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| infer_err("vad prob extract", e))?;
            let new_state = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| infer_err("vad state extract", e))?
                .to_owned()
                .into_dimensionality()
                .map_err(|e| infer_err("vad state shape", e))?;
            state = new_state;

            if step >= 0 {
                for (b, &stripe_start) in stripes.iter().enumerate() {
                    let w = stripe_start + step as usize;
                    if w < total_windows && (b + 1 >= stripes.len() || w < stripes[b + 1]) {
                        probs[w] = out[b];
                    }
                }
            }
        }

        Ok(probs)
    }

    /// Run VAD over a full recording and return speech segments in seconds.
    pub fn segment(
        &mut self,
        samples: &[f32],
        batch: usize,
        params: VadParams,
    ) -> Result<Vec<(f64, f64)>, AsrError> {
        let probs = self.window_probs(samples, batch)?;
        Ok(probs_to_segments(
            &probs,
            samples.len() as f64 / SR as f64,
            params,
        ))
    }
}

/// Hysteresis + padding + min/max rules, following silero's reference
/// `get_speech_timestamps` shape. Pure function for testability.
fn probs_to_segments(probs: &[f32], audio_secs: f64, params: VadParams) -> Vec<(f64, f64)> {
    let win_secs = WINDOW as f64 / SR as f64;
    let neg_threshold = (params.threshold - 0.15).max(0.01);
    let min_silence_wins = ((params.min_silence_ms as f64 / 1000.0) / win_secs).ceil() as usize;
    let min_speech_secs = params.min_speech_ms as f64 / 1000.0;
    let pad = params.speech_pad_ms as f64 / 1000.0;

    let mut raw: Vec<(f64, f64)> = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0.0f64;
    let mut silence_run = 0usize;

    for (w, &p) in probs.iter().enumerate() {
        let t = w as f64 * win_secs;
        if !in_speech {
            if p >= params.threshold {
                in_speech = true;
                seg_start = t;
                silence_run = 0;
            }
        } else if p < neg_threshold {
            silence_run += 1;
            if silence_run >= min_silence_wins {
                // Segment ended where the silence run began.
                let seg_end = t - (silence_run - 1) as f64 * win_secs;
                raw.push((seg_start, seg_end));
                in_speech = false;
                silence_run = 0;
            }
        } else {
            silence_run = 0;
        }
    }
    if in_speech {
        raw.push((seg_start, audio_secs));
    }

    // Pad, clamp, merge overlaps created by padding, drop too-short.
    let mut padded: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for (s, e) in raw {
        let s = (s - pad).max(0.0);
        let e = (e + pad).min(audio_secs);
        if e - s < min_speech_secs {
            continue;
        }
        match padded.last_mut() {
            Some(last) if s <= last.1 => last.1 = e.max(last.1),
            _ => padded.push((s, e)),
        }
    }

    // Hard-split anything longer than max_segment_sec into equal parts.
    let max_len = params.max_segment_sec.max(1.0);
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(padded.len());
    for (s, e) in padded {
        let len = e - s;
        if len <= max_len {
            out.push((s, e));
        } else {
            let parts = (len / max_len).ceil() as usize;
            let part_len = len / parts as f64;
            for i in 0..parts {
                let ps = s + i as f64 * part_len;
                let pe = if i + 1 == parts { e } else { ps + part_len };
                out.push((ps, pe));
            }
        }
    }
    out
}

fn infer_err<E: std::fmt::Display>(stage: &str, e: E) -> AsrError {
    AsrError::Inference {
        stage: stage.to_string(),
        reason: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> VadParams {
        VadParams {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 400,
            speech_pad_ms: 100,
            max_segment_sec: 15.0,
        }
    }

    /// ~32 ms per window; 100 windows ~= 3.2 s.
    fn probs(spans: &[(usize, usize)], total: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; total];
        for &(s, e) in spans {
            for v in &mut p[s..e] {
                *v = 0.9;
            }
        }
        p
    }

    #[test]
    fn detects_a_single_utterance_with_padding() {
        let p = probs(&[(10, 40)], 100);
        let segs = probs_to_segments(&p, 3.2, params());
        assert_eq!(segs.len(), 1);
        let (s, e) = segs[0];
        // 10*0.032=0.32 minus 0.1 pad; 40*0.032=1.28 plus pad.
        assert!((s - 0.22).abs() < 0.04, "start {s}");
        assert!((e - 1.38).abs() < 0.06, "end {e}");
    }

    #[test]
    fn short_blips_are_dropped_and_brief_dips_do_not_split() {
        // A 3-window blip (~96 ms < min_speech 250 ms) must vanish; a
        // 5-window dip (~160 ms < min_silence 400 ms) must not split.
        let mut p = probs(&[(10, 30), (35, 60)], 100); // dip 30..35
        p[70] = 0.9; // blip
        let segs = probs_to_segments(&p, 3.2, params());
        assert_eq!(segs.len(), 1, "{segs:?}");
    }

    #[test]
    fn long_speech_is_hard_split() {
        // 600 windows of continuous speech ~= 19.2 s > 15 s max.
        let p = probs(&[(0, 600)], 600);
        let segs = probs_to_segments(&p, 19.2, params());
        assert!(segs.len() >= 2);
        for (s, e) in &segs {
            assert!(e - s <= 15.01);
        }
        // Contiguous cover, no gaps introduced by splitting.
        assert!((segs.first().unwrap().0 - 0.0).abs() < 0.2);
        assert!((segs.last().unwrap().1 - 19.2).abs() < 0.2);
    }

    #[test]
    fn trailing_speech_reaching_eof_is_closed() {
        let p = probs(&[(80, 100)], 100);
        let segs = probs_to_segments(&p, 3.2, params());
        assert_eq!(segs.len(), 1);
        assert!((segs[0].1 - 3.2).abs() < 1e-9);
    }
}
