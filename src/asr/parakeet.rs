//! Parakeet TDT inference, loaded once and reused across chunks.
//!
//! The decode loop is a faithful port of the greedy RNN-T decoder that was
//! producing correct transcripts out-of-process; only the ONNX Runtime binding
//! (ort 1.16 -> 2.x) and the process lifetime have changed.

use crate::error::AsrError;
use ndarray::{Array1, Array2, Array3, s};
use ort::session::Session;
use ort::value::Tensor;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Audio contract for every chunk handed to [`ParakeetSession::transcribe`].
///
/// The chunker decodes to `-ar 16000 -ac 1 -c:a pcm_s16le` and slices with
/// `-c copy`, so chunks are guaranteed 16 kHz mono. Anything else is a bug
/// upstream and is rejected rather than silently resampled.
pub const SAMPLE_RATE: u32 = 16_000;

/// Weight precision of the ONNX export to load.
///
/// The upstream repos ship three encoder variants side by side. Note that
/// INT8 is deliberately absent: ONNX Runtime's CUDA provider has patchy
/// coverage for quantised ops and tends to fall back to CPU per-node, which
/// would quietly reintroduce the exact slowness this rewrite removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Full precision. ~2.3 GB of weights; the safest default and entirely
    /// affordable now that it is loaded once instead of once per chunk.
    Fp32,
    /// Half precision. ~1.2 GB and uses tensor cores, but FP16 exports of
    /// this model family have historically produced degenerate output on some
    /// runtimes, so it stays opt-in.
    Fp16,
}

impl Default for Precision {
    fn default() -> Self {
        Self::Fp32
    }
}

impl Precision {
    /// Filename infix used by the upstream ONNX exports.
    fn infix(self) -> &'static str {
        match self {
            Self::Fp32 => "",
            Self::Fp16 => ".fp16",
        }
    }
}

/// How cuDNN picks convolution algorithms when a graph is first run.
///
/// ort defaults this to `Exhaustive`, which benchmarks every candidate
/// algorithm at load time. On this model that alone cost ~190 s of startup --
/// far more than the entire transcription. Since the session is built once per
/// run and then discarded, that benchmarking can never amortise, so the
/// default here is `Heuristic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConvSearch {
    /// Benchmark every algorithm. Fastest steady-state inference, but adds
    /// minutes to load. Only worth it for very long inputs.
    Exhaustive,
    /// Pick by cuDNN's internal heuristic. Near-instant load.
    #[default]
    Heuristic,
    /// Always use the fixed `IMPLICIT_PRECOMP_GEMM` algorithm. Lowest memory.
    Fixed,
}

/// GPU knobs for session construction.
#[derive(Debug, Clone, Copy)]
pub struct GpuOptions {
    pub device_id: i32,
    pub conv_search: ConvSearch,
}

/// Fully resolved, on-disk model artefacts. Nothing here touches the network.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub preprocessor: PathBuf,
    pub encoder: PathBuf,
    pub decoder_joint: PathBuf,
    pub vocab: PathBuf,
}

impl ModelFiles {
    /// Resolve model files from a real local directory.
    ///
    /// This is the fix for a long-standing trap: the old `prkt --model-dir`
    /// flag was documented as "local model directory", but internally it was
    /// handed to `hf_hub::ApiBuilder::with_cache_dir()` -- an HF *cache root*.
    /// Pointing it at a local checkout therefore did not load those files at
    /// all; it treated the folder as an empty cache and re-downloaded ~2.3 GB
    /// from HuggingFace. Here a local directory means what it says.
    pub fn from_dir(dir: &Path, precision: Precision) -> Result<Self, AsrError> {
        let encoder = dir.join(format!("encoder-model{}.onnx", precision.infix()));
        let decoder_joint = dir.join(format!("decoder_joint-model{}.onnx", precision.infix()));
        let preprocessor = dir.join("nemo128.onnx");
        let vocab = dir.join("vocab.txt");

        for (desc, path) in [
            ("preprocessor (nemo128.onnx)", &preprocessor),
            ("encoder", &encoder),
            ("decoder+joint", &decoder_joint),
            ("vocabulary", &vocab),
        ] {
            if !path.is_file() {
                return Err(AsrError::ModelFileMissing {
                    description: desc.to_string(),
                    path: path.clone(),
                });
            }
        }

        Ok(Self {
            preprocessor,
            encoder,
            decoder_joint,
            vocab,
        })
    }

    /// Resolve model files from a HuggingFace repo, downloading into `cache`
    /// (or the default HF cache when `None`) if they are not already present.
    ///
    /// FP32 encoders store their weights in an external `.onnx.data` sidecar
    /// that ONNX Runtime loads by relative path, so it must be fetched even
    /// though it is never opened directly here.
    pub fn from_hf(repo: &str, cache: Option<&Path>, precision: Precision) -> Result<Self, AsrError> {
        let api = match cache {
            Some(dir) => hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(dir.to_path_buf())
                .build(),
            None => hf_hub::api::sync::Api::new(),
        }
        .map_err(|e| AsrError::Download {
            repo: repo.to_string(),
            reason: e.to_string(),
        })?;

        let model = api.model(repo.to_string());
        let fetch = |file: &str| -> Result<PathBuf, AsrError> {
            model.get(file).map_err(|e| AsrError::Download {
                repo: format!("{repo}:{file}"),
                reason: e.to_string(),
            })
        };

        let encoder_name = format!("encoder-model{}.onnx", precision.infix());
        let encoder = fetch(&encoder_name)?;
        if precision == Precision::Fp32 {
            let _external_weights = fetch("encoder-model.onnx.data")?;
        }

        Ok(Self {
            preprocessor: fetch("nemo128.onnx")?,
            encoder,
            decoder_joint: fetch(&format!("decoder_joint-model{}.onnx", precision.infix()))?,
            vocab: fetch("vocab.txt")?,
        })
    }
}

/// A loaded Parakeet model. Construct once per run, then call
/// [`ParakeetSession::transcribe`] for each chunk.
pub struct ParakeetSession {
    preprocessor: Session,
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    blank_id: usize,
}

impl ParakeetSession {
    /// Load all three ONNX graphs onto the given CUDA device.
    ///
    /// There is no CPU path. If the CUDA execution provider cannot register,
    /// this returns [`AsrError::CudaUnavailable`] instead of degrading, because
    /// a silent degrade is what hid a CUDA major-version mismatch for weeks.
    pub fn load(files: &ModelFiles, opts: GpuOptions) -> Result<Self, AsrError> {
        super::assert_cuda_available(opts.device_id)?;

        let preprocessor = super::build_cuda_session(&files.preprocessor, "preprocessor", opts)?;
        let encoder = super::build_cuda_session(&files.encoder, "encoder", opts)?;
        let decoder_joint = super::build_cuda_session(&files.decoder_joint, "decoder+joint", opts)?;

        let vocab = load_vocab(&files.vocab)?;
        let blank_id = vocab.len().saturating_sub(1);

        Ok(Self {
            preprocessor,
            encoder,
            decoder_joint,
            vocab,
            blank_id,
        })
    }

    /// Vocabulary size, for diagnostics.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Transcribe one 16 kHz mono chunk.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let (features, feature_lens) = self.preprocess(samples)?;
        self.decode_greedy(&features, &feature_lens)
    }

    /// Run the bundled NeMo mel-spectrogram graph over raw PCM.
    fn preprocess(&mut self, waveform: &[f32]) -> Result<(Array3<f32>, Array1<i64>), AsrError> {
        let waveforms = Array2::from_shape_vec((1, waveform.len()), waveform.to_vec())
            .map_err(|e| infer_err("preprocessor input shaping", e))?;
        let waveform_len = Array1::from_vec(vec![waveform.len() as i64]);

        let inputs = ort::inputs![
            Tensor::from_array(waveforms).map_err(|e| infer_err("preprocessor waveform", e))?,
            Tensor::from_array(waveform_len).map_err(|e| infer_err("preprocessor length", e))?,
        ];

        let outputs = self
            .preprocessor
            .run(inputs)
            .map_err(|e| infer_err("preprocessor", e))?;

        let features = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| infer_err("preprocessor features", e))?
            .to_owned()
            .into_dimensionality()
            .map_err(|e| shape_err("preprocessor features", e))?;
        let feature_lens = outputs[1]
            .try_extract_array::<i64>()
            .map_err(|e| infer_err("preprocessor feature lengths", e))?
            .to_owned()
            .into_dimensionality()
            .map_err(|e| shape_err("preprocessor feature lengths", e))?;

        Ok((features, feature_lens))
    }

    /// Greedy RNN-T / TDT decode. Ported verbatim from the working
    /// out-of-process implementation -- do not "improve" without a transcript
    /// diff, the emission rules are load-bearing.
    fn decode_greedy(
        &mut self,
        features: &Array3<f32>,
        feature_lens: &Array1<i64>,
    ) -> Result<String, AsrError> {
        // Scoped so the borrow of `self.encoder` ends here. `SessionOutputs`
        // borrows the session for as long as it lives, which would otherwise
        // block the `&self` call to `tokens_to_text` at the end of this fn.
        let (encoder_output_dyn, encoded_frames) = {
            let enc_inputs = ort::inputs![
                Tensor::from_array(features.clone())
                    .map_err(|e| infer_err("encoder features", e))?,
                Tensor::from_array(feature_lens.clone())
                    .map_err(|e| infer_err("encoder lengths", e))?,
            ];
            let encoder_outputs = self
                .encoder
                .run(enc_inputs)
                .map_err(|e| infer_err("encoder", e))?;

            let output = encoder_outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| infer_err("encoder output", e))?
                .to_owned();
            let encoded_len = encoder_outputs[1]
                .try_extract_array::<i64>()
                .map_err(|e| infer_err("encoder output length", e))?
                .to_owned();
            (output, encoded_len[[0]] as usize)
        };

        if encoder_output_dyn.shape().len() != 3 {
            return Err(AsrError::UnexpectedShape {
                stage: "encoder".to_string(),
                shape: encoder_output_dyn.shape().to_vec(),
            });
        }
        let encoder_output_3d: Array3<f32> = encoder_output_dyn
            .into_dimensionality()
            .map_err(|e| shape_err("encoder", e))?;
        // [batch, features, time] -> [time, features] for the single batch item.
        let encoder_permuted = encoder_output_3d.permuted_axes([0, 2, 1]);
        let encoder_2d = encoder_permuted.index_axis_move(ndarray::Axis(0), 0);

        const RNN_HIDDEN_SIZE: usize = 640;
        const MAX_TOKENS_PER_STEP: usize = 10;

        let mut tokens: Vec<i32> = Vec::new();
        let mut state_1 = Array3::<f32>::zeros((2, 1, RNN_HIDDEN_SIZE));
        let mut state_2 = Array3::<f32>::zeros((2, 1, RNN_HIDDEN_SIZE));
        let mut t = 0usize;
        let mut emitted_tokens = 0usize;

        while t < encoded_frames {
            let encoder_step = encoder_2d
                .slice(s![t, ..])
                .to_owned()
                .insert_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(2));

            let target_token = tokens.last().copied().unwrap_or(self.blank_id as i32);
            let targets = Array2::from_shape_vec((1, 1), vec![target_token])
                .map_err(|e| infer_err("decoder target shaping", e))?;
            let target_length = Array1::from_vec(vec![1i32]);

            let dec_inputs = ort::inputs![
                Tensor::from_array(encoder_step).map_err(|e| infer_err("decoder encoder step", e))?,
                Tensor::from_array(targets).map_err(|e| infer_err("decoder targets", e))?,
                Tensor::from_array(target_length)
                    .map_err(|e| infer_err("decoder target length", e))?,
                Tensor::from_array(state_1.clone()).map_err(|e| infer_err("decoder state 1", e))?,
                Tensor::from_array(state_2.clone()).map_err(|e| infer_err("decoder state 2", e))?,
            ];

            let outputs = self
                .decoder_joint
                .run(dec_inputs)
                .map_err(|e| infer_err("decoder+joint", e))?;

            let (_, logits) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| infer_err("joint logits", e))?;

            // The joint head emits vocabulary logits followed by TDT duration
            // logits; only the vocabulary span participates in the argmax.
            let vocab_size = self.vocab.len();
            let vocab_logits = if logits.len() > vocab_size {
                &logits[..vocab_size]
            } else {
                logits
            };

            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &val) in vocab_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            let new_state_1 = outputs[2]
                .try_extract_array::<f32>()
                .map_err(|e| infer_err("decoder new state 1", e))?
                .to_owned()
                .into_dimensionality()
                .map_err(|e| shape_err("decoder new state 1", e))?;
            let new_state_2 = outputs[3]
                .try_extract_array::<f32>()
                .map_err(|e| infer_err("decoder new state 2", e))?
                .to_owned()
                .into_dimensionality()
                .map_err(|e| shape_err("decoder new state 2", e))?;

            // State only advances on a real emission; blanks leave it untouched.
            if max_idx != self.blank_id {
                state_1 = new_state_1;
                state_2 = new_state_2;
                tokens.push(max_idx as i32);
                emitted_tokens += 1;
            }

            if max_idx == self.blank_id || emitted_tokens >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok(self.tokens_to_text(&tokens))
    }

    fn tokens_to_text(&self, tokens: &[i32]) -> String {
        let mut text = String::new();
        for &token_id in tokens {
            let idx = token_id as usize;
            if idx >= self.vocab.len() {
                continue;
            }
            let token = &self.vocab[idx];
            // SentencePiece marks word starts with U+2581 (3 bytes in UTF-8).
            if token == "\u{2581}" {
                text.push(' ');
            } else if let Some(rest) = token.strip_prefix('\u{2581}') {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(rest);
            } else {
                text.push_str(token);
            }
        }
        text.trim().to_string()
    }
}

/// Generic over the error type so ort and ndarray failures share one variant.
fn infer_err<E: std::fmt::Display>(stage: &str, e: E) -> AsrError {
    AsrError::Inference {
        stage: stage.to_string(),
        reason: e.to_string(),
    }
}

fn shape_err(stage: &str, e: ndarray::ShapeError) -> AsrError {
    AsrError::UnexpectedShape {
        stage: format!("{stage}: {e}"),
        shape: Vec::new(),
    }
}

fn load_vocab(path: &Path) -> Result<Vec<String>, AsrError> {
    let file = File::open(path).map_err(|e| AsrError::VocabLoad {
        path: path.to_path_buf(),
        source: e,
    })?;
    let reader = BufReader::new(file);
    let mut vocab = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| AsrError::VocabLoad {
            path: path.to_path_buf(),
            source: e,
        })?;
        if let Some(token) = line.split_whitespace().next() {
            vocab.push(token.to_string());
        }
    }
    Ok(vocab)
}

/// Read a chunk produced by the ffmpeg chunker as mono `f32` samples.
///
/// The chunker's output contract is 16 kHz mono PCM; a mismatch means the
/// upstream ffmpeg invocation changed, and is reported rather than papered
/// over with an implicit resample.
pub fn load_chunk_16k_mono(path: &Path) -> Result<Vec<f32>, AsrError> {
    let mut reader = hound::WavReader::open(path).map_err(|e| AsrError::AudioLoad {
        path: path.to_path_buf(),
        source: Box::new(e),
    })?;
    let spec = reader.spec();

    if spec.sample_rate != SAMPLE_RATE || spec.channels != 1 {
        return Err(AsrError::UnexpectedAudioFormat {
            path: path.to_path_buf(),
            rate: spec.sample_rate,
            channels: spec.channels,
        });
    }

    let samples: Result<Vec<f32>, hound::Error> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect()
        }
    };

    samples.map_err(|e| AsrError::AudioLoad {
        path: path.to_path_buf(),
        source: Box::new(e),
    })
}
