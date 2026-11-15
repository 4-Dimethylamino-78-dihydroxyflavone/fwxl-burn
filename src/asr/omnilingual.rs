//! Meta Omnilingual ASR (1600+ languages), in-process CTC inference.
//!
//! The sherpa-onnx export of `facebookresearch/omnilingual-asr` is a single
//! wav2vec2-style encoder with a CTC head -- dramatically simpler than
//! Parakeet's three-graph TDT stack:
//!
//! - **Input**: one f32 tensor of shape `[1, num_samples]` -- the raw 16 kHz
//!   mono waveform, mean/variance normalized over the whole chunk. There is
//!   no mel/preprocessor graph at all.
//! - **Output**: logits `[1, T, vocab]` (~50 frames/sec) plus a length tensor.
//! - **Decode**: greedy CTC -- argmax per frame, collapse repeats, drop
//!   blanks. Blank is id 0 (`<s>`), hardcoded the same way sherpa-onnx does
//!   ("for omnilingual asr, its blank id is 0").
//!
//! Contract derived from `sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model.cc`
//! and `offline-recognizer-ctc-impl.h`, and verified against the shipped
//! `tokens.txt` (character-level vocab, literal-space separator, no
//! SentencePiece markers).

use super::GpuOptions;
use crate::error::AsrError;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

/// CTC blank. The vocab's id 0 is `<s>`, which doubles as the blank in
/// wav2vec2 CTC fine-tuning; sherpa hardcodes the same.
const BLANK_ID: usize = 0;

/// Special tokens that must never surface in a transcript. `<s>` (0) is the
/// blank and is dropped by the CTC collapse; `<pad>`/`</s>`/`<unk>` are
/// dropped defensively.
const NUM_SPECIAL_TOKENS: usize = 4;

/// Resolved on-disk model artefacts. Nothing here touches the network.
#[derive(Debug, Clone)]
pub struct OmniModelFiles {
    pub model: PathBuf,
    pub tokens: PathBuf,
}

impl OmniModelFiles {
    /// Resolve from a real local directory (same semantics as the Parakeet
    /// fix: "local" means local -- no cache indirection, no network).
    ///
    /// `filename` picks the ONNX file inside the dir, e.g. `model.onnx`
    /// (fp32) or `model.int8.onnx`.
    pub fn from_dir(dir: &Path, filename: &str) -> Result<Self, AsrError> {
        let model = dir.join(filename);
        let tokens = dir.join("tokens.txt");
        for (desc, path) in [("CTC model", &model), ("tokens.txt", &tokens)] {
            if !path.is_file() {
                return Err(AsrError::ModelFileMissing {
                    description: desc.to_string(),
                    path: path.clone(),
                });
            }
        }
        Ok(Self { model, tokens })
    }

    /// Resolve from a HuggingFace repo, downloading into `cache` (or the
    /// default HF cache) when not already present.
    pub fn from_hf(repo: &str, cache: Option<&Path>, filename: &str) -> Result<Self, AsrError> {
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

        let model_repo = api.model(repo.to_string());
        let fetch = |file: &str| -> Result<PathBuf, AsrError> {
            model_repo.get(file).map_err(|e| AsrError::Download {
                repo: format!("{repo}:{file}"),
                reason: e.to_string(),
            })
        };

        Ok(Self {
            model: fetch(filename)?,
            tokens: fetch("tokens.txt")?,
        })
    }
}

/// A loaded Omnilingual model. Construct once per run, then call
/// [`OmnilingualSession::transcribe`] for each chunk.
pub struct OmnilingualSession {
    session: Session,
    vocab: Vec<String>,
}

impl OmnilingualSession {
    /// Load the CTC graph onto the given CUDA device. GPU-only; a failed
    /// provider registration is a hard error, never a CPU fallback.
    pub fn load(files: &OmniModelFiles, opts: GpuOptions) -> Result<Self, AsrError> {
        super::assert_cuda_available(opts.device_id)?;
        let session = super::build_cuda_session(&files.model, "omnilingual CTC", opts)?;
        let vocab = load_tokens(&files.tokens)?;
        Ok(Self { session, vocab })
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

        // Whole-chunk mean/variance normalization, matching sherpa's
        // NormalizeFeatures (wav2vec2's standard input preprocessing).
        let n = samples.len() as f64;
        let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = samples.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        if var <= 1e-10 {
            // Digital silence; normalizing would just amplify noise into the
            // model. There is nothing to transcribe.
            return Ok(String::new());
        }
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let normalized: Vec<f32> = samples
            .iter()
            .map(|&v| ((v as f64 - mean) * inv_std) as f32)
            .collect();

        let len = normalized.len();
        let input = Array2::from_shape_vec((1, len), normalized)
            .map_err(|e| infer_err("input shaping", e))?;

        // Destructure so the mutable borrow of `session` (held by `outputs`
        // for its whole lifetime) does not conflict with reading `vocab`
        // inside the decode loop below.
        let Self { session, vocab } = self;

        // Single positional input, exactly as sherpa feeds it. Outputs are
        // taken by index: [0] = logits (1, T, vocab), [1] = lengths.
        let outputs = session
            .run(ort::inputs![
                Tensor::from_array(input).map_err(|e| infer_err("input tensor", e))?
            ])
            .map_err(|e| infer_err("ctc forward", e))?;

        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| infer_err("logits extract", e))?;

        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(AsrError::UnexpectedShape {
                stage: "omnilingual logits".to_string(),
                shape: dims.iter().map(|&d| d.max(0) as usize).collect(),
            });
        }
        let frames = dims[1] as usize;
        let vocab_size = dims[2] as usize;
        if vocab_size < vocab.len() {
            // More vocab entries than logit columns means tokens.txt does not
            // belong to this model; transcripts would be garbage.
            return Err(AsrError::UnexpectedShape {
                stage: format!(
                    "omnilingual logits vocab axis {} < tokens.txt entries {}",
                    vocab_size,
                    vocab.len()
                ),
                shape: dims.iter().map(|&d| d.max(0) as usize).collect(),
            });
        }

        // Greedy CTC: per-frame argmax -> collapse consecutive repeats ->
        // drop blanks and special tokens. Character-level vocab with literal
        // spaces, so text assembly is plain concatenation.
        let mut text = String::new();
        let mut prev: usize = BLANK_ID;
        for t in 0..frames {
            let row = &logits[t * vocab_size..(t + 1) * vocab_size];
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            if best != prev && best >= NUM_SPECIAL_TOKENS {
                if let Some(tok) = vocab.get(best) {
                    text.push_str(tok);
                }
            }
            prev = best;
        }

        Ok(text.trim().to_string())
    }
}

fn infer_err<E: std::fmt::Display>(stage: &str, e: E) -> AsrError {
    AsrError::Inference {
        stage: format!("omnilingual {stage}"),
        reason: e.to_string(),
    }
}

/// Parse sherpa's `tokens.txt`: one `TOKEN ID` pair per line.
///
/// The split must be on the LAST space: the vocab contains a literal-space
/// token whose line reads `"  4"` (token = `" "`, id = 4). A first-field
/// parser -- like the Parakeet vocab loader -- would mis-read that line and
/// shift every space in every transcript.
fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let content = std::fs::read_to_string(path).map_err(|e| AsrError::VocabLoad {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut max_id = 0usize;
    for line in content.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let Some((token, id_str)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(id) = id_str.trim().parse::<usize>() else {
            continue;
        };
        max_id = max_id.max(id);
        pairs.push((id, token.to_string()));
    }

    if pairs.is_empty() {
        return Err(AsrError::VocabLoad {
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tokens.txt contained no `TOKEN ID` lines",
            ),
        });
    }

    let mut vocab = vec![String::new(); max_id + 1];
    for (id, token) in pairs {
        vocab[id] = token;
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_parser_survives_the_literal_space_token() {
        // Verbatim shape of the real omnilingual tokens.txt header.
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("tokens.txt");
        std::fs::write(&p, "<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n  4\na 5\nb 6\n").unwrap();
        let vocab = load_tokens(&p).expect("parses");
        assert_eq!(vocab.len(), 7);
        assert_eq!(vocab[0], "<s>");
        assert_eq!(vocab[4], " ", "token 4 is a literal space");
        assert_eq!(vocab[5], "a");
    }

    #[test]
    fn tokens_parser_rejects_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("tokens.txt");
        std::fs::write(&p, "\n\n").unwrap();
        assert!(load_tokens(&p).is_err());
    }
}
