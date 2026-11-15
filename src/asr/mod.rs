//! In-process, GPU-only speech recognition.
//!
//! This module replaces the previous out-of-process design, in which the
//! pipeline shelled out to `prkt.exe` once per audio chunk. That shape had two
//! fatal properties:
//!
//! 1. Every chunk paid a full model load (~2.3 GB of FP32 ONNX weights). A
//!    15-minute file at the default 20 s chunk size meant 45 sequential model
//!    loads, and up to four of those ran concurrently -- four identical copies
//!    of the same weights resident at once.
//! 2. `prkt` pinned `ort` v1.16.3, i.e. ONNX Runtime 1.16.3, whose CUDA
//!    execution provider is built against CUDA 11 + cuDNN 8. On a CUDA 12 /
//!    cuDNN 9 machine the provider silently failed to register and every run
//!    executed on the CPU while still reporting "Using CUDA execution".
//!
//! Here the model is loaded exactly once per run and every chunk streams
//! through the same sessions, and CUDA registration is a hard requirement --
//! see [`crate::error::AsrError::CudaUnavailable`].

pub mod omnilingual;
pub mod parakeet;
pub mod vad;

pub use parakeet::{ConvSearch, GpuOptions, ModelFiles, ParakeetSession, Precision};

use crate::error::AsrError;
use ort::environment::Environment;
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::ep::{CUDA, ExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;
use std::sync::OnceLock;

/// The process-wide ONNX Runtime environment.
///
/// Since ort 2.0.0-rc.14 every session is built against an explicit
/// [`Environment`], and `ort::init().build()` succeeds exactly once per
/// process -- a second call fails with "only one environment is allowed per
/// process", even after the first environment has been dropped. One run loads
/// several sessions (VAD, then the ASR graphs) and the TUI runs many files per
/// process, so the environment is created on first use and kept for the life
/// of the process.
///
/// No execution providers are registered on the environment itself: each
/// session registers CUDA explicitly with `.error_on_failure()`, so there is
/// no environment-level default that could quietly put a session on the CPU.
fn environment() -> Result<&'static Environment, AsrError> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    // Two threads racing here would both call `build()`, and the loser would
    // get ort's "only one environment" error. Sessions are loaded from one
    // pipeline task at a time, but serialise anyway so that can never surface
    // as a bogus model-load failure.
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .with_name("fwxl-burn")
        .build()
        .map_err(|e| AsrError::Runtime {
            reason: e.to_string(),
        })?;
    Ok(ENV.get_or_init(|| env))
}

/// Probe CUDA availability with a precise error before any session is built.
///
/// Distinguishes "the provider isn't in this build" from a later, vaguer
/// session failure. This only reports whether ONNX Runtime was *compiled*
/// with the CUDA provider; whether its runtime libraries actually load is
/// proven by `.error_on_failure()` in [`build_cuda_session`].
pub(crate) fn assert_cuda_available() -> Result<(), AsrError> {
    match CUDA::is_available() {
        Ok(true) => Ok(()),
        Ok(false) => Err(AsrError::CudaUnavailable {
            reason: "ONNX Runtime reports no CUDA execution provider in this build".to_string(),
        }),
        Err(e) => Err(AsrError::CudaUnavailable {
            reason: format!("availability probe failed: {e}"),
        }),
    }
}

/// Build one ONNX session pinned to CUDA. Shared by every ASR engine.
///
/// `.error_on_failure()` is the load-bearing call: without it, a provider
/// that fails to register is skipped and the session quietly runs on CPU --
/// which is exactly how a CUDA-version mismatch stayed invisible for months.
pub(crate) fn build_cuda_session(
    path: &Path,
    description: &str,
    opts: GpuOptions,
) -> Result<Session, AsrError> {
    let cuda = CUDA::default()
        .with_device_id(opts.device_id)
        .with_conv_algorithm_search(match opts.conv_search {
            ConvSearch::Exhaustive => ConvAlgorithmSearch::Exhaustive,
            ConvSearch::Heuristic => ConvAlgorithmSearch::Heuristic,
            ConvSearch::Fixed => ConvAlgorithmSearch::Default,
        })
        .build()
        .error_on_failure();

    // Builder methods return `Error<SessionBuilder>` (recoverable, carrying
    // the builder back) while `commit_from_file` returns a plain `ort::Error`,
    // so the stages are flattened to messages rather than chained with `?`.
    let env = environment()?;

    let built = (|| -> Result<Session, String> {
        let builder = Session::builder(env).map_err(|e| e.to_string())?;
        let builder = builder
            .with_execution_providers([cuda])
            .map_err(|e| e.to_string())?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?;
        builder.commit_from_file(path).map_err(|e| e.to_string())
    })();

    built.map_err(|reason| {
        // A registration failure surfaces here; report it as a CUDA problem
        // rather than a generic model-load problem.
        if reason.contains("CUDA") || reason.contains("cuda") {
            AsrError::CudaUnavailable { reason }
        } else {
            AsrError::ModelLoad {
                description: description.to_string(),
                path: path.to_path_buf(),
                reason,
            }
        }
    })
}
