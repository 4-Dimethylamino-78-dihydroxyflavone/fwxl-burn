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
use ort::ep::CUDA;
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::execution_providers::ExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;

/// Probe CUDA availability with a precise error before any session is built.
///
/// Distinguishes "the provider isn't in this build" / "its DLLs won't load"
/// from a later, vaguer session failure.
pub(crate) fn assert_cuda_available(device_id: i32) -> Result<(), AsrError> {
    let probe = CUDA::default().with_device_id(device_id);
    match probe.is_available() {
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
    let built = (|| -> Result<Session, String> {
        let builder = Session::builder().map_err(|e| e.to_string())?;
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
