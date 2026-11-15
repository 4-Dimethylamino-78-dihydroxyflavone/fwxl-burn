use std::path::PathBuf;

/// Top-level application error types.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    #[error("TUI error: {0}")]
    Tui(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Configuration loading and validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Config file not found (searched: {searched:?}). Run with --init to generate a default config.")]
    NotFound { searched: Vec<PathBuf> },

    #[error("Failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },

    #[error("Config file not found: {path} (given with --config)")]
    ExplicitNotFound { path: PathBuf },

    #[error("Tool not found: {tool} (expected at {path})")]
    ToolNotFound { tool: String, path: PathBuf },

    #[error(
        "Could not find {tool}. Install it so it is on PATH, place it beside \
         fwxl-burn, or set tools.{tool} in the config."
    )]
    ToolNotDiscovered { tool: String },

    #[error("Invalid config value: {field} -- {reason}")]
    Validation { field: String, reason: String },

    #[error("IO error reading config: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from external tool invocations.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("{tool} failed with exit code {code}: {stderr}")]
    NonZeroExit {
        tool: String,
        code: i32,
        stderr: String,
    },

    #[error("{tool} was killed by signal")]
    Killed { tool: String },

    #[error("{tool} timed out after {timeout_secs}s")]
    Timeout { tool: String, timeout_secs: u64 },

    #[error("Failed to spawn {tool}: {source}")]
    Spawn {
        tool: String,
        source: std::io::Error,
    },

    #[error("Missing output from {tool}: expected {description} at {path}")]
    MissingOutput {
        tool: String,
        description: String,
        path: PathBuf,
    },

    #[error("Empty output from {tool}: {description} at {path} is 0 bytes")]
    EmptyOutput {
        tool: String,
        description: String,
        path: PathBuf,
    },
}

/// What to check when the CUDA provider will not load, per platform. The
/// library names and the search rules differ enough that one generic message
/// would send people looking for files that do not exist on their OS.
fn cuda_runtime_hint() -> &'static str {
    if cfg!(target_os = "windows") {
        "Verify these are loadable: cudart64_12.dll, cublas64_12.dll, \
         cufft64_11.dll, cudnn64_9.dll.\n\
         Note that Windows does NOT search subdirectories of PATH entries, so \
         cuDNN 9 installs that keep their DLLs in bin/<cuda-major>/ are not \
         findable until those DLLs are copied up into a directory on PATH."
    } else if cfg!(target_os = "macos") {
        "CUDA does not exist on macOS, so the in-process engines (Parakeet, \
         Omnilingual) cannot run here yet."
    } else {
        "Verify the CUDA 12 and cuDNN 9 runtimes are installed and visible to \
         the dynamic loader: libcudart.so.12, libcublas.so.12, libcufft.so.11, \
         libcudnn.so.9 (check `ldconfig -p | grep -E 'cudart|cudnn'`, or add \
         their directory to LD_LIBRARY_PATH)."
    }
}

/// Errors from the in-process Parakeet ASR engine.
///
/// `CudaUnavailable` is deliberately fatal. This pipeline is GPU-only, and a
/// silent CPU fallback is precisely the failure mode that hid a whole CUDA
/// major-version mismatch: the old out-of-process `prkt.exe` printed
/// "Using CUDA execution", ran entirely on the CPU, and exited 0.
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error(
        "CUDA execution provider failed to register: {reason}\n\
         Parakeet is GPU-only; refusing to silently fall back to CPU.\n{}",
        cuda_runtime_hint()
    )]
    CudaUnavailable { reason: String },

    #[error("Model file missing: {description} at {path}")]
    ModelFileMissing { description: String, path: PathBuf },

    // These carry `reason: String` rather than a typed source because ort's
    // builder methods return a *recoverable* `Error<SessionBuilder>` that
    // differs from the plain `ort::Error` returned elsewhere. Flattening to a
    // message keeps one variant per failure site instead of one per error type.
    #[error("Failed to load model {description} from {path}: {reason}")]
    ModelLoad {
        description: String,
        path: PathBuf,
        reason: String,
    },

    #[error("Inference failed during {stage}: {reason}")]
    Inference { stage: String, reason: String },

    #[error("Model download failed for repo {repo}: {reason}")]
    Download { repo: String, reason: String },

    #[error("Failed to read audio {path}: {source}")]
    AudioLoad {
        path: PathBuf,
        source: Box<hound::Error>,
    },

    #[error(
        "Audio {path} is {rate} Hz / {channels}ch, but the chunker guarantees \
         16 kHz mono PCM. Refusing to resample silently."
    )]
    UnexpectedAudioFormat {
        path: PathBuf,
        rate: u32,
        channels: u16,
    },

    #[error("Unexpected tensor shape from {stage}: {shape:?}")]
    UnexpectedShape { stage: String, shape: Vec<usize> },

    #[error("Failed to read vocabulary at {path}: {source}")]
    VocabLoad {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Pipeline-level errors (orchestration logic).
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("No SRT output produced after transcription")]
    NoSrtOutput,

    /// A run finished but yielded nothing usable, *after* rebuilding from
    /// sibling formats and (for memory failures) one reduced-settings retry
    /// were both tried. `detail` names where the raw output was preserved --
    /// this error never implies the work was thrown away.
    #[error("Transcription unusable: {reason} ({detail})")]
    TranscriptionUnusable { reason: String, detail: String },

    #[error("Parakeet ASR error: {0}")]
    Asr(#[from] AsrError),

    #[error("Transcription mode {mode} is not available: {reason}")]
    ModeUnavailable { mode: String, reason: String },

    #[error("Media file not found: {0}")]
    MediaNotFound(PathBuf),

    #[error("Unsupported media format: {0}")]
    UnsupportedFormat(String),

    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),

    #[error("SRT parsing error: {0}")]
    SrtParse(String),

    #[error("Cancelled by user")]
    Cancelled,

    /// The re-timed temporary SRT a flow-style hardburn reads could not be
    /// written. Deliberately fatal rather than silently burning the
    /// original timings: the user chose that style for its timing.
    #[error("Could not stage subtitle for hardburn: {0}")]
    HardburnStaging(String),

    #[error("Tool error during {phase}: {source}")]
    ToolFailed {
        phase: String,
        source: Box<ToolError>,
    },
}
