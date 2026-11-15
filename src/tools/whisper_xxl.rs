use crate::config::AppConfig;
use crate::tools::{ToolInvocation, ToolLine, run_tool, run_tool_streaming};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;

/// Which Whisper transcription mode to use, mapping to VAD preset keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperMode {
    /// W - Test: silero_v6 + rnndn_xiph
    Test,
    /// E - Extended/Smooth: silero_v4_fw + speech padding
    Smooth,
    /// Q - Quality: silero_v6 + pyannote diarization + embeddings
    Quality,
    /// I - Inclusive: Q's diarization with a lenient VAD gate (low threshold,
    /// short-utterance capture, generous padding) so quiet speakers survive.
    Inclusive,
    /// U - Unfiltered/Lite: ten VAD, minimal processing
    Unfiltered,
    /// Y - Chunked: pyannote_v3 + roformer, used with pre-split audio
    Chunked,
    /// Default: ten + reverb_v2 diarization + roformer vocal extraction
    Default,
}

impl WhisperMode {
    /// Get the VAD preset key for this mode.
    pub fn preset_key(&self) -> &'static str {
        match self {
            Self::Test => "test",
            Self::Smooth => "smooth",
            Self::Quality => "exp2_diarize",
            Self::Inclusive => "diarize_lenient",
            Self::Unfiltered => "lite",
            Self::Chunked => "diarize",
            Self::Default => "default",
        }
    }

    /// Short mode tag for canonical SRT naming.
    pub fn mode_tag(&self) -> &'static str {
        match self {
            Self::Test => "W",
            Self::Smooth => "E",
            Self::Quality => "Q",
            Self::Inclusive => "I",
            Self::Unfiltered => "U",
            Self::Chunked => "Y",
            Self::Default => "D",
        }
    }
}

/// Build a ToolInvocation for faster-whisper-xxl.
///
/// Models: faster-whisper-xxl accepts `--model <name>` (it then looks under its own
/// `<whisper_dir>/_models/faster-whisper-<name>` tree) OR a directory override via
/// `--model_dir`. Passing a filesystem path as `--model` is INVALID — it causes
/// faster-whisper-xxl to concatenate `_models/faster-whisper-` with the path and
/// emit "Unknown model not found at ...". We therefore:
///   - Always pass `--model <name>`.
///   - If the model lives outside the default `<whisper_dir>/_models/` tree, also pass
///     `--model_dir <parent_of_model_dir>` so whisper looks there first.
pub fn build_whisper_invocation(
    config: &AppConfig,
    input: &Path,
    output_dir: &Path,
    model: &str,
    mode: WhisperMode,
) -> ToolInvocation {
    let preset_args = preset_args_for(config, mode);
    build_whisper_invocation_with_preset(config, input, output_dir, model, &preset_args)
}

/// The configured preset string for a mode, before any degradation.
///
/// Falls back to a compiled-in default when the user's config predates the
/// preset key -- running whisper with NO preset args (the old behavior for a
/// missing key) silently produced a completely different transcription.
pub fn preset_args_for(config: &AppConfig, mode: WhisperMode) -> String {
    let key = mode.preset_key();
    config
        .whisper
        .vad_presets
        .get(key)
        .cloned()
        .unwrap_or_else(|| builtin_preset_args(key).to_string())
}

/// Compiled-in preset strings, mirroring config.default.toml. Used only when
/// a key is missing from the loaded config (e.g. a config file generated
/// before the preset existed).
fn builtin_preset_args(key: &str) -> &'static str {
    match key {
        // Inclusive (I): Q's pyannote diarization, but the VAD gate opened up.
        // vad_threshold 0.60 -> 0.35 admits quiet speakers; min-speech 100ms
        // (XXL default 250) catches stutters/short interjections; 1200ms
        // padding (XXL default 900, which Q inherits) keeps soft onsets. No
        // rnndn / vocal extraction -- filtering is what eats the quiet channel
        // in the first place.
        "diarize_lenient" => {
            "--vad_method silero_v6 --vad_threshold 0.35 \
             --vad_min_speech_duration_ms 100 --vad_speech_pad_ms 1200 \
             --compression_ratio_threshold 2.9 --diarize_ff False \
             --voc_device cuda --diarize pyannote_v3.1 --beep_off --realign \
             --multilingual True --verbose true --return_embeddings --standard"
        }
        _ => "",
    }
}

/// Same as [`build_whisper_invocation`] but takes the preset string explicitly,
/// so a run that died on memory pressure can be retried with the heavy stages
/// stripped out instead of losing the whole transcription.
pub fn build_whisper_invocation_with_preset(
    config: &AppConfig,
    input: &Path,
    output_dir: &Path,
    model: &str,
    preset_args: &str,
) -> ToolInvocation {
    let resolved_dir = find_model_dir(config, model);
    let override_dir = model_dir_override(config, resolved_dir.as_deref());

    let mut args: Vec<OsString> = Vec::new();

    // Input file
    args.push(OsString::from(input));

    // Model name (never a path -- see doc comment above).
    args.push(OsString::from("--model"));
    args.push(OsString::from(model));

    // If the model is outside the default `<whisper_dir>/_models/` tree, tell
    // faster-whisper-xxl where to look via --model_dir.
    if let Some(ref dir) = override_dir {
        args.push(OsString::from("--model_dir"));
        args.push(OsString::from(dir));
    }

    // Output directory
    args.push(OsString::from("--output_dir"));
    args.push(OsString::from(output_dir));

    // Output format
    args.push(OsString::from("--output_format"));
    args.push(OsString::from("all"));

    // VAD preset arguments (split on whitespace)
    for part in preset_args.split_whitespace() {
        args.push(OsString::from(part));
    }

    ToolInvocation {
        program: config.tools.whisper.clone(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(7200)), // 2 hour timeout
    }
}

/// Build invocation for batch directory processing.
/// See doc comment on `build_whisper_invocation` for why `--model` receives a name, not a path.
pub fn build_whisper_batch_invocation(
    config: &AppConfig,
    input_dir: &Path,
    output_dir: &Path,
    model: &str,
    mode: WhisperMode,
) -> ToolInvocation {
    let preset_key = mode.preset_key();
    let preset_args = config
        .whisper
        .vad_presets
        .get(preset_key)
        .cloned()
        .unwrap_or_default();

    let resolved_dir = find_model_dir(config, model);
    let override_dir = model_dir_override(config, resolved_dir.as_deref());

    let mut args: Vec<OsString> = Vec::new();

    args.push(OsString::from(input_dir));

    args.push(OsString::from("--model"));
    args.push(OsString::from(model));

    if let Some(ref dir) = override_dir {
        args.push(OsString::from("--model_dir"));
        args.push(OsString::from(dir));
    }

    args.push(OsString::from("--output_dir"));
    args.push(OsString::from(output_dir));

    args.push(OsString::from("--output_format"));
    args.push(OsString::from("all"));

    // Batch-specific flags
    args.push(OsString::from("--batch_recursive"));
    args.push(OsString::from("--check_files"));

    for part in preset_args.split_whitespace() {
        args.push(OsString::from(part));
    }

    ToolInvocation {
        program: config.tools.whisper.clone(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(14400)), // 4 hour timeout for batch
    }
}

/// If the resolved model is NOT under the default `<whisper_dir>/_models/` tree,
/// return the parent directory (suitable for passing to `--model_dir`). Otherwise None.
fn model_dir_override(config: &AppConfig, resolved: Option<&Path>) -> Option<PathBuf> {
    let resolved = resolved?;
    let default_root = config.tools.whisper.parent()?.join("_models");
    if resolved.starts_with(&default_root) {
        None
    } else {
        resolved.parent().map(|p| p.to_path_buf())
    }
}

/// Run whisper with real-time streaming output.
pub async fn run_whisper_streaming(
    config: &AppConfig,
    input: &Path,
    output_dir: &Path,
    model: &str,
    mode: WhisperMode,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<Option<i32>, crate::error::ToolError> {
    let inv = build_whisper_invocation(config, input, output_dir, model, mode);
    run_tool_streaming(&inv, tx).await
}

/// Streaming run with an explicit preset, used for the degraded retry.
pub async fn run_whisper_streaming_with_preset(
    config: &AppConfig,
    input: &Path,
    output_dir: &Path,
    model: &str,
    preset_args: &str,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<Option<i32>, crate::error::ToolError> {
    let inv = build_whisper_invocation_with_preset(config, input, output_dir, model, preset_args);
    run_tool_streaming(&inv, tx).await
}

/// Find the model directory path. Searches a list of candidate locations so the
/// user is not required to keep models exactly in `<whisper>/_models/faster-whisper-<name>/`.
///
/// Returns the directory that actually contains the model files. Callers can use
/// this to:
///   (a) pre-flight check that the model exists before launching whisper, and
///   (b) pass `--model_dir <parent>` when the model is NOT in the default location.
pub fn find_model_dir(config: &AppConfig, model_name: &str) -> Option<PathBuf> {
    // 1. If model_name is itself an absolute path, trust it.
    let as_path = PathBuf::from(model_name);
    if as_path.is_absolute() && as_path.exists() {
        return Some(as_path);
    }

    // 2. Walk up to a few ancestor directories of the whisper binary and, at each
    //    level, try the usual `_models/` conventions. This covers both:
    //       <whisper_dir>/_models/faster-whisper-<name>
    //       <whisper_dir>/../_models/faster-whisper-<name>     (portable layouts)
    //       <whisper_dir>/<name>                              (inline unpacked)
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(p) = config.tools.whisper.parent() {
        roots.push(p.to_path_buf());
        if let Some(pp) = p.parent() {
            roots.push(pp.to_path_buf());
        }
    }
    // 3. User-requested explicit search paths. These are the Subtitle Edit
    //    companion install locations and are ALWAYS searched, regardless of
    //    where `config.tools.whisper` points. Adding both the directory and
    //    its `_models` subdirectory means the candidate loop below will try
    //    `..\_models\faster-whisper-<name>`, `..\_models\<name>`,
    //    `..\_models\_models\faster-whisper-<name>` etc. -- harmless extras.
    if let Some(appdata) = dirs::config_dir() {
        let se_root = appdata
            .join("Subtitle Edit")
            .join("Whisper")
            .join("Purfview-Whisper-Faster");
        roots.push(se_root.clone());
        roots.push(se_root.join("_models"));
    }
    // 4. Other common locations.
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".cache").join("huggingface").join("hub"));
    }
    if let Some(appdata) = dirs::config_dir() {
        roots.push(appdata.join("Purfview-Whisper-Faster"));
    }

    for root in &roots {
        let models_dir = root.join("_models");
        let candidates = [
            models_dir.join(format!("faster-whisper-{model_name}")),
            models_dir.join(model_name),
            root.join(format!("faster-whisper-{model_name}")),
            root.join(model_name),
        ];
        for c in &candidates {
            if c.is_dir() {
                return Some(c.clone());
            }
        }
    }
    None
}

/// Enumerate available whisper models by scanning the _models directory.
pub fn list_available_models(config: &AppConfig) -> Vec<String> {
    let mut models = Vec::new();

    if let Some(whisper_dir) = config.tools.whisper.parent() {
        let models_dir = whisper_dir.join("_models");
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(stripped) = name.strip_prefix("faster-whisper-") {
                        models.push(stripped.to_string());
                    } else {
                        models.push(name);
                    }
                }
            }
        }
    }

    models.sort();
    models
}
