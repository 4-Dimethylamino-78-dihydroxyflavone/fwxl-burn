pub use crate::error::ConfigError;
use crate::srt::CueTiming;
pub use crate::tools::ffmpeg::VideoEncoder;
use crate::tools::locate::{self, SearchContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Top-level application configuration, loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Output directory for transcriptions and hardcoded videos.
    /// If empty/None, saves to the input file's folder.
    pub output_dir: Option<PathBuf>,

    /// Create and use a `\source` subfolder for SRT outputs.
    #[serde(default = "default_true")]
    pub use_source_folder: bool,

    /// FFmpeg Constant Quality value (0-51, lower = better). Default: 19.
    #[serde(default = "default_cq")]
    pub cq: u8,

    /// Hardware video encoder used for the hardburn. Defaults per platform
    /// (NVENC on Windows/Linux, VideoToolbox on macOS).
    #[serde(default)]
    pub encoder: VideoEncoder,

    /// Tool binary paths. Every entry is optional: unset tools are discovered
    /// by [`resolve_tools`] (beside the executable, then on PATH).
    #[serde(default)]
    pub tools: ToolPaths,

    /// Whisper transcription settings.
    pub whisper: WhisperConfig,

    /// Parakeet settings, plus the GPU/chunking knobs shared by every
    /// in-process chunked ASR engine (Parakeet and Omnilingual).
    pub prkt: PrktConfig,

    /// Omnilingual (Meta 1600+ languages CTC) model definitions. Optional --
    /// existing configs without an [omni] section still parse; O mode then
    /// reports "no models configured".
    #[serde(default)]
    pub omni: OmniConfig,

    /// Output file naming.
    #[serde(default)]
    pub naming: NamingConfig,

    /// Subtitle hardcoding styles.
    #[serde(default)]
    pub styles: StyleConfig,

    /// Translation settings.
    #[serde(default)]
    pub translation: TranslationConfig,

    /// PGS/VOBSUB OCR settings.
    #[serde(default)]
    pub ocr: OcrConfig,

    /// Automatic repair of unreadable input containers.
    #[serde(default)]
    pub audio_repair: AudioRepairConfig,

    /// Batch/queue behavior (skip already-transcribed items, recover
    /// quarantined runs). Optional so configs predating the section parse.
    #[serde(default)]
    pub batch: BatchConfig,
}

/// Batch/queue behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Skip a queued item when a canonical SRT for the same mode + model
    /// already exists in its output directory. Multi-item queues only.
    #[serde(default = "default_true")]
    pub skip_existing: bool,
    /// Rebuild the SRT from a `<stem>.fwxl-recovery*` folder left by a failed
    /// run instead of re-transcribing, when one is salvageable.
    #[serde(default = "default_true")]
    pub recover_quarantine: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            skip_existing: true,
            recover_quarantine: true,
        }
    }
}

/// Paths to the external tools. On disk every key is optional; an empty
/// `whisper` / `ffmpeg` / `ffprobe` means "not pinned" and is filled in by
/// [`resolve_tools`] right after loading, so the rest of the program can keep
/// treating them as plain paths.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolPaths {
    /// Path to the faster-whisper-xxl executable. Stays empty when it is
    /// neither pinned nor discoverable; the Whisper modes then report
    /// themselves unavailable instead of blocking the whole app.
    #[serde(default)]
    pub whisper: PathBuf,
    /// Path to the ffmpeg executable.
    #[serde(default)]
    pub ffmpeg: PathBuf,
    /// Path to the ffprobe executable.
    #[serde(default)]
    pub ffprobe: PathBuf,
    /// Path to ffplay (optional).
    pub ffplay: Option<PathBuf>,
    /// Path to alass-cli (optional, for subtitle alignment).
    pub alass: Option<PathBuf>,
    /// Path to SubtitleEdit (optional, prefers SubtitleEditCmd).
    pub subtitle_edit: Option<PathBuf>,
    /// Path to fwxl-audiofix (optional). When present, an unreadable
    /// container is repaired automatically instead of failing the run.
    pub audiofix: Option<PathBuf>,
}

impl ToolPaths {
    /// Whether a faster-whisper-xxl binary was pinned or discovered.
    pub fn has_whisper(&self) -> bool {
        !self.whisper.as_os_str().is_empty()
    }
}

/// Automatic repair of unreadable input containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRepairConfig {
    /// Attempt repair when ffprobe cannot read the input.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Subdirectory (relative to the media file) for repaired copies. Empty
    /// means "beside the original". The original is never modified either way.
    #[serde(default)]
    pub out_subdir: Option<String>,
}

impl Default for AudioRepairConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            out_subdir: Some("repaired".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Default model name, e.g. "large-v2".
    #[serde(default = "default_model")]
    pub default_model: String,

    /// Default model folder name, e.g. "faster-whisper-large-v2".
    #[serde(default = "default_model_folder")]
    pub default_model_folder: String,

    /// Named VAD preset argument strings.
    /// Keys: "default", "test", "smooth", "exp1_vad", "exp2_diarize", "diarize", "lite"
    pub vad_presets: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrktConfig {
    /// CUDA device index. Parakeet runs in-process and is GPU-only; there is
    /// deliberately no device *kind* setting, because the CPU fallback that
    /// used to live here is what silently masked a CUDA runtime mismatch.
    #[serde(default)]
    pub device_id: i32,
    /// Weight precision of the ONNX export to load. `fp32` (~2.3 GB) is the
    /// safe default; `fp16` halves VRAM and uses tensor cores but has a
    /// history of degenerate output in this model family.
    #[serde(default)]
    pub precision: crate::asr::Precision,
    /// cuDNN convolution algorithm selection. `exhaustive` benchmarks every
    /// candidate at load time -- on this model that cost ~190s of startup,
    /// dwarfing the transcription itself, and it cannot amortise because the
    /// session is discarded at the end of the run. `heuristic` is the default.
    #[serde(default)]
    pub conv_algo_search: crate::asr::ConvSearch,

    // ── Segmentation (shared by every chunked ASR engine) ────────────────
    /// How audio is cut into transcription units:
    ///  - "vad": Silero VAD on the GPU finds real utterance boundaries; each
    ///    SRT entry is one utterance with tight timestamps. Default.
    ///  - "silence": legacy ffmpeg silencedetect ~target_chunk_sec windows;
    ///    entries are long multi-sentence strings.
    #[serde(default)]
    pub segmentation: Segmentation,
    /// Silero VAD ONNX (v5 schema: input/state/sr). When the file is missing,
    /// `vad_hf_repo` is fetched instead.
    #[serde(default = "default_vad_model")]
    pub vad_model: PathBuf,
    /// HF fallback repo holding `silero_vad.onnx`.
    #[serde(default = "default_vad_hf_repo")]
    pub vad_hf_repo: String,
    /// Speech probability threshold (silence closes below threshold - 0.15).
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
    /// Drop detected speech shorter than this (ms).
    #[serde(default = "default_vad_min_speech_ms")]
    pub vad_min_speech_ms: u64,
    /// Silence must persist this long to end an utterance (ms).
    #[serde(default = "default_vad_min_silence_ms")]
    pub vad_min_silence_ms: u64,
    /// Padding added to both ends of every utterance (ms).
    #[serde(default = "default_vad_speech_pad_ms")]
    pub vad_speech_pad_ms: u64,
    /// Hard-split utterances longer than this (s); also the SRT entry cap.
    ///
    /// 7 s is both the subtitle-readability norm and, empirically, inside the
    /// omnilingual 300M export's safe zone: fed ~13 s segments it drops
    /// characters and emits garbage tokens, while <=6 s segments transcribe
    /// perfectly (verified on looped continuous speech). Keep below ~10 s.
    #[serde(default = "default_vad_max_segment_sec")]
    pub vad_max_segment_sec: f64,
    /// Parallel stripes batched per GPU call. Higher = fewer round-trips;
    /// each stripe costs one extra second of warmup re-processing.
    #[serde(default = "default_vad_batch")]
    pub vad_batch: u32,
    /// Optional override for the HuggingFace cache directory used when a
    /// model is fetched by `hf_repo`. Defaults to
    /// `%USERPROFILE%\.cache\huggingface\hub`. Ignored when the selected
    /// model sets `model_dir`, which is read directly with no network access.
    pub model_cache_dir: Option<PathBuf>,
    /// Target chunk duration in seconds. prkt emits no timestamps, so
    /// each chunk becomes ONE SRT entry -- small values give utterance-
    /// sized entries comparable to a Whisper VAD pass.
    #[serde(default = "default_target_chunk")]
    pub target_chunk_sec: u64,
    /// Tolerance around target chunk size when picking silence boundaries.
    #[serde(default = "default_chunk_tolerance")]
    pub chunk_tolerance_sec: u64,
    /// Silence-detect noise floor in dB.
    #[serde(default = "default_silence_db")]
    pub silence_db: f64,
    /// Minimum silence duration (s) to register as a candidate split point.
    #[serde(default = "default_silence_min")]
    pub silence_min_sec: f64,
    /// Hardware accelerator for the master-WAV decode pass. "cuda" uses
    /// NVDEC on NVIDIA. Empty or None disables. ffmpeg silently ignores
    /// `-hwaccel` on pure-audio inputs, so leaving "cuda" on is safe.
    #[serde(default = "default_hwaccel")]
    pub hwaccel: Option<String>,
    // NOTE: the former `max_parallel` / `vram_per_instance_mib` /
    // `parallel_ceiling` / `vram_reserve_mib` knobs are gone. They existed to
    // budget VRAM across several concurrent `prkt.exe` processes, each of
    // which held its own copy of the same ~2.3 GB of weights. The model is now
    // loaded once per run and chunks stream through it, so there is no
    // per-instance budget to tune. Stale keys in an existing config.toml are
    // ignored harmlessly.
    /// Run the Phase 1 master-WAV decode at BELOW_NORMAL Windows priority
    /// so the audio decode (which is inherently CPU-bound -- ffmpeg has no
    /// GPU audio codec) yields to foreground apps. Same total CPU work,
    /// but the desktop stays responsive on long files. No-op on non-Windows.
    #[serde(default = "default_true")]
    pub low_priority_decode: bool,
    /// Available Parakeet model definitions.
    pub models: Vec<ParakeetModelDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParakeetModelDef {
    /// Internal name, e.g. "parakeet-tdt-0.6b-v3".
    pub name: String,
    /// Display name for TUI menu.
    pub display: String,
    /// HuggingFace repo to fetch the ONNX export from, e.g.
    /// "istupakov/parakeet-tdt-0.6b-v3-onnx". Used only when `model_dir` is
    /// unset. Required unless `model_dir` is set.
    #[serde(default)]
    pub hf_repo: String,
    /// Directory holding the ONNX export on disk (`nemo128.onnx`,
    /// `encoder-model*.onnx`, `decoder_joint-model*.onnx`, `vocab.txt`).
    ///
    /// When set, the files are read straight from here with no network access
    /// and `hf_repo` is ignored. This genuinely means "local model directory"
    /// -- the old `prkt --model-dir` flag of the same name was in fact an HF
    /// *cache root*, so pointing it at a checkout re-downloaded ~2.3 GB
    /// instead of reading the files sitting right there.
    pub model_dir: Option<PathBuf>,
    /// Language code, e.g. "en".
    pub lang_code: String,
    /// Mark as broken so the UI grays it out without removing it from the menu.
    #[serde(default)]
    pub broken: bool,
}

/// Audio segmentation strategy for chunked ASR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Segmentation {
    /// Silero VAD utterance boundaries (GPU, in-process).
    #[default]
    Vad,
    /// Legacy ffmpeg silencedetect windows around `target_chunk_sec`.
    Silence,
}

/// Omnilingual (Meta 1600+ languages CTC) settings. GPU/device and chunking
/// knobs are shared with `[prkt]` -- both engines run the same chunked
/// pipeline; only the model definitions differ.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OmniConfig {
    /// Available Omnilingual model definitions.
    #[serde(default)]
    pub models: Vec<OmniModelDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniModelDef {
    /// Internal name, e.g. "omni-300m-ctc".
    pub name: String,
    /// Display name for TUI menu.
    pub display: String,
    /// HuggingFace repo to fetch from. Used only when `model_dir` is unset.
    #[serde(default)]
    pub hf_repo: String,
    /// Directory holding `tokens.txt` plus the ONNX file on disk. Read
    /// directly with no network access; wins over `hf_repo` if both are set.
    pub model_dir: Option<PathBuf>,
    /// ONNX filename inside the model dir / repo: `model.onnx` (fp32) or
    /// `model.int8.onnx`. int8 is NOT recommended on the GPU-only path --
    /// quantized ops have patchy CUDA coverage and fall back per-node.
    #[serde(default = "default_omni_filename")]
    pub filename: String,
    /// Language tag for SRT naming. The model is multilingual with no
    /// language conditioning, so "multi" unless a run is known-monolingual.
    #[serde(default = "default_omni_lang")]
    pub lang_code: String,
    /// Mark as broken so the UI grays it out without removing it from the menu.
    #[serde(default)]
    pub broken: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingConfig {
    /// Jellyfin prefix for transcriptions.
    #[serde(default = "default_transcribe_prefix")]
    pub transcribe_prefix: String,
    /// Jellyfin prefix for translations.
    #[serde(default = "default_translate_prefix")]
    pub translate_prefix: String,
    /// Prefix for Parakeet transcriptions.
    #[serde(default = "default_parakeet_prefix")]
    pub parakeet_prefix: String,
    /// Prefix for Omnilingual transcriptions.
    #[serde(default = "default_omnilingual_prefix")]
    pub omnilingual_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConfig {
    /// Default style name: "youtube", "youtube_flow", "netflix", "minimal", "default".
    #[serde(default = "default_style_name")]
    pub default_style: String,
    /// ASS force_style strings keyed by style name.
    pub presets: HashMap<String, String>,
    /// Optional cue-timing rules keyed by style name. A style listed here
    /// burns from a re-timed temporary copy of the SRT (the file on disk is
    /// never modified); styles without an entry burn the timings as written.
    /// `youtube_flow` has built-in defaults even when absent here, so older
    /// user configs keep working.
    #[serde(default)]
    pub timing: HashMap<String, CueTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Path to the translation Python script.
    pub script_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrConfig {
    /// Path to the PGS OCR Python script.
    pub script_path: Option<PathBuf>,
    /// OCR language (default: "eng").
    #[serde(default = "default_ocr_lang")]
    pub lang: String,
}

// -- Defaults --

fn default_true() -> bool {
    true
}
fn default_cq() -> u8 {
    19
}
fn default_model() -> String {
    "large-v2".to_string()
}
fn default_model_folder() -> String {
    "faster-whisper-large-v2".to_string()
}
fn default_target_chunk() -> u64 {
    20
}
fn default_chunk_tolerance() -> u64 {
    10
}
fn default_silence_db() -> f64 {
    -30.0
}
fn default_silence_min() -> f64 {
    1.5
}
fn default_hwaccel() -> Option<String> {
    // Matches the platform's hardware decoder, like `VideoEncoder::default`.
    Some(if cfg!(target_os = "macos") { "videotoolbox" } else { "cuda" }.to_string())
}
fn default_transcribe_prefix() -> String {
    "FWXL-Transcribe".to_string()
}
fn default_translate_prefix() -> String {
    "FWXL-Translate".to_string()
}
fn default_parakeet_prefix() -> String {
    "Parakeet-Transcribe".to_string()
}
fn default_omnilingual_prefix() -> String {
    "Omnilingual-Transcribe".to_string()
}
fn default_omni_filename() -> String {
    "model.onnx".to_string()
}
fn default_vad_model() -> PathBuf {
    PathBuf::from("Faster-Whisper-XXL/_xxl_data/faster_whisper/assets/silero_v5/files/silero_vad.onnx")
}
fn default_vad_hf_repo() -> String {
    "deepghs/silero-vad-onnx".to_string()
}
fn default_vad_threshold() -> f32 {
    0.5
}
fn default_vad_min_speech_ms() -> u64 {
    250
}
fn default_vad_min_silence_ms() -> u64 {
    400
}
fn default_vad_speech_pad_ms() -> u64 {
    150
}
fn default_vad_max_segment_sec() -> f64 {
    7.0
}
fn default_vad_batch() -> u32 {
    64
}
fn default_omni_lang() -> String {
    "multi".to_string()
}
fn default_style_name() -> String {
    "youtube".to_string()
}
fn default_ocr_lang() -> String {
    "eng".to_string()
}

// -- Default impls for optional sections --

impl Default for NamingConfig {
    fn default() -> Self {
        Self {
            transcribe_prefix: default_transcribe_prefix(),
            translate_prefix: default_translate_prefix(),
            parakeet_prefix: default_parakeet_prefix(),
            omnilingual_prefix: default_omnilingual_prefix(),
        }
    }
}

impl Default for StyleConfig {
    fn default() -> Self {
        let mut presets = HashMap::new();
        let youtube = "FontName=Arial,FontSize=16.5,PrimaryColour=&H80FFFFFF,OutlineColour=&H40000000,BackColour=&H40000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=8".to_string();
        presets.insert("youtube".to_string(), youtube.clone());
        // Same look as youtube; the difference is the timing policy below.
        presets.insert("youtube_flow".to_string(), youtube);
        presets.insert(
            "netflix".to_string(),
            "FontName=Consolas,FontSize=14,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=18".to_string(),
        );
        presets.insert(
            "minimal".to_string(),
            "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0.5,MarginV=20".to_string(),
        );
        presets.insert(
            "default".to_string(),
            "FontName=Arial,FontSize=17,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=0.5,Shadow=0.5,MarginV=14".to_string(),
        );
        let mut timing = HashMap::new();
        timing.insert("youtube_flow".to_string(), CueTiming::default());
        Self {
            default_style: default_style_name(),
            presets,
            timing,
        }
    }
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self { script_path: None }
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            script_path: None,
            lang: default_ocr_lang(),
        }
    }
}

/// Load config from TOML file.
pub fn load_config(path: &Path) -> Result<AppConfig, ConfigError> {
    let content = std::fs::read_to_string(path)?;
    let config: AppConfig = toml::from_str(&content).map_err(|e| ConfigError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(config)
}

/// The config compiled into the binary. This is what `--init` writes out and
/// what a run with no config file on disk uses, so a fresh download works with
/// zero setup as long as ffmpeg is discoverable.
pub const DEFAULT_CONFIG_TOML: &str = include_str!("../config.default.toml");

/// Parse the built-in default config.
pub fn builtin_config() -> Result<AppConfig, ConfigError> {
    toml::from_str(DEFAULT_CONFIG_TOML).map_err(|e| ConfigError::Parse {
        path: PathBuf::from("<built-in defaults>"),
        source: e,
    })
}

/// Search for config file in standard locations.
pub fn find_config(cli_path: Option<&Path>) -> Result<PathBuf, ConfigError> {
    let mut searched = Vec::new();

    // 1. CLI argument. An explicit --config that does not exist is an error:
    //    quietly loading some *other* config instead would run the job with
    //    settings the user did not ask for.
    if let Some(p) = cli_path {
        if p.exists() {
            return Ok(p.to_path_buf());
        }
        return Err(ConfigError::ExplicitNotFound {
            path: p.to_path_buf(),
        });
    }

    // 2. Next to the executable
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.with_file_name("fwxl-burn.toml");
        if sibling.exists() {
            return Ok(sibling);
        }
        searched.push(sibling);
    }

    // 3. Per-user config dir: %APPDATA%\fwxl-burn\config.toml on Windows,
    //    ~/.config/fwxl-burn/config.toml on Linux,
    //    ~/Library/Application Support/fwxl-burn/config.toml on macOS.
    if let Some(app_data) = dirs::config_dir() {
        let appdata_config = app_data.join("fwxl-burn").join("config.toml");
        if appdata_config.exists() {
            return Ok(appdata_config);
        }
        searched.push(appdata_config);
    }

    // 4. Current working directory
    let cwd_config = PathBuf::from("fwxl-burn.toml");
    if cwd_config.exists() {
        return Ok(cwd_config);
    }
    searched.push(cwd_config);

    Err(ConfigError::NotFound { searched })
}

/// Well-known install locations for faster-whisper-xxl beyond the generic
/// search: Subtitle Edit downloads it into its own per-user data folder.
fn whisper_extra_dirs() -> Vec<PathBuf> {
    dirs::config_dir()
        .map(|d| {
            vec![d
                .join("Subtitle Edit")
                .join("Whisper")
                .join("Purfview-Whisper-Faster")]
        })
        .unwrap_or_default()
}

/// Pin down one tool: a configured path must exist (never silently replaced);
/// an unset one is discovered. `Ok(None)` means "unset and not found".
fn resolve_one(
    configured: &Path,
    tool: &str,
    stems: &[&str],
    ctx: &SearchContext,
    extra: &[PathBuf],
) -> Result<Option<PathBuf>, ConfigError> {
    if configured.as_os_str().is_empty() {
        return Ok(locate::discover(stems, ctx, extra));
    }
    match locate::resolve_configured(configured, ctx) {
        Some(p) => Ok(Some(p)),
        None => Err(ConfigError::ToolNotFound {
            tool: tool.to_string(),
            path: configured.to_path_buf(),
        }),
    }
}

/// Resolve every tool and data path in `config` against the real environment.
/// `config_path` is the loaded config file (None when running on built-in
/// defaults). Returns non-fatal warnings to show the user.
pub fn resolve_tools(
    config: &mut AppConfig,
    config_path: Option<&Path>,
) -> Result<Vec<String>, ConfigError> {
    let ctx = SearchContext::from_env(config_path);
    resolve_tools_in(config, &ctx)
}

/// [`resolve_tools`] against an explicit [`SearchContext`].
///
/// ffmpeg and ffprobe are mandatory -- nothing works without them. A missing
/// faster-whisper-xxl only disables the Whisper modes, since the in-process
/// engines (Parakeet, Omnilingual) and the hardburn do not need it.
pub fn resolve_tools_in(
    config: &mut AppConfig,
    ctx: &SearchContext,
) -> Result<Vec<String>, ConfigError> {
    let mut warnings = Vec::new();

    for (tool, slot) in [
        ("ffmpeg", &mut config.tools.ffmpeg),
        ("ffprobe", &mut config.tools.ffprobe),
    ] {
        match resolve_one(slot, tool, &[tool], ctx, &[])? {
            Some(p) => *slot = p,
            None => {
                return Err(ConfigError::ToolNotDiscovered {
                    tool: tool.to_string(),
                });
            }
        }
    }

    match resolve_one(
        &config.tools.whisper,
        "faster-whisper-xxl",
        &["faster-whisper-xxl"],
        ctx,
        &whisper_extra_dirs(),
    )? {
        Some(p) => config.tools.whisper = p,
        None => warnings.push(
            "faster-whisper-xxl was not found: the Whisper modes are unavailable \
             (Parakeet, Omnilingual and hardburn still work). Put it on PATH, unpack \
             it into a Faster-Whisper-XXL folder beside fwxl-burn, or set \
             tools.whisper in the config."
                .to_string(),
        ),
    }

    // Optional paths: only re-anchor relative ones that are missing from the
    // working directory but present beside the config or the executable. A
    // path that exists nowhere is left exactly as written so the eventual
    // error names what the user typed.
    let rebase = |p: &mut PathBuf| {
        if let Some(found) = locate::resolve_configured(p, ctx) {
            *p = found;
        }
    };
    for slot in [
        &mut config.tools.ffplay,
        &mut config.tools.alass,
        &mut config.tools.subtitle_edit,
        &mut config.tools.audiofix,
        &mut config.translation.script_path,
        &mut config.ocr.script_path,
    ] {
        if let Some(p) = slot.as_mut() {
            rebase(p);
        }
    }
    rebase(&mut config.prkt.vad_model);
    for dir in config
        .prkt
        .models
        .iter_mut()
        .map(|m| &mut m.model_dir)
        .chain(config.omni.models.iter_mut().map(|m| &mut m.model_dir))
    {
        if let Some(p) = dir.as_mut() {
            rebase(p);
        }
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::locate::exe_name;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    /// The config shipped inside the binary must always parse, must not pin
    /// any machine-specific tool path, and must not leak a home directory.
    #[test]
    fn builtin_config_parses_and_is_machine_neutral() {
        let cfg = builtin_config().expect("config.default.toml must parse");
        assert!(cfg.tools.ffmpeg.as_os_str().is_empty());
        assert!(cfg.tools.ffprobe.as_os_str().is_empty());
        assert!(!cfg.tools.has_whisper());
        assert!(!cfg.prkt.models.is_empty());
        assert!(!cfg.omni.models.is_empty());
        for needle in ["C:/Users", "C:\\Users", "/home/", "D:/"] {
            assert!(
                !DEFAULT_CONFIG_TOML.contains(needle),
                "config.default.toml contains a machine-specific path ({needle})"
            );
        }
    }

    #[test]
    fn unset_tools_are_discovered_and_missing_whisper_only_warns() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join(exe_name("ffmpeg")));
        touch(&tmp.path().join(exe_name("ffprobe")));
        let ctx = SearchContext {
            path_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        };
        let mut cfg = builtin_config().unwrap();
        let warnings = resolve_tools_in(&mut cfg, &ctx).unwrap();
        assert_eq!(cfg.tools.ffmpeg, tmp.path().join(exe_name("ffmpeg")));
        assert_eq!(cfg.tools.ffprobe, tmp.path().join(exe_name("ffprobe")));
        // The Subtitle Edit folder may genuinely hold a copy on a dev machine;
        // only assert the warning when discovery really came up empty.
        if !cfg.tools.has_whisper() {
            assert_eq!(warnings.len(), 1);
        }
    }

    #[test]
    fn missing_ffmpeg_is_fatal() {
        let mut cfg = builtin_config().unwrap();
        let err = resolve_tools_in(&mut cfg, &SearchContext::default()).unwrap_err();
        assert!(matches!(err, ConfigError::ToolNotDiscovered { ref tool } if tool == "ffmpeg"));
    }

    /// A pinned path that does not exist must fail loudly, even when a
    /// perfectly good copy is sitting on PATH.
    #[test]
    fn pinned_but_missing_tool_is_not_substituted() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join(exe_name("ffmpeg")));
        touch(&tmp.path().join(exe_name("ffprobe")));
        let ctx = SearchContext {
            path_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        };
        let mut cfg = builtin_config().unwrap();
        cfg.tools.ffmpeg = tmp.path().join("gone").join(exe_name("ffmpeg"));
        let err = resolve_tools_in(&mut cfg, &ctx).unwrap_err();
        assert!(matches!(err, ConfigError::ToolNotFound { ref tool, .. } if tool == "ffmpeg"));
    }
}
