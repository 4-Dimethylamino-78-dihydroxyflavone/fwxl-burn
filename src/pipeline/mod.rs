pub mod whisper;
pub mod chunked;
pub mod omnilingual;
pub mod parakeet;
pub mod salvage;
pub mod hardburn;
pub mod translate;
pub mod diagnostic;
pub mod native;
pub mod ocr;

use crate::config::AppConfig;
use crate::media::MediaFile;
use crate::srt::CueTiming;
use std::path::PathBuf;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// All available transcription modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionMode {
    /// N - Use existing embedded subtitles.
    Native,
    /// C - User provides a subtitle file.
    Custom,
    /// W - Test Whisper: Silero v6 + rnndn.
    TestWhisper,
    /// E - Extended/Smooth: Silero v4 + speech padding.
    Extended,
    /// Q - Quality: Silero v6 + Pyannote diarization + embeddings.
    Quality,
    /// I - Inclusive: Q's diarization with a lenient VAD gate, so quiet
    /// speakers and stutters that Q's stricter threshold drops still land in
    /// the transcript. Exists to A/B against Q/U on the same audio.
    Inclusive,
    /// T - Translate existing SRT to another language.
    Translate,
    /// Y - Chunked: Silence-based splitting for long-form diarization.
    Chunked,
    /// U - Unfiltered: TEN VAD, minimal processing.
    Unfiltered,
    /// P - Parakeet: NVIDIA Parakeet TDT via the eaRS `prkt` CLI.
    Parakeet,
    /// D - Parakeet + speaker diarization (deprecated; hidden from menu,
    /// retained only so legacy SRT filenames with `.D.` tags still parse).
    ParakeetDiarize,
    /// O - Omnilingual: Meta omnilingual-asr CTC, 1600+ languages.
    Omnilingual,
    /// X - Diagnostic: run all modes sequentially for comparison.
    Diagnostic,
}

impl TranscriptionMode {
    /// Single-character key for user selection.
    pub fn key(&self) -> char {
        match self {
            Self::Native => 'N',
            Self::Custom => 'C',
            Self::TestWhisper => 'W',
            Self::Extended => 'E',
            Self::Quality => 'Q',
            Self::Inclusive => 'I',
            Self::Translate => 'T',
            Self::Chunked => 'Y',
            Self::Unfiltered => 'U',
            Self::Parakeet => 'P',
            Self::ParakeetDiarize => 'D',
            Self::Omnilingual => 'O',
            Self::Diagnostic => 'X',
        }
    }

    /// Display label for the TUI menu.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Native => "Native - Use embedded subtitles",
            Self::Custom => "Custom - Drag and drop a subtitle file",
            Self::TestWhisper => "Test Whisper - Silero v6 + Diarization",
            Self::Extended => "Extended - Smooth Silero v4 + Padding",
            Self::Quality => "Quality - Diarization + RoFormer voice extraction",
            Self::Inclusive => "Inclusive - Q diarization, lenient VAD (keeps quiet speakers)",
            Self::Translate => "Translate - Translate existing SRT",
            Self::Chunked => "Chunked - Long-form silence-based splitting",
            Self::Unfiltered => "Unfiltered - Basic Faster-Whisper (low memory)",
            Self::Parakeet => "Parakeet - NVIDIA Parakeet TDT [Experimental]",
            Self::ParakeetDiarize => "Parakeet+Diarize - Parakeet + speaker diarization [Experimental]",
            Self::Omnilingual => "Omnilingual - Meta 1600+ languages CTC [Experimental]",
            Self::Diagnostic => "Diagnostic - Run ALL modes for comparison",
        }
    }

    /// Short mode tag for canonical file naming.
    pub fn mode_tag(&self) -> &'static str {
        match self {
            Self::Native => "N",
            Self::Custom => "C",
            Self::TestWhisper => "W",
            Self::Extended => "E",
            Self::Quality => "Q",
            Self::Inclusive => "I",
            Self::Translate => "T",
            Self::Chunked => "Y",
            Self::Unfiltered => "U",
            Self::Parakeet => "P",
            Self::ParakeetDiarize => "D",
            Self::Omnilingual => "O",
            Self::Diagnostic => "X",
        }
    }

    /// Whether this mode uses Whisper (vs Parakeet).
    pub fn uses_whisper(&self) -> bool {
        matches!(
            self,
            Self::TestWhisper
                | Self::Extended
                | Self::Quality
                | Self::Inclusive
                | Self::Unfiltered
                | Self::Chunked
        )
    }

    /// Whether this mode uses Parakeet.
    pub fn uses_parakeet(&self) -> bool {
        matches!(self, Self::Parakeet | Self::ParakeetDiarize)
    }

    /// Whether this mode uses the Omnilingual CTC engine.
    pub fn uses_omnilingual(&self) -> bool {
        matches!(self, Self::Omnilingual)
    }

    /// All selectable modes in menu order.
    ///
    /// `ParakeetDiarize` (D) is intentionally hidden: prkt has no built-in
    /// speaker diarization, and the old sherpa_diarize path was deleted.
    /// The variant is kept for filename-tag parsing on existing SRTs.
    pub fn all() -> &'static [TranscriptionMode] {
        &[
            Self::Native,
            Self::Custom,
            Self::TestWhisper,
            Self::Extended,
            Self::Quality,
            Self::Inclusive,
            Self::Translate,
            Self::Chunked,
            Self::Unfiltered,
            Self::Parakeet,
            Self::Omnilingual,
            Self::Diagnostic,
        ]
    }
}

/// Subtitle hardcoding style presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleStyle {
    Youtube,
    /// Youtube look, but the hardburn reads a re-timed copy of the SRT in
    /// which cues are held longer and chained (see [`CueTiming`]).
    YoutubeFlow,
    Netflix,
    Minimal,
    Default,
}

impl SubtitleStyle {
    pub fn key(&self) -> char {
        match self {
            Self::Youtube => '1',
            Self::YoutubeFlow => '2',
            Self::Netflix => '3',
            Self::Minimal => '4',
            Self::Default => '5',
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Youtube => "Youtube - Arial 16, semi-transparent",
            Self::YoutubeFlow => "Youtube Flow - Youtube look, cues held longer + chained",
            Self::Netflix => "Netflix - Consolas 14, dark background box",
            Self::Minimal => "Minimal - Arial 16, thin outline",
            Self::Default => "Default - Arial 17, outline + shadow",
        }
    }

    pub fn config_key(&self) -> &'static str {
        match self {
            Self::Youtube => "youtube",
            Self::YoutubeFlow => "youtube_flow",
            Self::Netflix => "netflix",
            Self::Minimal => "minimal",
            Self::Default => "default",
        }
    }

    /// Style whose ASS preset to borrow when the user's config predates
    /// this one and has no `[styles.presets]` entry for it.
    pub fn visual_fallback_key(&self) -> Option<&'static str> {
        match self {
            Self::YoutubeFlow => Some("youtube"),
            _ => None,
        }
    }

    /// Cue-timing rules used when the config has no `[styles.timing.<key>]`
    /// entry for this style. `None` means burn the timings as written.
    pub fn builtin_timing(&self) -> Option<CueTiming> {
        match self {
            Self::YoutubeFlow => Some(CueTiming::default()),
            _ => None,
        }
    }

    pub fn all() -> &'static [SubtitleStyle] {
        &[
            Self::Youtube,
            Self::YoutubeFlow,
            Self::Netflix,
            Self::Minimal,
            Self::Default,
        ]
    }
}

pub mod repair;

/// Shared context for pipeline execution.
pub struct PipelineContext {
    pub config: AppConfig,
    pub media: MediaFile,
    pub mode: TranscriptionMode,
    pub selected_model: String,
    pub srt_dir: PathBuf,
    pub final_out_dir: PathBuf,
    pub timestamp: String,
    pub is_batch: bool,
    pub cancel: CancellationToken,
}

/// Events emitted by pipeline stages for TUI progress display.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    PhaseStarted {
        name: String,
        phase_num: u8,
        total_phases: u8,
    },
    Progress {
        percent: f32,
        message: String,
    },
    StdoutLine(String),
    StderrLine(String),
    PhaseCompleted {
        name: String,
        duration: Duration,
    },
    Warning(String),
    Error(String),
    Completed(PipelineResult),
}

/// Result of a completed pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub srt_file: Option<PathBuf>,
    pub hardcoded_video: Option<PathBuf>,
    pub lang_code: String,
    pub mode_tag: String,
    pub warnings: Vec<String>,
}
