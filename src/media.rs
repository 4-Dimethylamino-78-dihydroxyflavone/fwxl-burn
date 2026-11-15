use std::path::{Path, PathBuf};
use std::time::Duration;

/// Classified media type based on file extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaType {
    Video,
    Audio,
    Subtitle,
}

/// An embedded subtitle stream discovered via FFprobe.
#[derive(Debug, Clone)]
pub struct EmbeddedSubStream {
    pub index: u32,
    pub codec: String,
    pub lang: String,
    pub stream_type: SubStreamType,
    pub is_default: bool,
    pub is_forced: bool,
    pub is_sdh: bool,
    pub title: Option<String>,
    pub score: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubStreamType {
    Text,
    Pgs,
    Vobsub,
    Unknown,
}

/// Represents an input media file with probed metadata.
#[derive(Debug, Clone)]
pub struct MediaFile {
    pub path: PathBuf,
    pub dir: PathBuf,
    pub stem: String,
    pub extension: String,
    pub media_type: MediaType,
    pub duration: Option<Duration>,
    pub embedded_subs: Vec<EmbeddedSubStream>,
}

// Video extensions (matches CMD lines 131-138 classification)
const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "wmv", "flv", "webm", "m4v", "ts", "m2ts", "vob", "ogv", "3gp",
    "mxf", "mpg", "mpeg",
];

// Subtitle extensions
const SUB_EXTS: &[&str] = &["srt", "vtt", "ass", "ssa", "sub", "sup"];

impl MediaFile {
    /// Create a MediaFile from a path. Does not probe -- call `probe()` separately.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let extension = path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        let media_type = classify_extension(&extension);

        Ok(Self {
            path,
            dir,
            stem,
            extension,
            media_type,
            duration: None,
            embedded_subs: Vec::new(),
        })
    }

    pub fn is_video(&self) -> bool {
        self.media_type == MediaType::Video
    }

    pub fn is_subtitle(&self) -> bool {
        self.media_type == MediaType::Subtitle
    }
}

fn classify_extension(ext: &str) -> MediaType {
    if VIDEO_EXTS.contains(&ext) {
        MediaType::Video
    } else if SUB_EXTS.contains(&ext) {
        MediaType::Subtitle
    } else {
        MediaType::Audio
    }
}

// Common audio extensions recognized for batch-folder enumeration. We accept
// anything whisper/ffmpeg can plausibly decode; the pipeline will surface
// errors for anything it can't handle.
const AUDIO_EXTS: &[&str] = &[
    "mp3", "wav", "flac", "m4a", "aac", "ogg", "opus", "wma", "aiff", "aif", "ac3", "amr",
];

/// Iterate media files in a directory (non-recursive). Includes both video and
/// audio extensions -- the user's batch/paste flows are typically audio.
pub fn iter_media_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| {
                        let ext = ext.to_lowercase();
                        VIDEO_EXTS.contains(&ext.as_str())
                            || AUDIO_EXTS.contains(&ext.as_str())
                    })
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}
