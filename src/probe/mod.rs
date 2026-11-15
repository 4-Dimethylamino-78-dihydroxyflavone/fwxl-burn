pub mod embedded;
pub mod scan;

use crate::config::AppConfig;
use crate::error::ToolError;
use crate::media::{EmbeddedSubStream, MediaFile, SubStreamType};
use crate::tools::ffprobe;
use std::path::Path;

pub use crate::tools::ffprobe::ProbeStatus;

/// Probe a media file for duration and embedded subtitles.
///
/// Returns whether the container was actually readable. `Err` means ffprobe could
/// not be run at all (spawn failure / timeout); a file ffmpeg rejects comes back
/// as `Ok(ProbeStatus::Unreadable { .. })` so callers can report or repair it
/// instead of aborting.
pub async fn probe_media_file(
    config: &AppConfig,
    media: &mut MediaFile,
) -> Result<ProbeStatus, ToolError> {
    let result = ffprobe::probe_media(&config.tools.ffprobe, &media.path).await?;
    media.duration = result.duration;
    media.embedded_subs = result.subtitle_streams;
    Ok(result.status)
}

/// Get the best-ranked text subtitle stream (for "Native" mode).
pub fn best_text_subtitle(media: &MediaFile) -> Option<&EmbeddedSubStream> {
    media
        .embedded_subs
        .iter()
        .filter(|s| s.stream_type == SubStreamType::Text)
        .max_by_key(|s| s.score)
}

/// Get bitmap subtitle streams (PGS/VOBSUB) for OCR.
pub fn bitmap_subtitle_streams(media: &MediaFile) -> Vec<&EmbeddedSubStream> {
    media
        .embedded_subs
        .iter()
        .filter(|s| matches!(s.stream_type, SubStreamType::Pgs | SubStreamType::Vobsub))
        .collect()
}
