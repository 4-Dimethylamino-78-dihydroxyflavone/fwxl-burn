use crate::config::AppConfig;
use crate::error::ToolError;
use crate::media::EmbeddedSubStream;
use crate::tools::ffprobe;
use std::path::{Path, PathBuf};

/// Extract the best-ranked subtitle stream to an SRT file.
pub async fn extract_best_subtitle(
    config: &AppConfig,
    media_path: &Path,
    streams: &[EmbeddedSubStream],
    output_dir: &Path,
    stem: &str,
) -> Result<Option<PathBuf>, ToolError> {
    // Find best text subtitle by score
    let best = streams.iter().max_by_key(|s| s.score);

    let sub = match best {
        Some(s) => s,
        None => return Ok(None),
    };

    let output_path = output_dir.join(format!("{stem}.embedded.srt"));

    ffprobe::extract_subtitle(
        &config.tools.ffmpeg,
        media_path,
        sub.index,
        &output_path,
    )
    .await?;

    Ok(Some(output_path))
}
