use crate::error::ToolError;
use crate::media::{EmbeddedSubStream, SubStreamType};
use crate::tools::{ToolInvocation, run_tool};
use serde::Deserialize;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Raw FFprobe JSON output structure.
#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    streams: Option<Vec<FfprobeStream>>,
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    index: u32,
    codec_name: Option<String>,
    codec_type: Option<String>,
    tags: Option<HashMap<String, String>>,
    disposition: Option<HashMap<String, i32>>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    filename: Option<String>,
}

/// Whether ffprobe could actually open and parse the container.
///
/// `run_tool` reports a non-zero exit as `Ok(ToolOutput { exit_code, .. })`, and
/// ffprobe still prints a well-formed `{}` on stdout when it gives up. Together
/// those meant a file ffmpeg cannot open at all sailed through `probe_media` as
/// merely "duration unknown", and whisper was launched against it anyway --
/// burning a run that could only ever produce an empty output directory.
///
/// This is deliberately NOT an `Err`: `Err` stays reserved for spawn failure and
/// timeout, because callers treat those as infrastructure faults. An unreadable
/// *file* is a fact about the input, and callers need to branch on it (report it,
/// skip it, or repair it) rather than abort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    /// ffprobe parsed the container.
    Readable,
    /// ffprobe rejected the file. `reason` is its stderr, condensed.
    Unreadable { reason: String },
}

impl ProbeStatus {
    pub fn is_readable(&self) -> bool {
        matches!(self, Self::Readable)
    }

    /// The failure reason, or `None` when the container was readable.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Unreadable { reason } => Some(reason),
            Self::Readable => None,
        }
    }
}

/// Probe result containing duration and subtitle streams.
#[derive(Debug)]
pub struct ProbeResult {
    pub duration: Option<Duration>,
    pub subtitle_streams: Vec<EmbeddedSubStream>,
    pub status: ProbeStatus,
}

/// Condense ffprobe's stderr into one short human-readable reason.
///
/// ffprobe prefixes most diagnostics with a `[demuxer @ 0xaddr]` tag and usually
/// emits both a cause and a generic follow-up, e.g.
///   `[mov,mp4,m4a,3gp,3g2,mj2 @ 0000...] moov atom not found`
///   `D:\path\file.m4a: Invalid data found when processing input`
/// The tag is noise (the address changes every run, which would defeat any
/// downstream dedup), so it is stripped and the first two distinct lines kept.
fn condense_ffprobe_error(stderr: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for line in stderr.lines() {
        let mut line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Strip a leading "[...] " demuxer tag.
        if line.starts_with('[') {
            if let Some(close) = line.find(']') {
                line = line[close + 1..].trim_start();
            }
        }
        if line.is_empty() {
            continue;
        }
        let owned = line.to_string();
        if !parts.contains(&owned) {
            parts.push(owned);
        }
        if parts.len() == 2 {
            break;
        }
    }
    if parts.is_empty() {
        return "ffprobe rejected the file without explanation".to_string();
    }
    let mut joined = parts.join("; ");
    if joined.len() > 300 {
        joined.truncate(300);
        joined.push('…');
    }
    joined
}

/// Run ffprobe and parse the JSON output for stream/format info.
pub async fn probe_media(ffprobe_path: &Path, media_path: &Path) -> Result<ProbeResult, ToolError> {
    let inv = ToolInvocation {
        program: ffprobe_path.to_path_buf(),
        args: vec![
            OsString::from("-v"),
            // NOT `quiet`: the whole point is to keep the reason a file could not
            // be opened ("moov atom not found") instead of discarding it.
            OsString::from("error"),
            OsString::from("-print_format"),
            OsString::from("json"),
            OsString::from("-show_format"),
            OsString::from("-show_streams"),
            OsString::from(media_path),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(30)),
    };

    let output = run_tool(&inv).await?;

    // A container ffprobe cannot walk (interrupted recording with no `moov`,
    // truncated download, wrong extension) exits non-zero AND still prints a
    // parseable `{}`. Read the exit code before the JSON or the file reads as a
    // valid media file that merely has no duration and no streams.
    if output.exit_code != Some(0) {
        return Ok(ProbeResult {
            duration: None,
            subtitle_streams: Vec::new(),
            status: ProbeStatus::Unreadable {
                reason: condense_ffprobe_error(&output.stderr),
            },
        });
    }

    let parsed: FfprobeOutput = match serde_json::from_str(&output.stdout) {
        Ok(p) => p,
        // Exit 0 with unparseable stdout is not an infrastructure fault; it still
        // means we learned nothing about this file, so report it the same way.
        Err(e) => {
            return Ok(ProbeResult {
                duration: None,
                subtitle_streams: Vec::new(),
                status: ProbeStatus::Unreadable {
                    reason: format!("ffprobe emitted unparseable JSON: {e}"),
                },
            });
        }
    };

    // Extract duration
    let duration = parsed
        .format
        .as_ref()
        .and_then(|f| f.duration.as_ref())
        .and_then(|d| d.parse::<f64>().ok())
        .map(Duration::from_secs_f64);

    let streams = parsed.streams.unwrap_or_default();

    // ffprobe occasionally exits 0 on a container it opened but found nothing in.
    // There is no audio to transcribe in that case, so say so here rather than
    // letting whisper discover it an hour later.
    let status = if streams.is_empty() {
        ProbeStatus::Unreadable {
            reason: "ffprobe found no streams in this container".to_string(),
        }
    } else {
        ProbeStatus::Readable
    };

    // Extract subtitle streams
    let subtitle_streams = streams
        .into_iter()
        .filter(|s| s.codec_type.as_deref() == Some("subtitle"))
        .map(parse_subtitle_stream)
        .collect();

    Ok(ProbeResult {
        duration,
        subtitle_streams,
        status,
    })
}

/// Probe only the duration of a media file (lightweight).
pub async fn probe_duration(ffprobe_path: &Path, media_path: &Path) -> Result<Option<f64>, ToolError> {
    let inv = ToolInvocation {
        program: ffprobe_path.to_path_buf(),
        args: vec![
            OsString::from("-v"),
            OsString::from("error"),
            OsString::from("-show_entries"),
            OsString::from("format=duration"),
            OsString::from("-of"),
            OsString::from("default=noprint_wrappers=1:nokey=1"),
            OsString::from(media_path),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(30)),
    };

    let output = run_tool(&inv).await?;
    let duration = output.stdout.trim().parse::<f64>().ok();
    Ok(duration)
}

fn parse_subtitle_stream(stream: FfprobeStream) -> EmbeddedSubStream {
    let codec = stream.codec_name.unwrap_or_default();
    let tags = stream.tags.unwrap_or_default();
    let disposition = stream.disposition.unwrap_or_default();

    let lang = tags
        .get("language")
        .cloned()
        .unwrap_or_else(|| "und".to_string());
    let title = tags.get("title").cloned();

    let is_default = disposition.get("default").copied().unwrap_or(0) != 0;
    let is_forced = disposition.get("forced").copied().unwrap_or(0) != 0;

    let title_lower = title.as_deref().unwrap_or("").to_lowercase();
    let is_sdh = title_lower.contains("sdh")
        || title_lower.contains("hearing impaired")
        || title_lower.contains("hard of hearing");

    let stream_type = classify_codec(&codec);

    // Scoring logic matching CMD script's ranking algorithm:
    // Text codec: 100, default flag: 40, non-commentary: 20, non-SDH: 10, non-forced: 5
    let is_commentary = title_lower.contains("commentary") || title_lower.contains("director");
    let is_english = lang == "eng" || lang == "en" || lang == "und";

    let mut score: i32 = 0;
    if stream_type == SubStreamType::Text {
        score += 100;
    }
    if is_default {
        score += 40;
    }
    if !is_commentary {
        score += 20;
    }
    if !is_sdh {
        score += 10;
    }
    if !is_forced {
        score += 5;
    }
    if is_english {
        score += 50;
    }

    EmbeddedSubStream {
        index: stream.index,
        codec,
        lang,
        stream_type,
        is_default,
        is_forced,
        is_sdh,
        title,
        score,
    }
}

fn classify_codec(codec: &str) -> SubStreamType {
    match codec {
        "subrip" | "srt" | "ass" | "ssa" | "webvtt" | "mov_text" | "text" => SubStreamType::Text,
        "hdmv_pgs_subtitle" | "pgssub" => SubStreamType::Pgs,
        "dvd_subtitle" | "dvdsub" => SubStreamType::Vobsub,
        _ => SubStreamType::Unknown,
    }
}

/// Extract a specific subtitle stream to an SRT file using ffmpeg.
pub async fn extract_subtitle(
    ffmpeg_path: &Path,
    media_path: &Path,
    stream_index: u32,
    output_path: &Path,
) -> Result<PathBuf, ToolError> {
    let inv = ToolInvocation {
        program: ffmpeg_path.to_path_buf(),
        args: vec![
            OsString::from("-y"),
            OsString::from("-i"),
            OsString::from(media_path),
            OsString::from("-map"),
            OsString::from(format!("0:{stream_index}")),
            OsString::from("-c:s"),
            OsString::from("srt"),
            OsString::from(output_path),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(60)),
    };

    run_tool(&inv).await?;
    Ok(output_path.to_path_buf())
}
