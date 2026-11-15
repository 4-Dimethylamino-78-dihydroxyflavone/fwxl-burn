use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A single SRT subtitle entry.
#[derive(Debug, Clone)]
pub struct SrtEntry {
    pub index: u32,
    pub start: Duration,
    pub end: Duration,
    pub text: String,
}

/// Cue-timing adjustments applied to the temporary SRT copy a hardburn
/// reads. Whisper ends every cue the instant speech stops, which reads as
/// abrupt on screen; these rules hold cues longer and chain neighbours so
/// the text flows. All values are milliseconds. See [`extend_cue_timing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueTiming {
    /// Gaps up to this long are closed so consecutive cues chain.
    #[serde(default = "default_bridge_gap_ms")]
    pub bridge_gap_ms: u64,
    /// Across bigger gaps (real pauses) each cue is held this much longer.
    #[serde(default = "default_extend_ms")]
    pub extend_ms: u64,
    /// Cues shorter than this are lengthened to it when there is room.
    #[serde(default = "default_min_duration_ms")]
    pub min_duration_ms: u64,
    /// Never end a cue closer than this to the next cue's start. Roughly
    /// two frames: enough of a blink that the eye registers the change.
    #[serde(default = "default_min_gap_ms")]
    pub min_gap_ms: u64,
}

fn default_bridge_gap_ms() -> u64 {
    1500
}
fn default_extend_ms() -> u64 {
    800
}
fn default_min_duration_ms() -> u64 {
    1200
}
fn default_min_gap_ms() -> u64 {
    80
}

impl Default for CueTiming {
    fn default() -> Self {
        Self {
            bridge_gap_ms: default_bridge_gap_ms(),
            extend_ms: default_extend_ms(),
            min_duration_ms: default_min_duration_ms(),
            min_gap_ms: default_min_gap_ms(),
        }
    }
}

/// What [`extend_cue_timing`] changed, for the progress line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetimeStats {
    pub total: usize,
    pub extended: usize,
    pub added_ms: u64,
}

/// Lengthen cue end times per `timing`.
///
/// Cues are first sorted by start so "next cue" is meaningful. Ends are only
/// ever moved later, never earlier, so overlapping or already-tight cues are
/// left exactly as they were. For each cue the new end is the largest of:
/// the original end, end + `extend_ms`, start + `min_duration_ms`, and (when
/// the gap to the next cue is at most `bridge_gap_ms`) the next start minus
/// `min_gap_ms`; the result is then capped at the next start minus
/// `min_gap_ms`. The last cue has no cap: ffmpeg simply stops drawing at the
/// end of the video.
pub fn extend_cue_timing(entries: &mut [SrtEntry], timing: &CueTiming) -> RetimeStats {
    entries.sort_by_key(|e| e.start);
    let mut stats = RetimeStats {
        total: entries.len(),
        ..Default::default()
    };
    for i in 0..entries.len() {
        let start = entries[i].start.as_millis() as u64;
        let end = entries[i].end.as_millis() as u64;
        let next_start = entries.get(i + 1).map(|e| e.start.as_millis() as u64);

        let mut target = (end + timing.extend_ms).max(start + timing.min_duration_ms);
        if let Some(ns) = next_start {
            let cap = ns.saturating_sub(timing.min_gap_ms);
            if ns.saturating_sub(end) <= timing.bridge_gap_ms {
                target = target.max(cap);
            }
            target = target.min(cap);
        }
        let new_end = target.max(end);
        if new_end != end {
            entries[i].end = Duration::from_millis(new_end);
            stats.extended += 1;
            stats.added_ms += new_end - end;
        }
    }
    stats
}

/// Verification summary for an SRT picked for hardburn. The TUI displays
/// `message` directly; `valid` gates whether the hardburn flow may proceed.
#[derive(Debug, Clone)]
pub struct SrtVerification {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub entry_count: usize,
    pub valid: bool,
    pub message: String,
}

/// Check that an SRT exists, has bytes, and parses to at least one timed entry.
/// Cheap (full file read + parse) but only called on user action, never per
/// frame -- the picker calls it for the highlighted row, hardburn calls it
/// once before launching ffmpeg.
pub fn verify_srt(path: &Path) -> SrtVerification {
    let path_buf = path.to_path_buf();
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            return SrtVerification {
                path: path_buf,
                size_bytes: 0,
                entry_count: 0,
                valid: false,
                message: format!("File not accessible: {e}"),
            };
        }
    };
    let size_bytes = metadata.len();
    if size_bytes == 0 {
        return SrtVerification {
            path: path_buf,
            size_bytes: 0,
            entry_count: 0,
            valid: false,
            message: "SRT is empty (0 bytes)".to_string(),
        };
    }
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            return SrtVerification {
                path: path_buf,
                size_bytes,
                entry_count: 0,
                valid: false,
                message: format!("Could not read SRT: {e}"),
            };
        }
    };
    let entries = parse_srt(&content);
    let entry_count = entries.len();
    if entry_count == 0 {
        return SrtVerification {
            path: path_buf,
            size_bytes,
            entry_count: 0,
            valid: false,
            message: format!("SRT has {size_bytes} bytes but no parseable timing entries"),
        };
    }
    SrtVerification {
        path: path_buf,
        size_bytes,
        entry_count,
        valid: true,
        message: format!("{entry_count} entries, {size_bytes} bytes"),
    }
}

/// Parse an SRT file into entries.
pub fn parse_srt(content: &str) -> Vec<SrtEntry> {
    let mut entries = Vec::new();
    let mut lines = content.lines().peekable();

    while lines.peek().is_some() {
        // Skip blank lines
        while lines.peek().is_some_and(|l| l.trim().is_empty()) {
            lines.next();
        }

        // Index line
        let index_line = match lines.next() {
            Some(l) => l.trim(),
            None => break,
        };
        let index: u32 = match index_line.parse() {
            Ok(i) => i,
            Err(_) => continue,
        };

        // Timestamp line: "00:00:01,000 --> 00:00:04,500"
        let ts_line = match lines.next() {
            Some(l) => l.trim().to_string(),
            None => break,
        };

        let (start, end) = match parse_timestamp_line(&ts_line) {
            Some(pair) => pair,
            None => continue,
        };

        // Text lines until blank line or EOF
        let mut text_parts = Vec::new();
        while lines.peek().is_some_and(|l| !l.trim().is_empty()) {
            text_parts.push(lines.next().unwrap().to_string());
        }
        let text = text_parts.join("\n");

        entries.push(SrtEntry {
            index,
            start,
            end,
            text,
        });
    }

    entries
}

/// Write SRT entries to a string.
pub fn write_srt(entries: &[SrtEntry]) -> String {
    let mut output = String::new();
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(&format!("{}\n", i + 1));
        output.push_str(&format!(
            "{} --> {}\n",
            format_timestamp(entry.start),
            format_timestamp(entry.end)
        ));
        output.push_str(&entry.text);
        output.push('\n');
    }
    output
}

/// Write SRT entries to a file.
pub fn write_srt_file(entries: &[SrtEntry], path: &Path) -> std::io::Result<()> {
    let content = write_srt(entries);
    std::fs::write(path, content)
}

/// Merge multiple SRT files from chunks, adjusting timestamps by chunk offsets.
/// `chunks` is a list of (start_offset_secs, srt_content).
pub fn merge_chunk_srts(chunks: &[(f64, String)]) -> Vec<SrtEntry> {
    let mut merged = Vec::new();

    for (offset, content) in chunks {
        let entries = parse_srt(content);
        let offset_dur = Duration::from_secs_f64(*offset);

        for entry in entries {
            merged.push(SrtEntry {
                index: 0, // will be re-indexed
                start: entry.start + offset_dur,
                end: entry.end + offset_dur,
                text: entry.text,
            });
        }
    }

    // Re-index
    for (i, entry) in merged.iter_mut().enumerate() {
        entry.index = (i + 1) as u32;
    }

    merged
}

/// Extract the detected language from a Whisper JSON output file.
pub fn extract_language_from_json(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("language")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

// -- Internal helpers --

fn parse_timestamp_line(line: &str) -> Option<(Duration, Duration)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return None;
    }
    let start = parse_timestamp(parts[0].trim())?;
    let end = parse_timestamp(parts[1].trim())?;
    Some((start, end))
}

fn parse_timestamp(ts: &str) -> Option<Duration> {
    // Format: "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    let ts = ts.replace(',', ".");
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(Duration::from_secs_f64(h * 3600.0 + m * 60.0 + s))
}

fn format_timestamp(d: Duration) -> String {
    let total_ms = d.as_millis() as u64;
    let ms = (total_ms % 1000) as u32;
    let total_secs = total_ms / 1000;
    let s = (total_secs % 60) as u32;
    let total_mins = total_secs / 60;
    let m = (total_mins % 60) as u32;
    let h = (total_mins / 60) as u32;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_write_roundtrip() {
        let input = "1\n00:00:01,000 --> 00:00:04,500\nHello world\n\n2\n00:01:00,000 --> 00:01:05,250\nSecond line\n";
        let entries = parse_srt(input);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello world");
        assert_eq!(entries[1].text, "Second line");

        let output = write_srt(&entries);
        let re_parsed = parse_srt(&output);
        assert_eq!(re_parsed.len(), 2);
        assert_eq!(re_parsed[0].text, "Hello world");
    }

    #[test]
    fn test_timestamp_parsing() {
        let ts = parse_timestamp("01:30:45,123").unwrap();
        let total_ms = ts.as_millis();
        assert_eq!(total_ms, 5445123);
    }

    #[test]
    fn test_format_timestamp() {
        let d = Duration::from_millis(5445123);
        assert_eq!(format_timestamp(d), "01:30:45,123");
    }

    #[test]
    fn test_merge_chunks() {
        let chunk0 = "1\n00:00:01,000 --> 00:00:04,000\nFirst\n";
        let chunk1 = "1\n00:00:00,500 --> 00:00:03,000\nSecond\n";

        let merged = merge_chunk_srts(&[(0.0, chunk0.to_string()), (10.0, chunk1.to_string())]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start, Duration::from_secs(1));
        assert_eq!(merged[1].start, Duration::from_millis(10500));
    }

    fn cue(start_ms: u64, end_ms: u64) -> SrtEntry {
        SrtEntry {
            index: 0,
            start: Duration::from_millis(start_ms),
            end: Duration::from_millis(end_ms),
            text: String::new(),
        }
    }

    fn ends(entries: &[SrtEntry]) -> Vec<u64> {
        entries.iter().map(|e| e.end.as_millis() as u64).collect()
    }

    #[test]
    fn retime_bridges_small_gaps_to_next_cue() {
        // defaults: bridge 1500, extend 800, min_duration 1200, min_gap 80
        let t = CueTiming::default();
        let mut e = vec![cue(1000, 2000), cue(2800, 4000), cue(4500, 5000)];
        let stats = extend_cue_timing(&mut e, &t);
        // gap 800 -> chained to 2800-80; gap 500 -> chained to 4500-80;
        // last cue: max(5000+800, 4500+1200) = 5800
        assert_eq!(ends(&e), vec![2720, 4420, 5800]);
        assert_eq!(
            stats,
            RetimeStats {
                total: 3,
                extended: 3,
                added_ms: 720 + 420 + 800
            }
        );
    }

    #[test]
    fn retime_holds_across_real_pauses_but_does_not_bridge_them() {
        let t = CueTiming::default();
        let mut e = vec![cue(1000, 2000), cue(6000, 7000)];
        extend_cue_timing(&mut e, &t);
        assert_eq!(ends(&e), vec![2800, 7800]);
    }

    #[test]
    fn retime_lifts_short_cues_to_min_duration() {
        let t = CueTiming::default();
        let mut e = vec![cue(1000, 1300), cue(5000, 6000)];
        extend_cue_timing(&mut e, &t);
        // 1300+800 = 2100 < 1000+1200 = 2200; gap 3700 > bridge; cap 4920
        assert_eq!(ends(&e)[0], 2200);
    }

    #[test]
    fn retime_never_shortens_overlapping_or_tight_cues() {
        let t = CueTiming::default();
        let mut e = vec![cue(1000, 3000), cue(2500, 4000), cue(4050, 5000)];
        let stats = extend_cue_timing(&mut e, &t);
        // first overlaps second: untouched; second ends 50 ms before the
        // third (< min_gap): untouched; only the last cue moves.
        assert_eq!(ends(&e)[0], 3000);
        assert_eq!(ends(&e)[1], 4000);
        assert_eq!(stats.extended, 1);
    }

    #[test]
    fn retime_sorts_by_start_before_chaining() {
        let t = CueTiming::default();
        let mut e = vec![cue(2800, 4000), cue(1000, 2000)];
        extend_cue_timing(&mut e, &t);
        assert_eq!(e[0].start, Duration::from_millis(1000));
        assert_eq!(ends(&e), vec![2720, 4800]);
    }

    #[test]
    fn retime_seamless_when_min_gap_is_zero() {
        let t = CueTiming {
            min_gap_ms: 0,
            ..CueTiming::default()
        };
        let mut e = vec![cue(1000, 2000), cue(2800, 4000)];
        extend_cue_timing(&mut e, &t);
        assert_eq!(ends(&e)[0], 2800);
    }

    #[test]
    fn cue_timing_partial_toml_fills_defaults() {
        let t: CueTiming = toml::from_str("bridge_gap_ms = 900").unwrap();
        assert_eq!(
            t,
            CueTiming {
                bridge_gap_ms: 900,
                ..CueTiming::default()
            }
        );
    }
}
