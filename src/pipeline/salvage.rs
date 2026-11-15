//! Recovery for transcription runs that finish work but fail to hand back a
//! usable SRT.
//!
//! Whisper is invoked with `--output_format all`, so a run writes `.srt`,
//! `.vtt`, `.tsv`, `.txt` and `.json` side by side. Previously, if the `.srt`
//! was missing or 0 bytes, the pipeline returned an error and the run's
//! `TempDir` was dropped -- which **deletes the directory and every sibling
//! file with it**. An hour of transcription could disappear because one of
//! five output writers produced an empty file.
//!
//! Two guarantees are provided here:
//!
//! 1. **Reconstruct.** If any sibling format carries timed text, rebuild the
//!    SRT from it. `.json` and `.vtt` and `.tsv` all carry per-segment
//!    timings, so the result is a genuine subtitle file, not a plain dump.
//! 2. **Never destroy.** If nothing can be reconstructed, the temp directory
//!    is moved somewhere permanent instead of being swept, so the raw output
//!    (and whatever partial text it holds) survives for inspection.

use crate::srt::SrtEntry;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Which sibling format an SRT was rebuilt from, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SalvageSource {
    Json,
    Vtt,
    Tsv,
}

impl SalvageSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Json => "JSON segments",
            Self::Vtt => "WebVTT",
            Self::Tsv => "TSV",
        }
    }
}

/// Try to rebuild subtitle entries from any non-SRT output file.
///
/// Ordered by fidelity: JSON keeps exact float timings and is what whisper
/// writes first; VTT and TSV are derived views but still fully timed.
pub fn salvage_entries(outputs: &[PathBuf]) -> Option<(Vec<SrtEntry>, SalvageSource)> {
    let find = |ext: &str| -> Option<&PathBuf> {
        outputs.iter().find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
        })
    };

    if let Some(p) = find("json") {
        if let Some(entries) = std::fs::read_to_string(p).ok().and_then(|s| from_json(&s)) {
            if !entries.is_empty() {
                return Some((entries, SalvageSource::Json));
            }
        }
    }
    if let Some(p) = find("vtt") {
        if let Some(entries) = std::fs::read_to_string(p).ok().map(|s| from_vtt(&s)) {
            if !entries.is_empty() {
                return Some((entries, SalvageSource::Vtt));
            }
        }
    }
    if let Some(p) = find("tsv") {
        if let Some(entries) = std::fs::read_to_string(p).ok().map(|s| from_tsv(&s)) {
            if !entries.is_empty() {
                return Some((entries, SalvageSource::Tsv));
            }
        }
    }
    None
}

/// Parse whisper's JSON sidecar: `{"segments":[{"start":f,"end":f,"text":s}]}`.
fn from_json(content: &str) -> Option<Vec<SrtEntry>> {
    let root: serde_json::Value = serde_json::from_str(content).ok()?;
    let segments = root.get("segments")?.as_array()?;

    let mut entries = Vec::with_capacity(segments.len());
    for seg in segments {
        let text = seg.get("text").and_then(|t| t.as_str()).unwrap_or("").trim();
        if text.is_empty() {
            continue;
        }
        let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(start);
        entries.push(SrtEntry {
            index: 0,
            start: secs(start),
            end: secs(end.max(start)),
            text: text.to_string(),
        });
    }
    reindex(&mut entries);
    Some(entries)
}

/// Parse WebVTT cues. Format matches SRT closely; the separator is `.` for
/// fractional seconds rather than `,`, and cues may carry an identifier line.
fn from_vtt(content: &str) -> Vec<SrtEntry> {
    let mut entries = Vec::new();
    let mut lines = content.lines().peekable();

    while let Some(line) = lines.next() {
        let Some((from, to)) = parse_timing_line(line) else {
            continue;
        };
        let mut text_lines: Vec<String> = Vec::new();
        while let Some(next) = lines.peek() {
            if next.trim().is_empty() {
                break;
            }
            text_lines.push(lines.next().unwrap_or("").trim().to_string());
        }
        let text = text_lines.join("\n").trim().to_string();
        if text.is_empty() {
            continue;
        }
        entries.push(SrtEntry {
            index: 0,
            start: from,
            end: to.max(from),
            text,
        });
    }
    reindex(&mut entries);
    entries
}

/// Parse whisper's TSV: a `start<TAB>end<TAB>text` table in milliseconds.
///
/// Whisper wraps long cue text across following lines that contain **no tabs
/// at all**. Those are continuations of the preceding row, not new rows, and
/// treating them as malformed would silently truncate most of the transcript.
fn from_tsv(content: &str) -> Vec<SrtEntry> {
    let mut entries: Vec<SrtEntry> = Vec::new();

    for line in content.lines() {
        let cols: Vec<&str> = line.splitn(3, '\t').collect();

        if cols.len() == 3 {
            // A real row, or the `start end text` header (non-numeric -> skip).
            if let (Ok(start_ms), Ok(end_ms)) =
                (cols[0].trim().parse::<u64>(), cols[1].trim().parse::<u64>())
            {
                entries.push(SrtEntry {
                    index: 0,
                    start: Duration::from_millis(start_ms),
                    end: Duration::from_millis(end_ms.max(start_ms)),
                    text: cols[2].trim().to_string(),
                });
            }
            continue;
        }

        let cont = line.trim();
        if cont.is_empty() {
            continue;
        }
        if let Some(last) = entries.last_mut() {
            if !last.text.is_empty() {
                last.text.push(' ');
            }
            last.text.push_str(cont);
        }
    }

    // Rows whose text was blank and never continued carry no information.
    entries.retain(|e| !e.text.is_empty());
    reindex(&mut entries);
    entries
}

/// Parse an `HH:MM:SS[.,]mmm --> HH:MM:SS[.,]mmm` line, tolerating both the
/// VTT (`.`) and SRT (`,`) fractional separators and an optional cue suffix.
fn parse_timing_line(line: &str) -> Option<(Duration, Duration)> {
    let (lhs, rhs) = line.split_once("-->")?;
    let start = parse_timestamp(lhs.trim())?;
    // VTT cue settings (e.g. "align:start position:0%") follow the end stamp.
    let end_token = rhs.trim().split_whitespace().next()?;
    let end = parse_timestamp(end_token)?;
    Some((start, end))
}

fn parse_timestamp(s: &str) -> Option<Duration> {
    let s = s.replace(',', ".");
    let mut parts = s.split(':').collect::<Vec<_>>();
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    // VTT permits MM:SS.mmm with the hour omitted.
    while parts.len() < 3 {
        parts.insert(0, "0");
    }
    let hours: u64 = parts[0].trim().parse().ok()?;
    let minutes: u64 = parts[1].trim().parse().ok()?;
    let seconds: f64 = parts[2].trim().parse().ok()?;
    Some(secs(hours as f64 * 3600.0 + minutes as f64 * 60.0 + seconds))
}

fn secs(v: f64) -> Duration {
    Duration::from_secs_f64(v.max(0.0))
}

fn reindex(entries: &mut [SrtEntry]) {
    for (i, e) in entries.iter_mut().enumerate() {
        e.index = (i + 1) as u32;
    }
}

/// Move a run's temp directory somewhere permanent so its contents survive.
///
/// Returns the destination. The caller must have taken ownership of the path
/// away from `TempDir` (via `into_path`) first, otherwise `TempDir::drop` will
/// still try to sweep the original location.
pub fn quarantine(tmp_path: &Path, srt_dir: &Path, stem: &str) -> std::io::Result<PathBuf> {
    let mut dest = srt_dir.join(format!("{stem}.fwxl-recovery"));
    // Never clobber a previous recovery from an earlier attempt.
    let mut n = 1;
    while dest.exists() {
        dest = srt_dir.join(format!("{stem}.fwxl-recovery-{n}"));
        n += 1;
        if n > 100 {
            break;
        }
    }

    // Same-volume rename is atomic; fall back to a copy for cross-volume temp
    // dirs (which happens when srt_dir is read-only and the tempdir landed in
    // %TEMP% instead).
    if std::fs::rename(tmp_path, &dest).is_ok() {
        return Ok(dest);
    }

    std::fs::create_dir_all(&dest)?;
    for entry in std::fs::read_dir(tmp_path)? {
        let entry = entry?;
        if !entry.path().is_file() {
            continue;
        }
        let target = dest.join(entry.file_name());
        std::fs::copy(entry.path(), &target)?;
    }
    let _ = std::fs::remove_dir_all(tmp_path);
    Ok(dest)
}

/// A quarantined run that can be turned back into a subtitle file.
pub struct QuarantineRecovery {
    pub entries: Vec<SrtEntry>,
    pub source: SalvageSource,
    /// The recovery directory the entries came from (for logging).
    pub dir: PathBuf,
    /// Language extracted from the quarantined JSON sidecar, if present.
    pub lang: Option<String>,
}

/// Look for `{stem}.fwxl-recovery*` directories in `srt_dir` (as written by
/// [`quarantine`]) and try to rebuild subtitle entries from the newest one
/// that salvages. The recovery directory itself is never modified or removed
/// -- once the caller writes a canonical SRT, the skip-existing check keeps
/// this from re-firing on later runs.
pub fn recover_from_quarantine(srt_dir: &Path, stem: &str) -> Option<QuarantineRecovery> {
    let prefix = format!("{stem}.fwxl-recovery");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(srt_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix))
        })
        .collect();
    // `-1`, `-2` suffixes sort after the bare name; try the latest attempt first.
    candidates.sort();
    candidates.reverse();

    for dir in candidates {
        let files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        if let Some((entries, source)) = salvage_entries(&files) {
            let lang = files
                .iter()
                .find(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                })
                .and_then(|p| crate::srt::extract_language_from_json(p));
            return Some(QuarantineRecovery {
                entries,
                source,
                dir,
                lang,
            });
        }
    }
    None
}

/// Heuristic: did this run die from memory exhaustion rather than a logic bug?
///
/// Matched against whisper/CUDA/PyTorch phrasing. Used to decide whether an
/// automatic retry with memory-heavy stages disabled is worth attempting --
/// there is no point re-running a run that failed for an unrelated reason.
pub fn looks_like_oom(log: &str) -> bool {
    const NEEDLES: [&str; 16] = [
        "out of memory",
        "outofmemory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
        "failed to allocate",
        "memoryerror",
        "bad_alloc",
        "not enough memory",
        "insufficient memory",
        // numpy raises `_ArrayMemoryError: Unable to allocate 12.3 GiB ...`,
        // which is how diarization/clustering on a multi-hour file dies. The
        // wording shares no substring with "failed to allocate".
        "unable to allocate",
        "arraymemoryerror",
        "cannot allocate memory",
        // PyTorch's CPU allocator.
        "can't allocate memory",
        "defaultcpuallocator",
        // Windows-specific: the pagefile, not the GPU, is what ran out.
        "paging file is too small",
        "insufficient system resources",
    ];
    let lower = log.to_lowercase();
    NEEDLES.iter().any(|n| lower.contains(n))
}

/// Strip the memory-hungry stages from a whisper preset for a retry.
///
/// Vocal extraction (mb-roformer), diarization and embedding return are the
/// large VRAM consumers; transcription itself is comparatively cheap. Dropping
/// them costs some output quality but converts a total loss into a usable
/// transcript. Returns `None` when the preset has nothing left to strip, so
/// callers can skip a retry that would run identically.
pub fn degrade_preset(preset: &str) -> Option<String> {
    // Flags that take a value; both the flag and its argument must go.
    const WITH_VALUE: [&str; 4] = [
        "--ff_vocal_extract",
        "--diarize",
        "--roformer_vram",
        "--diarize_ff",
    ];
    // Standalone switches.
    const STANDALONE: [&str; 1] = ["--return_embeddings"];

    let tokens: Vec<&str> = preset.split_whitespace().collect();
    let mut out: Vec<&str> = Vec::with_capacity(tokens.len());
    let mut changed = false;
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        if WITH_VALUE.contains(&tok) {
            // Skip the flag plus its value, if a value is actually present.
            i += if tokens.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
                2
            } else {
                1
            };
            changed = true;
            continue;
        }
        if STANDALONE.contains(&tok) {
            i += 1;
            changed = true;
            continue;
        }
        out.push(tok);
        i += 1;
    }

    if changed { Some(out.join(" ")) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_segments_become_entries() {
        let json = r#"{"segments":[{"start":0.0,"end":1.5,"text":" Hello"},
                                   {"start":1.5,"end":3.0,"text":" world"}]}"#;
        let entries = from_json(json).expect("parses");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[0].index, 1);
        assert_eq!(entries[1].start, Duration::from_millis(1500));
    }

    #[test]
    fn json_skips_empty_text_segments() {
        let json = r#"{"segments":[{"start":0.0,"end":1.0,"text":"   "},
                                   {"start":1.0,"end":2.0,"text":"kept"}]}"#;
        let entries = from_json(json).expect("parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "kept");
        assert_eq!(entries[0].index, 1, "indices renumber after skips");
    }

    #[test]
    fn vtt_cues_become_entries() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nfirst line\n\n\
                   00:01:05.500 --> 00:01:07.000 align:start\nsecond line\n";
        let entries = from_vtt(vtt);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "first line");
        assert_eq!(entries[1].start, Duration::from_millis(65_500));
    }

    #[test]
    fn vtt_accepts_short_timestamps_and_multiline_text() {
        let vtt = "WEBVTT\n\n01:05.000 --> 01:07.000\nline one\nline two\n";
        let entries = from_vtt(vtt);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start, Duration::from_secs(65));
        assert_eq!(entries[0].text, "line one\nline two");
    }

    #[test]
    fn tsv_rows_become_entries_and_header_is_skipped() {
        let tsv = "start\tend\ttext\n0\t1500\tHello\n1500\t3000\tworld\n";
        let entries = from_tsv(tsv);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[1].end, Duration::from_millis(3000));
    }

    /// Verbatim shape of real faster-whisper-xxl TSV output: long cue text is
    /// wrapped onto a following line carrying no tabs. Dropping those would
    /// silently truncate most of the transcript.
    #[test]
    fn tsv_continuation_lines_join_the_previous_row() {
        let tsv = "start\tend\ttext\n\
                   11580\t21580\tHi, I'm Jordan Example,\n\
                   calling about that referral and\n\
                   21580\t26040\tappointment for Saturday 10, 15,\n";
        let entries = from_tsv(tsv);
        assert_eq!(entries.len(), 2, "continuations must not become rows");
        assert_eq!(
            entries[0].text,
            "Hi, I'm Jordan Example, calling about that referral and"
        );
        assert_eq!(entries[0].start, Duration::from_millis(11_580));
        assert_eq!(entries[1].text, "appointment for Saturday 10, 15,");
    }

    /// Verbatim shape of real faster-whisper-xxl VTT output: `MM:SS.mmm`
    /// timestamps with the hour field omitted, and multi-line cue text.
    #[test]
    fn vtt_matches_real_whisper_output_shape() {
        let vtt = "WEBVTT\n\n\
                   00:11.580 --> 00:21.580\n\
                   Hi, I'm Jordan Example,\n\
                   calling about that referral and\n\n\
                   00:21.580 --> 00:26.040\n\
                   appointment for Saturday 10, 15,\n";
        let entries = from_vtt(vtt);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start, Duration::from_millis(11_580));
        assert_eq!(entries[0].end, Duration::from_millis(21_580));
        assert_eq!(
            entries[0].text,
            "Hi, I'm Jordan Example,\ncalling about that referral and"
        );
    }

    /// Verbatim shape of real faster-whisper-xxl JSON: segments carry extra
    /// keys (id/seek/tokens) and text has a leading space.
    #[test]
    fn json_matches_real_whisper_output_shape() {
        let json = r#"{"segments":[
            {"id":1,"seek":0,"start":11.58,"end":23.22,
             "text":" Hi, I'm Jordan Example, calling about that referral",
             "tokens":[50365,2421,11]}
        ]}"#;
        let entries = from_json(json).expect("parses");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start, Duration::from_millis(11_580));
        assert_eq!(entries[0].end, Duration::from_millis(23_220));
        assert_eq!(
            entries[0].text,
            "Hi, I'm Jordan Example, calling about that referral",
            "leading space is trimmed"
        );
    }

    #[test]
    fn degrade_removes_memory_heavy_flags_with_their_values() {
        let preset = "--vad_method ten --voc_device cuda --diarize reverb_v2 \
                      --ff_vocal_extract mb-roformer --roformer_vram 8 --beep_off --standard";
        let degraded = degrade_preset(preset).expect("something was stripped");
        assert!(!degraded.contains("--diarize"));
        assert!(!degraded.contains("reverb_v2"), "value must go with its flag");
        assert!(!degraded.contains("mb-roformer"));
        assert!(!degraded.contains("--roformer_vram"));
        assert!(!degraded.contains(" 8"));
        // Everything unrelated survives.
        assert!(degraded.contains("--vad_method ten"));
        assert!(degraded.contains("--beep_off"));
        assert!(degraded.contains("--standard"));
    }

    #[test]
    fn degrade_returns_none_when_nothing_to_strip() {
        assert!(degrade_preset("--vad_method ten --beep_off --standard").is_none());
    }

    #[test]
    fn oom_detection_matches_common_phrasings() {
        assert!(looks_like_oom("RuntimeError: CUDA out of memory. Tried to allocate"));
        assert!(looks_like_oom("CUBLAS_STATUS_ALLOC_FAILED"));
        assert!(looks_like_oom("std::bad_alloc"));
        assert!(!looks_like_oom("model loaded; transcription complete"));
    }

    /// Phrasings that reach us from the non-CUDA side of a long run. Each of
    /// these used to fall through to "not memory", which skipped the degraded
    /// retry and sent an otherwise recoverable run straight to quarantine.
    #[test]
    fn oom_detection_covers_host_side_allocators() {
        assert!(looks_like_oom(
            "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 12.3 GiB \
             for an array with shape (40000, 40000)"
        ));
        assert!(looks_like_oom(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8589934592 bytes"
        ));
        assert!(looks_like_oom(
            "OSError: [WinError 1455] The paging file is too small for this operation to complete"
        ));
        assert!(looks_like_oom("Cannot allocate memory"));
        // Still must not fire on ordinary progress chatter.
        assert!(!looks_like_oom(
            "Allocating output buffer; VAD complete; 1200 segments"
        ));
    }
}
