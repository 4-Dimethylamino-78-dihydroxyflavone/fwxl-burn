use crate::config::StyleConfig;
use crate::error::{PipelineError, ToolError};
use crate::pipeline::{PipelineContext, PipelineEvent, SubtitleStyle};
use crate::srt::{self, CueTiming, RetimeStats};
use crate::tools::ffmpeg;
use crate::tools::ToolLine;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// Longest sanitised stem used for a staged SRT copy. yt-dlp titles can be
/// long; the temp name must not be what pushes a path over the limit.
const MAX_STAGED_STEM: usize = 80;

/// Run the FFmpeg NVENC hardburn pipeline.
///
/// Two kinds of temporary SRT copy can be involved, and the file on disk is
/// never modified by either:
///
/// * Styles with a timing policy (`[styles.timing.<key>]` in config, or the
///   built-in one for Youtube Flow) burn from a re-timed copy in which cues
///   are held longer and chained. See [`srt::extend_cue_timing`].
/// * The subtitle path is embedded in ffmpeg's filtergraph mini-language,
///   which layers its own quoting rules on top of the OS path rules. The
///   first attempt uses the real path, escaped by
///   [`ffmpeg::subtitles_filter_path`]. If ffmpeg still rejects the *path*
///   (as opposed to failing the encode), the SRT is copied beside the
///   output under a plain-ASCII name and the burn is retried once.
pub async fn run_hardburn(
    ctx: &PipelineContext,
    srt_path: &Path,
    style: SubtitleStyle,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PathBuf, PipelineError> {
    let style_string = resolve_style_preset(&ctx.config.styles, style);

    let output_name = format!("{}.{}.hardcoded.mp4", ctx.media.stem, ctx.timestamp);
    let output_path = ctx.final_out_dir.join(&output_name);

    // Styles with a timing policy burn from a re-timed temporary copy.
    let timing = ctx
        .config
        .styles
        .timing
        .get(style.config_key())
        .cloned()
        .or_else(|| style.builtin_timing());

    let mut burn_src = srt_path.to_path_buf();
    let mut staged_retimed: Option<PathBuf> = None;
    let mut phase_name = "Hardburning subtitles with NVENC".to_string();
    if let Some(policy) = timing {
        let (path, stats) =
            stage_retimed_copy(srt_path, &policy, &ctx.final_out_dir, &ctx.timestamp)
                .map_err(PipelineError::HardburnStaging)?;
        phase_name = format!(
            "Hardburning with NVENC · flow timing: {}/{} cues held longer (+{:.1}s)",
            stats.extended,
            stats.total,
            stats.added_ms as f64 / 1000.0
        );
        burn_src = path.clone();
        staged_retimed = Some(path);
    }

    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: phase_name,
        phase_num: 1,
        total_phases: 1,
    });

    let outcome = match burn_once(ctx, &burn_src, &output_path, &style_string, &event_tx).await {
        Err(first) if ffmpeg::is_subtitle_path_failure(&first) => {
            match stage_sanitized_copy(&burn_src, &ctx.final_out_dir, &ctx.timestamp) {
                Ok(staged) => {
                    let _ = event_tx.send(PipelineEvent::Warning(format!(
                        "ffmpeg rejected the subtitle path; retrying from a sanitized copy: {}",
                        staged.display()
                    )));
                    let second =
                        burn_once(ctx, &staged, &output_path, &style_string, &event_tx).await;
                    let _ = std::fs::remove_file(&staged);
                    second
                }
                Err(io) => {
                    let _ = event_tx.send(PipelineEvent::Warning(format!(
                        "ffmpeg rejected the subtitle path and a sanitized copy could not be staged in {}: {io}",
                        ctx.final_out_dir.display()
                    )));
                    Err(first)
                }
            }
        }
        other => other,
    };

    if let Some(p) = staged_retimed {
        let _ = std::fs::remove_file(&p);
    }

    outcome.map_err(|e| PipelineError::ToolFailed {
        phase: "hardburn".to_string(),
        source: Box::new(e),
    })?;

    let _ = event_tx.send(PipelineEvent::PhaseCompleted {
        name: "Hardburn complete".to_string(),
        duration: std::time::Duration::ZERO,
    });

    Ok(output_path)
}

/// The ASS `force_style` string for a style. Borrows a visual twin's preset
/// when the user's config predates this style, then falls back to
/// `default`, then to an empty string (libass defaults).
pub fn resolve_style_preset(styles: &StyleConfig, style: SubtitleStyle) -> String {
    let presets = &styles.presets;
    presets
        .get(style.config_key())
        .or_else(|| style.visual_fallback_key().and_then(|k| presets.get(k)))
        .or_else(|| presets.get("default"))
        .cloned()
        .unwrap_or_default()
}

/// One ffmpeg run: wires the tool's line stream into progress events and
/// returns ffmpeg's verdict.
async fn burn_once(
    ctx: &PipelineContext,
    srt_path: &Path,
    output_path: &Path,
    style: &str,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<(), ToolError> {
    let (tool_tx, mut tool_rx) = mpsc::unbounded_channel::<ToolLine>();

    // Forward and parse FFmpeg progress
    let event_tx2 = event_tx.clone();
    let total_duration = ctx.media.duration.map(|d| d.as_secs_f64()).unwrap_or(0.0);

    let forward = tokio::spawn(async move {
        while let Some(line) = tool_rx.recv().await {
            match line {
                ToolLine::Stderr(ref l) => {
                    // Parse FFmpeg progress
                    if let Some((current, speed)) = ffmpeg::parse_ffmpeg_progress(l) {
                        if total_duration > 0.0 {
                            let percent = (current / total_duration * 100.0).min(100.0) as f32;
                            let remaining = if speed > 0.0 {
                                (total_duration - current) / speed
                            } else {
                                0.0
                            };
                            let _ = event_tx2.send(PipelineEvent::Progress {
                                percent,
                                message: format!(
                                    "{:.0}% @ {:.1}x speed, ~{:.0}s remaining",
                                    percent, speed, remaining
                                ),
                            });
                        }
                    } else {
                        let _ = event_tx2.send(PipelineEvent::StderrLine(l.clone()));
                    }
                }
                ToolLine::Stdout(l) => {
                    let _ = event_tx2.send(PipelineEvent::StdoutLine(l));
                }
                ToolLine::Done { .. } => break,
            }
        }
    });

    let result = ffmpeg::hardburn(
        &ctx.config,
        &ctx.media.path,
        srt_path,
        output_path,
        style,
        tool_tx,
    )
    .await;

    let _ = forward.await;
    result.map(|_| ())
}

/// Reduce a file stem to plain ASCII `[A-Za-z0-9.-]` words joined by `_`.
///
/// Everything else (apostrophes, brackets, fullwidth punctuation, spaces)
/// becomes a single `_`, runs collapse, leading and trailing `_`/`.` are
/// trimmed, and the result is capped at [`MAX_STAGED_STEM`] characters.
/// This only ever names a temporary copy; the original file is untouched.
pub fn sanitize_stem(stem: &str) -> String {
    let mut out = String::with_capacity(stem.len());
    let mut pending_sep = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '-') {
            if pending_sep && !out.is_empty() {
                out.push('_');
            }
            pending_sep = false;
            out.push(c);
        } else {
            pending_sep = true;
        }
    }
    let trimmed = out.trim_matches(|c| c == '_' || c == '.');
    let capped: String = trimmed.chars().take(MAX_STAGED_STEM).collect();
    if capped.is_empty() {
        "subtitle".to_string()
    } else {
        capped
    }
}

/// Temp-copy name for `srt` inside `dir`: sanitised stem, run tag, and a
/// `kind` marker so leftovers are recognisable.
fn staged_name(srt: &Path, dir: &Path, tag: &str, kind: &str) -> PathBuf {
    let stem = srt
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    dir.join(format!(
        "{}.{}.{kind}.fwxl-burn-tmp.srt",
        sanitize_stem(&stem),
        sanitize_stem(tag)
    ))
}

/// Copy `srt` into `dir` under a sanitised name that survives ffmpeg's
/// filtergraph parser untouched. Placed beside the output (never %TEMP%)
/// so it lives on the same disk the burn already needs to write to.
pub fn stage_sanitized_copy(srt: &Path, dir: &Path, tag: &str) -> std::io::Result<PathBuf> {
    let staged = staged_name(srt, dir, tag, "ascii");
    std::fs::copy(srt, &staged)?;
    Ok(staged)
}

/// Write a copy of `srt` into `dir` with [`srt::extend_cue_timing`]
/// applied. The copy already carries a sanitised ASCII name. Returns the
/// copy's path and what changed.
pub fn stage_retimed_copy(
    srt: &Path,
    timing: &CueTiming,
    dir: &Path,
    tag: &str,
) -> Result<(PathBuf, RetimeStats), String> {
    let content = std::fs::read_to_string(srt)
        .map_err(|e| format!("could not read {}: {e}", srt.display()))?;
    let mut entries = srt::parse_srt(content.trim_start_matches('\u{feff}'));
    if entries.is_empty() {
        return Err(format!("{} has no parseable cues", srt.display()));
    }
    let stats = srt::extend_cue_timing(&mut entries, timing);
    let staged = staged_name(srt, dir, tag, "flow");
    srt::write_srt_file(&entries, &staged)
        .map_err(|e| format!("could not write {}: {e}", staged.display()))?;
    Ok((staged, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn sanitize_stem_flattens_ytdlp_names_to_ascii() {
        let s = sanitize_stem(
            "What's Wrong With Society ｜ Mr. Robot.[20220927].[137+140].[jhxqDkaxoZU].FWXL-Transcribe.U.large-v2_en",
        );
        assert!(s.is_ascii());
        assert!(s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')));
        assert!(!s.contains('\''));
        assert!(s.starts_with("What_s_Wrong_With_Society_Mr._Robot._20220927_._137_140_"));
        assert!(s.len() <= MAX_STAGED_STEM);
    }

    #[test]
    fn sanitize_stem_collapses_and_caps() {
        assert_eq!(sanitize_stem("  ''[[]]  "), "subtitle");
        assert_eq!(sanitize_stem("a'''b   c"), "a_b_c");
        assert_eq!(sanitize_stem(&"x".repeat(200)).len(), MAX_STAGED_STEM);
    }

    #[test]
    fn stage_sanitized_copy_writes_ascii_named_twin() {
        let dir = std::env::temp_dir().join(format!("fwxl-hardburn-stage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("What's [odd] ｜ name.srt");
        std::fs::write(&src, "1\n00:00:00,000 --> 00:00:01,000\nhi\n").unwrap();

        let staged = stage_sanitized_copy(&src, &dir, "2026-09-07-1600").unwrap();
        let name = staged.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(name, "What_s_odd_name.2026-09-07-1600.ascii.fwxl-burn-tmp.srt");
        assert_eq!(std::fs::read(&staged).unwrap(), std::fs::read(&src).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_retimed_copy_extends_cues_and_leaves_original_alone() {
        let dir = std::env::temp_dir().join(format!("fwxl-hardburn-retime-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("What's [odd].srt");
        // CRLF + BOM, as some tools write it.
        let original = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,000\r\nfirst\r\n\r\n2\r\n00:00:02,800 --> 00:00:04,000\r\nsecond\r\n";
        std::fs::write(&src, original).unwrap();

        let (staged, stats) =
            stage_retimed_copy(&src, &CueTiming::default(), &dir, "2026-09-07-1600").unwrap();
        assert_eq!(
            staged.file_name().unwrap().to_string_lossy(),
            "What_s_odd.2026-09-07-1600.flow.fwxl-burn-tmp.srt"
        );
        assert_eq!(stats, RetimeStats { total: 2, extended: 2, added_ms: 720 + 800 });

        let retimed = srt::parse_srt(&std::fs::read_to_string(&staged).unwrap());
        assert_eq!(retimed.len(), 2);
        assert_eq!(retimed[0].end.as_millis(), 2720);
        assert_eq!(retimed[1].end.as_millis(), 4800);
        assert_eq!(retimed[0].text, "first");
        // Source file is byte-for-byte untouched.
        assert_eq!(std::fs::read_to_string(&src).unwrap(), original);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_style_preset_borrows_youtube_when_flow_is_missing() {
        let mut styles = StyleConfig::default();
        let youtube = styles.presets["youtube"].clone();
        // Simulate a user config written before youtube_flow existed.
        styles.presets.remove("youtube_flow");
        styles.timing.clear();

        assert_eq!(resolve_style_preset(&styles, SubtitleStyle::YoutubeFlow), youtube);
        assert_eq!(
            resolve_style_preset(&styles, SubtitleStyle::Netflix),
            styles.presets["netflix"]
        );
        assert!(SubtitleStyle::YoutubeFlow.builtin_timing().is_some());
        assert!(SubtitleStyle::Youtube.builtin_timing().is_none());
    }

    /// Real-file check: re-time the Whisper SRT that motivated the flow
    /// style and report what changed. Ignored: needs this machine's media.
    ///   cargo test --release -- --ignored retime_real --nocapture
    #[test]
    #[ignore]
    fn retime_real_whisper_srt_keeps_text_and_order() {
        let src = Path::new(
            "D:/YTDL.cmd/Mr. Robot/137+140/source/What's Wrong With Society ｜ Mr. Robot.[20220927].[137+140].[jhxqDkaxoZU].FWXL-Transcribe.U.large-v2_en.srt",
        );
        let dir = std::env::temp_dir();
        let (staged, stats) =
            stage_retimed_copy(src, &CueTiming::default(), &dir, "real").unwrap();

        let mut before = srt::parse_srt(&std::fs::read_to_string(src).unwrap());
        before.sort_by_key(|e| e.start);
        let after = srt::parse_srt(&std::fs::read_to_string(&staged).unwrap());
        assert_eq!(before.len(), after.len());
        for (i, (b, a)) in before.iter().zip(&after).enumerate() {
            assert_eq!(a.text, b.text, "cue {i} text changed");
            assert_eq!(a.start, b.start, "cue {i} start changed");
            assert!(a.end >= b.end, "cue {i} was shortened");
            if let Some(next) = before.get(i + 1) {
                let allowed = b.end.max(next.start.saturating_sub(Duration::from_millis(80)));
                assert!(a.end <= allowed, "cue {i} runs into the next cue: {:?} > {:?}", a.end, allowed);
            }
        }
        eprintln!(
            "retimed {}/{} cues, +{:.1}s total",
            stats.extended,
            stats.total,
            stats.added_ms as f64 / 1000.0
        );
        for (b, a) in before.iter().zip(&after).take(6) {
            eprintln!(
                "  {:>8.3}-{:>8.3}  ->  end {:>8.3}  {:?}",
                b.start.as_secs_f64(),
                b.end.as_secs_f64(),
                a.end.as_secs_f64(),
                &b.text[..b.text.len().min(40)]
            );
        }
        // FWXL_KEEP_STAGED=1 leaves the re-timed copy in %TEMP% for a manual burn.
        if std::env::var_os("FWXL_KEEP_STAGED").is_some() {
            eprintln!("kept: {}", staged.display());
        } else {
            let _ = std::fs::remove_file(&staged);
        }
    }
}
