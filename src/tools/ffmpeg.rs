use crate::config::AppConfig;
use crate::error::ToolError;
use crate::tools::{ToolInvocation, ToolLine, run_tool, run_tool_lowprio, run_tool_streaming, validate_output};
use regex::Regex;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc;

/// Hardware H.264 encoder used for the hardburn.
///
/// Every variant is a GPU encoder on purpose -- there is no software (libx264)
/// option, in keeping with the rest of the pipeline: if the GPU path is
/// unavailable the burn fails with ffmpeg's own error rather than quietly
/// taking ten times longer on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoEncoder {
    /// NVIDIA NVENC (`h264_nvenc`), decode via NVDEC. Windows and Linux.
    Nvenc,
    /// Apple VideoToolbox (`h264_videotoolbox`). macOS.
    VideoToolbox,
}

impl Default for VideoEncoder {
    /// VideoToolbox on macOS, where NVENC does not exist; NVENC elsewhere.
    fn default() -> Self {
        if cfg!(target_os = "macos") {
            Self::VideoToolbox
        } else {
            Self::Nvenc
        }
    }
}

impl VideoEncoder {
    /// Value for ffmpeg's `-hwaccel` (hardware *decode*) input option.
    fn hwaccel(self) -> &'static str {
        match self {
            Self::Nvenc => "cuda",
            Self::VideoToolbox => "videotoolbox",
        }
    }

    /// Encoder selection plus quality arguments for a constant-quality `cq`
    /// on the NVENC/x264 scale (0-51, lower = better).
    fn codec_args(self, cq: &str) -> Vec<OsString> {
        let args: Vec<String> = match self {
            Self::Nvenc => vec![
                "-c:v".into(),
                "h264_nvenc".into(),
                "-preset".into(),
                "p5".into(),
                "-cq".into(),
                cq.to_string(),
            ],
            Self::VideoToolbox => {
                // VideoToolbox has no -cq; its constant-quality knob is
                // `-q:v` on a 1-100 scale where HIGHER is better (and it is
                // only honoured on Apple Silicon). Map linearly so the
                // default cq=19 lands at 62 -- an approximation, not a
                // calibrated equivalence.
                let cq: i32 = cq.parse().unwrap_or(19);
                let q = (100 - cq * 2).clamp(1, 100);
                vec![
                    "-c:v".into(),
                    "h264_videotoolbox".into(),
                    "-q:v".into(),
                    q.to_string(),
                ]
            }
        };
        args.into_iter().map(OsString::from).collect()
    }
}

/// Run FFmpeg silencedetect filter and return raw stderr output.
pub async fn detect_silence(
    config: &AppConfig,
    media_path: &Path,
    noise_db: f64,
    min_duration: f64,
) -> Result<String, ToolError> {
    let inv = ToolInvocation {
        program: config.tools.ffmpeg.clone(),
        args: vec![
            OsString::from("-i"),
            OsString::from(media_path),
            OsString::from("-af"),
            OsString::from(format!(
                "silencedetect=noise={noise_db}dB:d={min_duration}"
            )),
            OsString::from("-f"),
            OsString::from("null"),
            OsString::from("-"),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(300)),
    };

    let output = run_tool(&inv).await?;
    Ok(output.stderr)
}

/// Split audio at specified time points, outputting WAV chunks.
/// Returns the list of output chunk paths.
pub async fn split_audio(
    config: &AppConfig,
    media_path: &Path,
    split_points: &[f64],
    output_dir: &Path,
    stem: &str,
) -> Result<Vec<PathBuf>, ToolError> {
    let mut chunks = Vec::new();
    let mut start = 0.0;

    for (i, &split_at) in split_points.iter().enumerate() {
        let chunk_path = output_dir.join(format!("{stem}_chunk_{i:03}.wav"));
        let duration = split_at - start;

        let inv = ToolInvocation {
            program: config.tools.ffmpeg.clone(),
            args: vec![
                OsString::from("-y"),
                OsString::from("-i"),
                OsString::from(media_path),
                OsString::from("-ss"),
                OsString::from(format!("{start:.3}")),
                OsString::from("-t"),
                OsString::from(format!("{duration:.3}")),
                OsString::from("-ar"),
                OsString::from("16000"),
                OsString::from("-ac"),
                OsString::from("1"),
                OsString::from("-c:a"),
                OsString::from("pcm_s16le"),
                OsString::from(&chunk_path),
            ],
            working_dir: None,
            env_overrides: Vec::new(),
            timeout: Some(Duration::from_secs(120)),
        };

        run_tool(&inv).await?;
        validate_output(&chunk_path, "ffmpeg", &format!("audio chunk {i}"))?;
        chunks.push(chunk_path);
        start = split_at;
    }

    // Final chunk (from last split point to end)
    let i = split_points.len();
    let chunk_path = output_dir.join(format!("{stem}_chunk_{i:03}.wav"));
    let inv = ToolInvocation {
        program: config.tools.ffmpeg.clone(),
        args: vec![
            OsString::from("-y"),
            OsString::from("-i"),
            OsString::from(media_path),
            OsString::from("-ss"),
            OsString::from(format!("{start:.3}")),
            OsString::from("-ar"),
            OsString::from("16000"),
            OsString::from("-ac"),
            OsString::from("1"),
            OsString::from("-c:a"),
            OsString::from("pcm_s16le"),
            OsString::from(&chunk_path),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(120)),
    };

    run_tool(&inv).await?;
    validate_output(&chunk_path, "ffmpeg", &format!("audio chunk {i} (final)"))?;
    chunks.push(chunk_path);

    Ok(chunks)
}

/// How many non-progress stderr lines are kept for the error report when
/// ffmpeg exits non-zero. ffmpeg's banner is ~15 lines and the actual
/// complaint is always at the very end, so a short tail is enough.
const STDERR_TAIL_LINES: usize = 12;

/// Quote a value for use inside an ffmpeg filtergraph argument.
///
/// A `-vf` string is parsed twice. The graph parser strips one level of
/// `'...'` quoting and splits on `[ ] , ;`; the filter's own option parser
/// then splits the result on `:` and honours `\` escapes and `'` quotes a
/// second time. Wrapping the value in single quotes protects everything
/// from the first pass; two characters still need work for the second:
///
/// * `:`  -> `\:`      (inside quotes the backslash survives pass one, so
///   pass two sees `\:` and un-escapes it)
/// * `'`  -> `'\\\''`  (close the quote, emit `\\` + `\'` so pass one
///   yields `\'` and pass two yields `'`, then reopen the quote)
///
/// A bare `'` is exactly how "What's Wrong With Society" broke the
/// hardburn: it terminated the quoted filename early and the rest of the
/// path was parsed as graph syntax ("Trailing garbage after a filter").
pub fn filter_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for c in value.chars() {
        match c {
            ':' => out.push_str("\\:"),
            '\'' => out.push_str("'\\\\\\''"),
            _ => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// Build the `filename=` argument of the `subtitles` filter from a path.
///
/// Backslashes become forward slashes first (the option parser treats `\`
/// as an escape character), then the result is quoted by [`filter_quote`].
/// The `\\?\` extended-length prefix that `std::fs::canonicalize` adds on
/// Windows becomes `//?/`, which ffmpeg's file protocol opens fine.
pub fn subtitles_filter_path(path: &Path) -> String {
    filter_quote(&path.to_string_lossy().replace('\\', "/"))
}

/// True when ffmpeg's stderr shows it never got past *building* the
/// `subtitles` filter -- the subtitle path was rejected, not the encode.
/// Used to decide whether retrying from a sanitised copy of the SRT could
/// help. Encoder, NVENC and disk failures deliberately do not match.
pub fn stderr_indicates_subtitle_path_failure(stderr: &str) -> bool {
    const MARKERS: &[&str] = &[
        "Error parsing filterchain",
        "Trailing garbage after a filter",
        "No such filter",
        "Error initializing filter 'subtitles'",
        "Parsed_subtitles",
        "Unable to open",
    ];
    MARKERS.iter().any(|m| stderr.contains(m))
}

/// [`stderr_indicates_subtitle_path_failure`] lifted to a [`ToolError`].
pub fn is_subtitle_path_failure(err: &ToolError) -> bool {
    match err {
        ToolError::NonZeroExit { stderr, .. } => stderr_indicates_subtitle_path_failure(stderr),
        _ => false,
    }
}

/// Build the hardware-encoded hardburn invocation without running it.
pub fn hardburn_invocation(
    ffmpeg: &Path,
    video_path: &Path,
    srt_path: &Path,
    output_path: &Path,
    style: &str,
    cq: &str,
    encoder: VideoEncoder,
) -> ToolInvocation {
    let vf = format!(
        "subtitles=filename={}:force_style={}",
        subtitles_filter_path(srt_path),
        filter_quote(style)
    );

    let mut args = vec![
        OsString::from("-y"),
        OsString::from("-hwaccel"),
        OsString::from(encoder.hwaccel()),
        OsString::from("-i"),
        OsString::from(video_path),
        OsString::from("-vf"),
        OsString::from(&vf),
    ];
    args.extend(encoder.codec_args(cq));
    args.extend([
        OsString::from("-c:a"),
        OsString::from("copy"),
        OsString::from("-movflags"),
        OsString::from("+faststart"),
        OsString::from(output_path),
    ]);

    ToolInvocation {
        program: ffmpeg.to_path_buf(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(7200)),
    }
}

/// Hardburn subtitles into video using the configured hardware encoder.
///
/// Every tool line is forwarded to `tx` for live progress. On a non-zero
/// exit the last few non-progress stderr lines are returned inside
/// [`ToolError::NonZeroExit`], so the caller can both show *why* ffmpeg
/// bailed and decide whether a retry makes sense. Previously the exit
/// code was ignored and the only symptom was "Missing output".
pub async fn hardburn(
    config: &AppConfig,
    video_path: &Path,
    srt_path: &Path,
    output_path: &Path,
    style: &str,
    tx: mpsc::UnboundedSender<ToolLine>,
) -> Result<PathBuf, ToolError> {
    let cq = config.cq.to_string();
    let inv = hardburn_invocation(
        &config.tools.ffmpeg,
        video_path,
        srt_path,
        output_path,
        style,
        &cq,
        config.encoder,
    );

    // Tee: forward everything to the caller, keep a tail of stderr.
    let (tee_tx, mut tee_rx) = mpsc::unbounded_channel::<ToolLine>();
    let tee = tokio::spawn(async move {
        let mut tail: VecDeque<String> = VecDeque::with_capacity(STDERR_TAIL_LINES);
        while let Some(line) = tee_rx.recv().await {
            if let ToolLine::Stderr(ref l) = line {
                if !l.trim().is_empty() && parse_ffmpeg_progress(l).is_none() {
                    if tail.len() == STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(l.clone());
                }
            }
            let done = matches!(line, ToolLine::Done { .. });
            let _ = tx.send(line);
            if done {
                break;
            }
        }
        tail
    });

    let code = run_tool_streaming(&inv, tee_tx).await?;
    let tail = tee.await.unwrap_or_default();

    if code != Some(0) {
        return Err(ToolError::NonZeroExit {
            tool: "ffmpeg".to_string(),
            code: code.unwrap_or(-1),
            stderr: tail.into_iter().collect::<Vec<_>>().join("\n"),
        });
    }

    validate_output(output_path, "ffmpeg", "hardcoded video")?;
    Ok(output_path.to_path_buf())
}

/// Parse FFmpeg progress output from a stderr line.
/// Returns (current_time_secs, speed_multiplier) if the line contains progress info.
pub fn parse_ffmpeg_progress(line: &str) -> Option<(f64, f64)> {
    // Example: "frame= 1234 fps=60.3 q=17.0 size=  128000kB time=00:03:45.12 bitrate=4660.2kbits/s speed=7.86x"
    let time_re = Regex::new(r"time=(\d+):(\d+):(\d+)\.(\d+)").ok()?;
    let speed_re = Regex::new(r"speed=\s*([\d.]+)x").ok()?;

    let time_caps = time_re.captures(line)?;
    let h: f64 = time_caps[1].parse().ok()?;
    let m: f64 = time_caps[2].parse().ok()?;
    let s: f64 = time_caps[3].parse().ok()?;
    let cs: f64 = time_caps[4].parse().ok()?;
    let current = h * 3600.0 + m * 60.0 + s + cs / 100.0;

    let speed = speed_re
        .captures(line)
        .and_then(|c| c[1].parse::<f64>().ok())
        .unwrap_or(1.0);

    Some((current, speed))
}

/// Decode + resample any source media to a single 16kHz mono PCM s16le WAV.
/// This "master" WAV is the cheap intermediate that subsequent silence-detect
/// and chunk-splitting both operate on, so the source codec is decoded ONCE
/// per pipeline run instead of N+1 times.
///
/// When `hwaccel` is `Some("cuda")` (or any other accelerator name), passes
/// `-hwaccel <name>` BEFORE `-i` so NVDEC handles the video decode path on
/// supported codecs (H.264, HEVC, AV1). ffmpeg ignores `-hwaccel` on pure-
/// audio inputs, so leaving it on for everything is safe.
pub async fn resample_to_master_wav(
    config: &AppConfig,
    src: &Path,
    dest: &Path,
    hwaccel: Option<&str>,
) -> Result<PathBuf, ToolError> {
    let mut args: Vec<OsString> = vec![OsString::from("-y")];
    if let Some(h) = hwaccel.filter(|s| !s.is_empty()) {
        args.push(OsString::from("-hwaccel"));
        args.push(OsString::from(h));
    }
    args.extend([
        OsString::from("-i"),
        OsString::from(src),
        // Explicit stream selection: only the first audio stream, no video,
        // no subtitle, no data. Prevents ffmpeg from instantiating decoders
        // for streams we'll just throw away on long video sources.
        OsString::from("-map"),
        OsString::from("0:a:0"),
        OsString::from("-vn"),
        OsString::from("-sn"),
        OsString::from("-dn"),
        OsString::from("-ar"),
        OsString::from("16000"),
        OsString::from("-ac"),
        OsString::from("1"),
        OsString::from("-c:a"),
        OsString::from("pcm_s16le"),
        OsString::from(dest),
    ]);

    let inv = ToolInvocation {
        program: config.tools.ffmpeg.clone(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(1800)),
    };

    if config.prkt.low_priority_decode {
        run_tool_lowprio(&inv).await?;
    } else {
        run_tool(&inv).await?;
    }
    validate_output(dest, "ffmpeg", "master WAV resample")?;
    Ok(dest.to_path_buf())
}

/// Split a master WAV at the given timestamps using ffmpeg's segment muxer
/// with `-c copy`. PCM is uncompressed so "copy" is a literal byte-slice
/// with a fresh RIFF header per chunk -- one decode pass for the whole
/// split, vs N decodes in the old `split_audio` path.
///
/// `split_points` are cumulative seconds from the start (the same shape
/// `compute_split_points` produces). Emits N+1 numbered files matching the
/// `{stem}_chunk_{:03}.wav` pattern.
pub async fn split_master_wav(
    config: &AppConfig,
    master_wav: &Path,
    split_points: &[f64],
    output_dir: &Path,
    stem: &str,
) -> Result<Vec<PathBuf>, ToolError> {
    let pattern = output_dir.join(format!("{stem}_chunk_%03d.wav"));

    let mut args: Vec<OsString> = vec![
        OsString::from("-y"),
        OsString::from("-i"),
        OsString::from(master_wav),
        OsString::from("-c"),
        OsString::from("copy"),
        OsString::from("-f"),
        OsString::from("segment"),
    ];
    if !split_points.is_empty() {
        let segment_times = split_points
            .iter()
            .map(|t| format!("{t:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        args.push(OsString::from("-segment_times"));
        args.push(OsString::from(segment_times));
    }
    args.push(OsString::from(&pattern));

    let inv = ToolInvocation {
        program: config.tools.ffmpeg.clone(),
        args,
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(600)),
    };
    run_tool(&inv).await?;

    let expected = split_points.len() + 1;
    let mut chunks = Vec::with_capacity(expected);
    for i in 0..expected {
        let p = output_dir.join(format!("{stem}_chunk_{i:03}.wav"));
        validate_output(&p, "ffmpeg", &format!("audio chunk {i}"))?;
        chunks.push(p);
    }
    Ok(chunks)
}

/// Convert media to mono 16kHz WAV for Parakeet (prkt expects this format).
pub async fn convert_to_wav(
    config: &AppConfig,
    media_path: &Path,
    output_path: &Path,
) -> Result<PathBuf, ToolError> {
    let inv = ToolInvocation {
        program: config.tools.ffmpeg.clone(),
        args: vec![
            OsString::from("-y"),
            OsString::from("-i"),
            OsString::from(media_path),
            OsString::from("-ar"),
            OsString::from("16000"),
            OsString::from("-ac"),
            OsString::from("1"),
            OsString::from("-c:a"),
            OsString::from("pcm_s16le"),
            OsString::from(output_path),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(300)),
    };

    run_tool(&inv).await?;
    validate_output(output_path, "ffmpeg", "WAV conversion")?;
    Ok(output_path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_quote_wraps_and_escapes_colon() {
        assert_eq!(filter_quote("D:/x.srt"), "'D\\:/x.srt'");
    }

    #[test]
    fn filter_quote_escapes_apostrophe_for_both_parse_passes() {
        // close quote, `\\` + `\'`, reopen quote
        assert_eq!(filter_quote("What's"), "'What'\\\\\\''s'");
    }

    #[test]
    fn filter_quote_leaves_style_strings_untouched() {
        let style = "FontName=Arial,FontSize=16.5,PrimaryColour=&H80FFFFFF,BorderStyle=4";
        assert_eq!(filter_quote(style), format!("'{style}'"));
    }

    #[test]
    fn subtitles_filter_path_handles_the_real_offender() {
        // The exact yt-dlp name that broke the hardburn, as canonicalize hands it over.
        let p = Path::new(
            "\\\\?\\D:\\YTDL.cmd\\Mr. Robot\\137+140\\source\\What's Wrong With Society ｜ Mr. Robot.[20220927].[137+140].[jhxqDkaxoZU].FWXL-Transcribe.U.large-v2_en.srt",
        );
        assert_eq!(
            subtitles_filter_path(p),
            "'//?/D\\:/YTDL.cmd/Mr. Robot/137+140/source/What'\\\\\\''s Wrong With Society ｜ Mr. Robot.[20220927].[137+140].[jhxqDkaxoZU].FWXL-Transcribe.U.large-v2_en.srt'"
        );
    }

    #[test]
    fn hardburn_invocation_builds_expected_vf() {
        let inv = hardburn_invocation(
            Path::new("C:\\ff\\ffmpeg.exe"),
            Path::new("D:\\v\\a.mp4"),
            Path::new("D:\\v\\a's.srt"),
            Path::new("D:\\v\\a.hardcoded.mp4"),
            "FontName=Arial",
            "19",
            VideoEncoder::Nvenc,
        );
        let args: Vec<String> = inv
            .args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let vf_idx = args.iter().position(|a| a == "-vf").unwrap();
        assert_eq!(
            args[vf_idx + 1],
            "subtitles=filename='D\\:/v/a'\\\\\\''s.srt':force_style='FontName=Arial'"
        );
        assert_eq!(args[vf_idx + 3], "h264_nvenc");
        assert_eq!(args.last().unwrap(), "D:\\v\\a.hardcoded.mp4");
    }

    /// The NVENC argument list is the one that has been burning real videos;
    /// making the encoder selectable must not have changed a single token.
    #[test]
    fn nvenc_invocation_is_unchanged_by_encoder_selection() {
        let inv = hardburn_invocation(
            Path::new("ffmpeg"),
            Path::new("in.mp4"),
            Path::new("in.srt"),
            Path::new("out.mp4"),
            "FontName=Arial",
            "19",
            VideoEncoder::Nvenc,
        );
        let args: Vec<String> = inv
            .args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-y",
                "-hwaccel",
                "cuda",
                "-i",
                "in.mp4",
                "-vf",
                "subtitles=filename='in.srt':force_style='FontName=Arial'",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-cq",
                "19",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                "out.mp4",
            ]
        );
    }

    #[test]
    fn videotoolbox_maps_cq_onto_its_inverted_quality_scale() {
        let args: Vec<String> = VideoEncoder::VideoToolbox
            .codec_args("19")
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["-c:v", "h264_videotoolbox", "-q:v", "62"]);
        // Out-of-range input clamps instead of producing an invalid -q:v.
        let worst = VideoEncoder::VideoToolbox.codec_args("51");
        assert_eq!(worst.last().unwrap(), "1");
    }

    #[test]
    fn encoder_default_follows_the_platform() {
        let expected = if cfg!(target_os = "macos") {
            VideoEncoder::VideoToolbox
        } else {
            VideoEncoder::Nvenc
        };
        assert_eq!(VideoEncoder::default(), expected);
    }

    #[test]
    fn subtitle_path_failure_classification() {
        let broken_quote = "[AVFilterGraph @ 0x1] Trailing garbage after a filter: .[137+140].[x].srt':force_style=...\n[AVFilterGraph @ 0x1] Error parsing filterchain ...\nError opening output files: Invalid argument";
        assert!(stderr_indicates_subtitle_path_failure(broken_quote));

        let missing_file = "[Parsed_subtitles_0 @ 0x1] Unable to open D:/x/y.srt\n[AVFilterGraph @ 0x1] Error initializing filter 'subtitles' with args '...'\n";
        assert!(stderr_indicates_subtitle_path_failure(missing_file));

        let nvenc = "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)\n[vost#0:0/h264_nvenc @ 0x1] Error while opening encoder - maybe incorrect parameters such as bit_rate, rate, width or height.\n";
        assert!(!stderr_indicates_subtitle_path_failure(nvenc));

        let disk_full = "av_interleaved_write_frame(): No space left on device\n[out#0/mp4 @ 0x1] Error muxing a packet\n";
        assert!(!stderr_indicates_subtitle_path_failure(disk_full));

        assert!(is_subtitle_path_failure(&ToolError::NonZeroExit {
            tool: "ffmpeg".into(),
            code: -22,
            stderr: broken_quote.into(),
        }));
        assert!(!is_subtitle_path_failure(&ToolError::Timeout {
            tool: "ffmpeg".into(),
            timeout_secs: 1,
        }));
    }

    /// End-to-end against a real ffmpeg and a real video whose subtitle name
    /// contains an apostrophe (the case that used to break the filtergraph).
    /// Ignored by default: needs a hardware encoder and local media. Point it
    /// at a video with a sibling `<stem>.srt`:
    ///   FWXL_TEST_VIDEO=/path/to/it's.mp4 cargo test --release -- --ignored hardburn_real
    /// ffmpeg is taken from FWXL_TEST_FFMPEG, else from PATH.
    #[tokio::test]
    #[ignore]
    async fn hardburn_real_apostrophe_file_two_seconds() {
        let video = std::env::var_os("FWXL_TEST_VIDEO").expect("set FWXL_TEST_VIDEO");
        let video = std::fs::canonicalize(video).expect("test media present");
        let srt = std::fs::canonicalize(video.with_extension("srt")).expect("test srt present");
        let ffmpeg = std::env::var_os("FWXL_TEST_FFMPEG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("ffmpeg"));
        let ffmpeg = ffmpeg.as_path();
        let out = std::env::temp_dir().join("fwxl-burn-hardburn-real-test.mp4");
        let _ = std::fs::remove_file(&out);

        let mut inv = hardburn_invocation(
            ffmpeg,
            &video,
            &srt,
            &out,
            "FontName=Arial,FontSize=16.5",
            "19",
            VideoEncoder::default(),
        );
        // Only encode two seconds: insert `-t 2` before the output path.
        let n = inv.args.len();
        inv.args.insert(n - 1, OsString::from("2"));
        inv.args.insert(n - 1, OsString::from("-t"));

        let (tx, mut rx) = mpsc::unbounded_channel();
        let code = run_tool_streaming(&inv, tx).await.expect("spawn");
        let mut stderr = String::new();
        while let Some(line) = rx.recv().await {
            if let ToolLine::Stderr(l) = line {
                stderr.push_str(&l);
                stderr.push('\n');
            }
        }
        assert_eq!(code, Some(0), "ffmpeg failed:\n{stderr}");
        assert!(out.metadata().map(|m| m.len() > 0).unwrap_or(false));
        let _ = std::fs::remove_file(&out);
    }
}
