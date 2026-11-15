use crate::error::PipelineError;
use crate::naming;
use crate::pipeline::salvage;
use crate::pipeline::{PipelineContext, PipelineEvent, PipelineResult};
use crate::srt;
use crate::tools::ToolLine;
use crate::tools::whisper_xxl::{self, WhisperMode};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// Cap on the retained run log used for out-of-memory detection. Long runs
/// emit a lot of progress chatter and the whole thing is only ever substring
/// searched, so keeping the tail is sufficient and bounds memory.
const MAX_RUN_LOG_BYTES: usize = 256 * 1024;

/// Parse a Whisper-style segment timestamp line and return the END time in seconds.
/// Matches `[HH:MM:SS.sss --> HH:MM:SS.sss]` and `[MM:SS.sss --> MM:SS.sss]`.
/// Returns None for non-matching lines.
fn parse_segment_end_secs(line: &str) -> Option<f64> {
    let open = line.find('[')?;
    let close = line[open..].find(']')?;
    let inside = &line[open + 1..open + close];
    let arrow = inside.find("-->")?;
    let end = inside[arrow + 3..].trim();
    parse_timestamp_secs(end)
}

/// Parse "MM:SS.sss" or "HH:MM:SS.sss" into seconds.
fn parse_timestamp_secs(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        2 => {
            let m: f64 = parts[0].parse().ok()?;
            let sec: f64 = parts[1].parse().ok()?;
            Some(m * 60.0 + sec)
        }
        3 => {
            let h: f64 = parts[0].parse().ok()?;
            let m: f64 = parts[1].parse().ok()?;
            let sec: f64 = parts[2].parse().ok()?;
            Some(h * 3600.0 + m * 60.0 + sec)
        }
        _ => None,
    }
}

fn to_whisper_mode(mode: crate::pipeline::TranscriptionMode) -> WhisperMode {
    match mode {
        crate::pipeline::TranscriptionMode::TestWhisper => WhisperMode::Test,
        crate::pipeline::TranscriptionMode::Extended => WhisperMode::Smooth,
        crate::pipeline::TranscriptionMode::Quality => WhisperMode::Quality,
        crate::pipeline::TranscriptionMode::Inclusive => WhisperMode::Inclusive,
        crate::pipeline::TranscriptionMode::Unfiltered => WhisperMode::Unfiltered,
        crate::pipeline::TranscriptionMode::Chunked => WhisperMode::Chunked,
        _ => WhisperMode::Default,
    }
}

/// List the regular files a run wrote into its temp dir.
///
/// Everything under the temp dir is, by construction, this run's output and
/// nobody else's -- no fuzzy stem matching, no dodging legacy files, and no
/// chance of a concurrent fwxl-burn instance touching these paths.
fn read_output_files(dir: &Path) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn find_by_ext(outputs: &[PathBuf], ext: &str) -> Option<PathBuf> {
    outputs
        .iter()
        .find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
        })
        .cloned()
}

/// Run whisper once into `out_dir`, streaming progress to the TUI.
///
/// Returns the exit code and the tail of the combined output, which the caller
/// scans to decide whether a failure was memory exhaustion (worth retrying with
/// lighter settings) or something else (not worth burning another hour on).
async fn run_whisper_attempt(
    ctx: &PipelineContext,
    out_dir: &Path,
    preset: &str,
    total_secs: f64,
    event_tx: &mpsc::UnboundedSender<PipelineEvent>,
) -> Result<(Option<i32>, String), PipelineError> {
    let (tool_tx, mut tool_rx) = mpsc::unbounded_channel::<ToolLine>();
    let event_tx2 = event_tx.clone();

    let forward_handle = tokio::spawn(async move {
        // Throttle Progress events to at most ~10/sec to avoid flooding the TUI.
        let mut last_progress = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or_else(std::time::Instant::now);
        let mut last_pct: f32 = 0.0;
        let mut log = String::new();

        // Both streams carry segment lines depending on the whisper build, and
        // both can carry the OOM message, so both are mirrored into `log`.
        let mut note = |l: &str, log: &mut String| {
            if log.len() < MAX_RUN_LOG_BYTES {
                log.push_str(l);
                log.push('\n');
            }
        };

        while let Some(line) = tool_rx.recv().await {
            let (text, is_stdout) = match line {
                ToolLine::Stdout(l) => (l, true),
                ToolLine::Stderr(l) => (l, false),
                ToolLine::Done { .. } => break,
            };

            note(&text, &mut log);

            if total_secs > 0.0 {
                if let Some(end) = parse_segment_end_secs(&text) {
                    let pct = ((end / total_secs) * 100.0).clamp(0.0, 100.0) as f32;
                    if pct >= last_pct
                        && last_progress.elapsed() >= std::time::Duration::from_millis(100)
                    {
                        let _ = event_tx2.send(PipelineEvent::Progress {
                            percent: pct,
                            message: format!("Transcribed {end:.0}s / {total_secs:.0}s"),
                        });
                        last_progress = std::time::Instant::now();
                        last_pct = pct;
                    }
                }
            }

            let _ = event_tx2.send(if is_stdout {
                PipelineEvent::StdoutLine(text)
            } else {
                PipelineEvent::StderrLine(text)
            });
        }
        log
    });

    let exit_code = whisper_xxl::run_whisper_streaming_with_preset(
        &ctx.config,
        &ctx.media.path,
        out_dir,
        &ctx.selected_model,
        preset,
        tool_tx,
    )
    .await
    .map_err(|e| {
        let _ = event_tx.send(PipelineEvent::Error(format!("Whisper failed to start: {e}")));
        PipelineError::ToolFailed {
            phase: "transcription".to_string(),
            source: Box::new(e),
        }
    })?;

    let log = forward_handle.await.unwrap_or_default();
    Ok((exit_code, log))
}

/// Run a Whisper transcription pipeline (single file).
pub async fn run_whisper_pipeline(
    ctx: &PipelineContext,
    event_tx: mpsc::UnboundedSender<PipelineEvent>,
) -> Result<PipelineResult, PipelineError> {
    let whisper_mode = to_whisper_mode(ctx.mode);
    let mode_tag = ctx.mode.mode_tag().to_string();

    let dur_str = ctx
        .media
        .duration
        .map(|d| format!("{:.0}s", d.as_secs_f64()))
        .unwrap_or_else(|| "unknown duration".to_string());

    // Without a faster-whisper-xxl binary there is nothing to run. Say so
    // up front instead of failing later with an opaque spawn error.
    if !ctx.config.tools.has_whisper() {
        return Err(PipelineError::ModeUnavailable {
            mode: mode_tag,
            reason: "faster-whisper-xxl was not found. Put it on PATH, unpack it into a \
                     Faster-Whisper-XXL folder beside fwxl-burn, or set tools.whisper in \
                     the config"
                .to_string(),
        });
    }

    // Phase 1: Transcribe
    let _ = event_tx.send(PipelineEvent::PhaseStarted {
        name: format!(
            "Transcribing with Faster-Whisper (model={}, mode={}, {})",
            ctx.selected_model, mode_tag, dur_str
        ),
        phase_num: 1,
        total_phases: 2,
    });

    // Log the command being run
    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Whisper binary: {}",
        ctx.config.tools.whisper.display()
    )));
    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Model: {}",
        ctx.selected_model
    )));
    // Pre-flight: locate the model on disk so the user sees where we looked.
    match whisper_xxl::find_model_dir(&ctx.config, &ctx.selected_model) {
        Some(p) => {
            let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                "Model dir found: {}",
                p.display()
            )));
        }
        None => {
            let _ = event_tx.send(PipelineEvent::Warning(format!(
                "Model '{}' not found in any known location; faster-whisper-xxl may try to download it",
                ctx.selected_model
            )));
        }
    }
    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Final output dir: {}",
        ctx.srt_dir.display()
    )));

    let total_secs = ctx
        .media
        .duration
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    // A run can fail in three ways that all used to end the same way: an error
    // return, and `TempDir::drop` deleting every file whisper had written.
    // Hours of transcription vanished because one of five output writers
    // produced an empty file. The loop below never discards work:
    //   1. a usable .srt is used as-is;
    //   2. otherwise the SRT is rebuilt from the .json/.vtt/.tsv siblings;
    //   3. otherwise, if the run died on memory pressure, it is retried once
    //      with the VRAM-heavy stages stripped out;
    //   4. and if all of that fails the temp dir is *moved somewhere
    //      permanent* rather than swept, so the raw output survives.
    let base_preset = whisper_xxl::preset_args_for(&ctx.config, whisper_mode);
    let degraded_preset = salvage::degrade_preset(&base_preset);

    let mut salvage_warnings: Vec<String> = Vec::new();
    let mut attempt: u32 = 0;

    let (tmp_dir, outputs, exit_code) = loop {
        attempt += 1;
        let preset = if attempt == 1 {
            base_preset.clone()
        } else {
            degraded_preset.clone().unwrap_or_else(|| base_preset.clone())
        };

        // CONCURRENCY ISOLATION:
        // Run whisper into a per-invocation temp dir created as a sibling of
        // the final output dir (same volume -> rename is a fast metadata move).
        // - Two concurrent fwxl-burn processes get distinct random tempdirs;
        //   neither can stomp the other's in-flight output files.
        // - Legacy raw files left over from older runs in `srt_dir` cannot be
        //   confused with this run's output (they're in different directories).
        // The `.fwxl-tmp-` prefix makes orphans (after a hard kill) trivial to
        // identify and clean up manually.
        // Ensure the output dir exists before asking for a tempdir inside it.
        let _ = std::fs::create_dir_all(&ctx.srt_dir);
        let this_tmp = match tempfile::Builder::new()
            .prefix(".fwxl-tmp-")
            .tempdir_in(&ctx.srt_dir)
        {
            Ok(t) => t,
            Err(e) => {
                // Fallback if srt_dir is unwritable (read-only, FULL disk,
                // special filesystem). Say WHY out loud: this used to swallow
                // the reason, and "output drive is full" then surfaced an
                // hour later as a wall of cryptic cross-drive move failures
                // (os error 17) instead of one clear warning here.
                let _ = event_tx.send(PipelineEvent::Warning(format!(
                    "Cannot create temp dir inside {} ({e}). Falling back to the \
                     system temp dir; delivery becomes a cross-drive copy. If the \
                     output drive is out of space, free some now or that copy \
                     will fail as well.",
                    ctx.srt_dir.display()
                )));
                tempfile::Builder::new()
                    .prefix(".fwxl-tmp-")
                    .tempdir()
                    .map_err(|e| {
                        PipelineError::SrtParse(format!("Failed to create temp dir: {e}"))
                    })?
            }
        };

        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "Whisper output (isolated temp){}: {}",
            if attempt > 1 {
                format!(", attempt {attempt}")
            } else {
                String::new()
            },
            this_tmp.path().display()
        )));

        let (exit_code, run_log) = run_whisper_attempt(
            ctx,
            this_tmp.path(),
            &preset,
            total_secs,
            &event_tx,
        )
        .await?;

        // Phase 2: Validate and rename output
        let _ = event_tx.send(PipelineEvent::PhaseStarted {
            name: "Validating output and renaming".to_string(),
            phase_num: 2,
            total_phases: 2,
        });
        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "Whisper exit code: {exit_code:?}"
        )));

        let mut found = read_output_files(this_tmp.path());

        // (1) Whisper's own SRT, if it is actually usable.
        if let Some(p) = find_by_ext(&found, "srt") {
            if std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) > 0 {
                break (this_tmp, found, exit_code);
            }
            let _ = event_tx.send(PipelineEvent::Warning(format!(
                "Whisper's SRT is empty (0 bytes): {}",
                p.file_name().unwrap_or_default().to_string_lossy()
            )));
        } else if !found.is_empty() {
            let _ = event_tx.send(PipelineEvent::Warning(
                "Whisper wrote output but no .srt file".to_string(),
            ));
        }

        // (2) Rebuild the SRT from a sibling format. These carry real
        //     per-segment timings, so the result is a true subtitle file.
        if let Some((entries, source)) = salvage::salvage_entries(&found) {
            let rebuilt = this_tmp.path().join(format!("{}.srt", ctx.media.stem));
            match srt::write_srt_file(&entries, &rebuilt) {
                Ok(()) => {
                    let msg = format!(
                        "Recovered {} subtitle entries from {} after the SRT came back unusable",
                        entries.len(),
                        source.label()
                    );
                    let _ = event_tx.send(PipelineEvent::Warning(msg.clone()));
                    salvage_warnings.push(msg);
                    found = read_output_files(this_tmp.path());
                    break (this_tmp, found, exit_code);
                }
                Err(e) => {
                    let _ = event_tx.send(PipelineEvent::Warning(format!(
                        "Could not write recovered SRT: {e}"
                    )));
                }
            }
        }

        // (3) Nothing to salvage. If this looks like memory exhaustion, retry
        //     once without the stages that consume the VRAM.
        let oom = salvage::looks_like_oom(&run_log);
        if attempt == 1 && oom {
            if let Some(ref degraded) = degraded_preset {
                let _ = event_tx.send(PipelineEvent::Warning(
                    "Run ran out of memory and produced nothing usable. Retrying once \
                     without vocal extraction / diarization."
                        .to_string(),
                ));
                let _ = event_tx.send(PipelineEvent::StderrLine(format!(
                    "  Retry preset: {degraded}"
                )));
                salvage_warnings.push(
                    "First attempt ran out of memory; retried with reduced settings".to_string(),
                );
                drop(this_tmp); // nothing usable inside; safe to sweep
                continue;
            }
        }

        // (4) Give up, but preserve everything for inspection.
        let kept = this_tmp.keep();
        let detail = match salvage::quarantine(&kept, &ctx.srt_dir, &ctx.media.stem) {
            Ok(dest) => {
                let _ = event_tx.send(PipelineEvent::Error(format!(
                    "No usable transcript. Raw output preserved at: {}",
                    dest.display()
                )));
                format!("raw output preserved at {}", dest.display())
            }
            Err(e) => {
                let _ = event_tx.send(PipelineEvent::Error(format!(
                    "No usable transcript, and preserving the temp dir failed: {e}. \
                     It is still on disk at: {}",
                    kept.display()
                )));
                format!("temp dir left at {}", kept.display())
            }
        };

        return Err(PipelineError::TranscriptionUnusable {
            reason: if oom {
                "whisper ran out of memory and wrote no recoverable output".to_string()
            } else {
                "whisper produced no .srt and no recoverable sibling output".to_string()
            },
            detail,
        });
    };

    // The loop only breaks once a non-empty SRT exists here, whether whisper
    // wrote it or it was rebuilt from a sibling format; the rename loop below
    // finds it in `outputs`. Guard anyway so a logic change upstream cannot
    // silently produce an SRT-less success.
    find_by_ext(&outputs, "srt").ok_or(PipelineError::NoSrtOutput)?;

    // Extract language from JSON sidecar if whisper emitted one. Default "en"
    // matches prior behavior.
    let lang = outputs
        .iter()
        .find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        })
        .and_then(|p| srt::extract_language_from_json(p))
        .unwrap_or_else(|| "en".to_string());

    let prefix = &ctx.config.naming.transcribe_prefix;

    // Move every file out of tempdir into srt_dir under its canonical name.
    // rename_with_retry handles transient Windows sharing violations from
    // whisper-xxl subprocesses still releasing handles. Same-volume moves
    // (default; tempdir is sibling of srt_dir) are atomic metadata ops.
    //
    // Quality (Q) mode adds `--diarize <model>` + `--return_embeddings`, which
    // produces per-speaker embedding files. Two paths exist depending on the
    // whisper-xxl build:
    //   1. Numpy variants (.npy/.npz/.pkl) land in the tempdir -- handled
    //      below by the in-loop `is_embedding_file` branch.
    //   2. Per-speaker `<audio>.<ext>_diarize_<model>_SPEAKER_<NN>.txt` files
    //      are dropped NEXT TO THE SOURCE AUDIO, bypassing `--output_dir`
    //      entirely -- collected via `find_diarize_embeddings` after the loop.
    // Either way the user's SRT directory stays clean and embeddings live
    // under `embeddings/` (sibling of `source/`, or a subdir of `srt_dir`).
    let embeddings_dir = naming::resolve_embeddings_dir(&ctx.srt_dir);
    let mut embeddings_dir_created = false;
    let mut final_srt: Option<PathBuf> = None;
    let mut moved_count: usize = 0;
    let mut embeddings_moved: usize = 0;
    let mut failures: Vec<(PathBuf, std::io::Error)> = Vec::new();

    for src in &outputs {
        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext.is_empty() {
            // Unexpected -- log and skip rather than risk overwriting something.
            let _ = event_tx.send(PipelineEvent::Warning(format!(
                "Skipping extensionless file in temp dir: {}",
                src.display()
            )));
            continue;
        }
        let src_name = src
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let is_embedding = naming::is_embedding_file(src_name);
        // Pick the destination dir based on file type. The embeddings dir is
        // created lazily so non-Q runs never leave an empty `embeddings/` behind.
        let dest_dir: &PathBuf = if is_embedding {
            if !embeddings_dir_created {
                match std::fs::create_dir_all(&embeddings_dir) {
                    Ok(()) => embeddings_dir_created = true,
                    Err(e) => {
                        let _ = event_tx.send(PipelineEvent::Warning(format!(
                            "Failed to create embeddings dir {}: {e}; \
                             placing alongside SRT instead",
                            embeddings_dir.display()
                        )));
                    }
                }
            }
            if embeddings_dir_created {
                &embeddings_dir
            } else {
                &ctx.srt_dir
            }
        } else {
            &ctx.srt_dir
        };
        // Embedding files keep their original filename: the diarize-model +
        // SPEAKER_NN suffix already encodes everything needed for uniqueness,
        // and the canonical name format would collapse N per-speaker files
        // into one collision target.
        let new_name = if is_embedding {
            src_name.to_string()
        } else {
            format!(
                "{}.{}.{}.{}_{}.{}",
                ctx.media.stem, prefix, mode_tag, ctx.selected_model, lang, ext
            )
        };
        let dst = dest_dir.join(&new_name);
        match naming::rename_with_retry(src, &dst) {
            Ok(()) => {
                moved_count += 1;
                if is_embedding {
                    embeddings_moved += 1;
                }
                if ext == "srt" {
                    final_srt = Some(dst);
                }
            }
            Err(e) => failures.push((src.clone(), e)),
        }
    }

    // Second pass: the `--diarize` per-speaker outputs are written to the
    // SOURCE AUDIO's directory rather than the tempdir, so the loop above
    // never sees them. Collect them now and relocate them under `embeddings/`
    // so a batch run doesn't pollute the user's audio folder.
    let media_filename = ctx
        .media
        .path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let diarize_srcs = if media_filename.is_empty() {
        Vec::new()
    } else {
        naming::find_diarize_embeddings(&ctx.media.dir, media_filename)
    };
    for src in &diarize_srcs {
        if !embeddings_dir_created {
            match std::fs::create_dir_all(&embeddings_dir) {
                Ok(()) => embeddings_dir_created = true,
                Err(e) => {
                    let _ = event_tx.send(PipelineEvent::Warning(format!(
                        "Failed to create embeddings dir {}: {e}; \
                         leaving diarize speaker file(s) in {}",
                        embeddings_dir.display(),
                        ctx.media.dir.display()
                    )));
                    break;
                }
            }
        }
        let name = src
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let dst = embeddings_dir.join(name);
        // Same-file (dest == src) can happen if the user has previously
        // pointed an output_dir at media.dir; skip to avoid a self-rename
        // EBUSY on Windows.
        if src == &dst {
            continue;
        }
        match naming::rename_with_retry(src, &dst) {
            Ok(()) => {
                moved_count += 1;
                embeddings_moved += 1;
            }
            Err(e) => {
                let _ = event_tx.send(PipelineEvent::Warning(format!(
                    "Failed to move diarize file {} -> {}: {e}",
                    src.display(),
                    dst.display()
                )));
            }
        }
    }

    let _ = event_tx.send(PipelineEvent::StderrLine(format!(
        "Moved {moved_count}/{} canonical output(s) from temp -> {}",
        outputs.len(),
        ctx.srt_dir.display()
    )));
    if embeddings_moved > 0 {
        let _ = event_tx.send(PipelineEvent::StderrLine(format!(
            "  of which {embeddings_moved} embedding file(s) -> {}",
            embeddings_dir.display()
        )));
    }

    for (src, e) in &failures {
        let _ = event_tx.send(PipelineEvent::Warning(format!(
            "Failed to move {} after retries: {e}",
            src.display()
        )));
    }

    // Sweep the temp dir ONLY after the SRT is confirmed delivered. Dropping
    // unconditionally used to destroy a finished transcription whenever every
    // move failed (e.g. cross-drive moves against a full output volume) --
    // instead, preserve the whole run dir exactly like the no-usable-output
    // path does, so nothing hours-long is ever lost to an I/O failure.
    let final_srt = match final_srt {
        Some(p) => {
            drop(tmp_dir);
            p
        }
        None => {
            let kept = tmp_dir.keep();
            let detail = match salvage::quarantine(&kept, &ctx.srt_dir, &ctx.media.stem) {
                Ok(dest) => {
                    let _ = event_tx.send(PipelineEvent::Error(format!(
                        "SRT was produced but could not be moved to the final dir. \
                         Raw output preserved at: {}",
                        dest.display()
                    )));
                    format!("raw output preserved at {}", dest.display())
                }
                Err(e) => {
                    let _ = event_tx.send(PipelineEvent::Error(format!(
                        "SRT was produced, the move to the final dir failed, and \
                         preserving it beside the output failed too: {e}. It is \
                         still on disk at: {}",
                        kept.display()
                    )));
                    format!("temp dir left at {}", kept.display())
                }
            };
            return Err(PipelineError::TranscriptionUnusable {
                reason: "output was produced but could not be moved to the final directory"
                    .to_string(),
                detail,
            });
        }
    };

    // Optional sanity sweep: legacy raw files left over from runs predating
    // the temp-dir era. Surface them once so the user can clean up manually.
    // This will be silent on healthy folders; only fires when the user has
    // historical clutter from before this fix.
    let legacy_raw = naming::verify_no_raw_outputs_remain(&ctx.srt_dir, &ctx.media.stem);
    if !legacy_raw.is_empty() {
        let _ = event_tx.send(PipelineEvent::Warning(format!(
            "Found {} legacy raw output file(s) for '{}' (from a pre-temp-dir run):",
            legacy_raw.len(),
            ctx.media.stem
        )));
        for p in &legacy_raw {
            let _ = event_tx.send(PipelineEvent::Warning(format!("  -> {}", p.display())));
        }
    }

    let srt_size = std::fs::metadata(&final_srt).map(|m| m.len()).unwrap_or(0);
    let display_path = std::fs::canonicalize(&final_srt)
        .map(|p| {
            let s = p.display().to_string();
            s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
        })
        .unwrap_or_else(|_| final_srt.display().to_string());
    let _ = event_tx.send(PipelineEvent::PhaseCompleted {
        name: format!(
            "Transcription complete: {} ({} bytes, lang={})",
            display_path, srt_size, lang
        ),
        duration: std::time::Duration::ZERO,
    });

    // Recovery notes lead, so the batch summary shows that this file's
    // transcript was rebuilt or retried rather than produced cleanly.
    let mut warnings = salvage_warnings;
    if exit_code != Some(0) {
        warnings.push(format!(
            "Whisper exited with code {:?} but SRT was produced",
            exit_code
        ));
    }
    if !failures.is_empty() {
        warnings.push(format!(
            "{} sidecar file(s) failed to move out of temp dir",
            failures.len()
        ));
    }

    Ok(PipelineResult {
        srt_file: Some(final_srt),
        hardcoded_video: None,
        lang_code: lang,
        mode_tag: mode_tag.to_string(),
        warnings,
    })
}
