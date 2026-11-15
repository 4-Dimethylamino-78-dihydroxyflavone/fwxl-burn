use std::path::{Path, PathBuf};

/// Known subtitle file prefixes used for canonical naming.
const KNOWN_PREFIXES: &[&str] = &[
    "FWXL-Transcribe",
    "FWXL-Translate",
    "Parakeet-Transcribe",
    "Parakeet-Diarize",
];

/// Build the canonical output SRT filename.
///
/// Format: `{stem}.{prefix}.{mode_tag}.{model}_{lang}.srt`
///
/// Example: `MyVideo.FWXL-Transcribe.W.large-v2_en.srt`
pub fn canonical_srt_name(
    stem: &str,
    prefix: &str,
    mode_tag: &str,
    model: &str,
    lang: &str,
) -> String {
    format!("{stem}.{prefix}.{mode_tag}.{model}_{lang}.srt")
}

/// Check if a filename already has a known prefix (to avoid double-renaming).
pub fn has_known_prefix(filename: &str) -> bool {
    KNOWN_PREFIXES.iter().any(|p| filename.contains(p))
}

/// Look up an existing canonical SRT for `(stem, prefix, mode_tag, model)` in
/// `dir`, any language. Non-empty files only.
///
/// This is the "transcription index" lookup: the canonical name format
/// (`{stem}.{prefix}.{TAG}.{model}_{lang}.srt`) already encodes everything a
/// batch run needs to decide "was this file already done with these exact
/// settings?", so the on-disk names ARE the index -- no sidecar database that
/// can drift out of sync with reality.
pub fn find_canonical_srt(
    dir: &Path,
    stem: &str,
    prefix: &str,
    mode_tag: &str,
    model: &str,
) -> Option<PathBuf> {
    let wanted_prefix = format!("{stem}.{prefix}.{mode_tag}.{model}_");
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("srt"))
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&wanted_prefix))
                && std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false)
        })
}

/// Find SRT files in a directory matching a stem prefix.
pub fn find_srt_files(dir: &Path, stem: &str) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("srt")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with(stem))
        })
        .collect()
}

/// Find the most recent Whisper JSON output file in a directory.
/// Whisper outputs JSON files named like `{model}.json` or `{stem}.json`.
pub fn find_whisper_json(dir: &Path, stem: &str) -> Option<PathBuf> {
    let mut json_files: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("json")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with(stem) || !s.contains('.'))
        })
        .collect();

    // Sort by modification time, newest first
    json_files.sort_by(|a, b| {
        let a_time = std::fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let b_time = std::fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        b_time.cmp(&a_time)
    });

    json_files.into_iter().next()
}

/// Rename a raw Whisper output SRT to the canonical naming convention.
/// Returns the new path.
pub fn rename_to_canonical(
    srt_path: &Path,
    stem: &str,
    prefix: &str,
    mode_tag: &str,
    model: &str,
    lang: &str,
) -> std::io::Result<PathBuf> {
    let new_name = canonical_srt_name(stem, prefix, mode_tag, model, lang);
    let new_path = srt_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&new_name);

    if srt_path != new_path {
        rename_with_retry(srt_path, &new_path)?;
    }

    Ok(new_path)
}

/// Rename `from` → `to`, retrying on transient Windows sharing violations.
///
/// Faster-whisper-xxl is a Python+native stack: even after the top-level
/// process exits, subprocesses (ffmpeg, cuDNN workers) may hold handles to
/// the output files for a moment longer. `std::fs::rename` on Windows turns
/// that into `ErrorKind::PermissionDenied` (ERROR_SHARING_VIOLATION / 32).
/// We retry with backoff so transient failures don't cascade into silent
/// data loss for a whole queue.
pub fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    const MAX_ATTEMPTS: u32 = 5;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(200);

    let mut last_err: Option<std::io::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let kind = e.kind();
                // ERROR_NOT_SAME_DEVICE (os error 17): the source lives on a
                // different volume -- which happens whenever the run's temp
                // dir fell back to %TEMP% on C: because the output drive was
                // full/unwritable at creation time. A rename is a same-volume
                // metadata op and can NEVER succeed here, so deliver by
                // copy + best-effort delete instead of failing the run.
                if e.raw_os_error() == Some(17)
                    || kind == std::io::ErrorKind::CrossesDevices
                {
                    std::fs::copy(from, to)?;
                    // Source cleanup is best-effort: it normally sits in a
                    // TempDir that sweeps itself, and a locked source must
                    // not fail an already-delivered copy.
                    let _ = std::fs::remove_file(from);
                    return Ok(());
                }
                // Only retry on transient locking errors. Permanent errors
                // (NotFound, InvalidInput) fail fast.
                let transient = matches!(
                    kind,
                    std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Interrupted
                ) || e.raw_os_error() == Some(32) // ERROR_SHARING_VIOLATION
                    || e.raw_os_error() == Some(33); // ERROR_LOCK_VIOLATION
                if !transient || attempt == MAX_ATTEMPTS {
                    return Err(e);
                }
                last_err = Some(e);
                std::thread::sleep(BACKOFF * attempt);
            }
        }
    }
    // Unreachable barring a loop-counter bug; satisfy the compiler.
    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "rename_with_retry exhausted")
    }))
}

/// Extensions that faster-whisper-xxl emits alongside the .srt when
/// `--output_format all` is used. These need the same canonical rename as the
/// .srt so the user's folder stays organized.
pub const WHISPER_SIDECAR_EXTS: &[&str] = &["json", "lrc", "text", "tsv", "txt", "vtt"];

/// Extensions for raw numpy-format speaker/voice embedding files. faster-
/// whisper-xxl emits these when `--return_embeddings` is set in some builds.
/// Current builds instead write per-speaker `.txt` files (see
/// `is_embedding_file`); these extensions are kept for forward/backward
/// compatibility with the numpy output path.
pub const WHISPER_EMBEDDING_EXTS: &[&str] = &["npy", "npz", "pkl"];

/// True iff `file_name` looks like a speaker/voice embedding produced by
/// faster-whisper-xxl.
///
/// Q-mode runs with `--diarize <model>` write per-speaker outputs alongside
/// the source audio, named like:
///
/// ```text
///   <media_filename_with_ext>_diarize_<diarize_model>_SPEAKER_<NN>.txt
/// ```
///
/// We match by substring rather than rely on extension because (a) the
/// extension is `.txt`, which collides with the regular whisper transcript
/// sidecar, and (b) the per-speaker outputs are uniquely identified by the
/// `_diarize_` / `_SPEAKER_` markers anywhere in the filename. Numpy-format
/// outputs (`.npy` / `.npz` / `.pkl`) are also recognised in case a future
/// build routes embeddings that way instead.
pub fn is_embedding_file(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    if lower.contains("_diarize_") || lower.contains("_speaker_") {
        return true;
    }
    if let Some(dot) = lower.rfind('.') {
        let ext = &lower[dot + 1..];
        return WHISPER_EMBEDDING_EXTS
            .iter()
            .any(|e| e.eq_ignore_ascii_case(ext));
    }
    false
}

/// Find diarization speaker-output files for the audio file `media_filename`
/// (including its extension) in `media_dir`.
///
/// faster-whisper-xxl's `--diarize` flag bypasses `--output_dir`: the
/// per-speaker `.txt` outputs land next to the source audio, not in the run's
/// tempdir. The whisper pipeline scans `media_dir` for them after the main
/// finalize step so they can be relocated into `embeddings/`.
///
/// Matching rules:
///   1. The filename starts with `media_filename` (audio name *with* its
///      original extension, because that's the prefix faster-whisper-xxl uses).
///   2. The trailing portion identifies it as an embedding via
///      `is_embedding_file`.
pub fn find_diarize_embeddings(media_dir: &Path, media_filename: &str) -> Vec<PathBuf> {
    if media_filename.is_empty() {
        return Vec::new();
    }
    let prefix = media_filename.to_ascii_lowercase();
    std::fs::read_dir(media_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            if !p.is_file() {
                return false;
            }
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                return false;
            };
            let lname = name.to_ascii_lowercase();
            if !lname.starts_with(&prefix) || lname.len() == prefix.len() {
                return false;
            }
            is_embedding_file(name)
        })
        .collect()
}

/// Pick the directory that should receive embedding files for a run whose
/// SRT output goes into `srt_dir`.
///
/// When `srt_dir` is itself named `source` (the default layout when
/// `use_source_folder = true`), the embeddings dir is placed as a *sibling*
/// of `source/`, so the on-disk layout becomes:
///
/// ```text
///   media_dir/
///     source/      <- SRTs + sidecars live here
///     embeddings/  <- diarize speaker files / numpy embeddings live here
/// ```
///
/// In all other layouts (explicit `output_dir`, or `use_source_folder = false`)
/// there is no `source/` to be next to, so the embeddings dir is created as a
/// subdirectory of `srt_dir`.
pub fn resolve_embeddings_dir(srt_dir: &Path) -> PathBuf {
    let is_source = srt_dir
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|n| n.eq_ignore_ascii_case("source"));
    if is_source {
        if let Some(parent) = srt_dir.parent() {
            return parent.join("embeddings");
        }
    }
    srt_dir.join("embeddings")
}

/// Strict raw-output discovery: find files that whisper-xxl JUST produced for
/// this transcription run. Unlike `find_srt_files`, this uses an exact stem
/// match (no `starts_with`) and excludes anything with a known canonical
/// prefix. Without this, a previously-renamed file from an earlier run
/// (e.g. `foo.FWXL-Transcribe.U.large-v2_en.srt`) would be "found" alongside
/// the new `foo.srt` and non-deterministic filesystem order would sometimes
/// cause the finalizer to pick the already-renamed one and skip rename of the
/// raw output -- the root cause of the batch-rename bug we hit.
///
/// Only returns .srt and WHISPER_SIDECAR_EXTS files.
pub fn find_raw_whisper_outputs(dir: &Path, stem: &str) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let Some(ext) = p.extension().and_then(|s| s.to_str()) else {
                return false;
            };
            let ext_lc = ext.to_ascii_lowercase();
            let is_whisper_ext = ext_lc == "srt"
                || WHISPER_SIDECAR_EXTS
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(&ext_lc));
            if !is_whisper_ext {
                return false;
            }
            let Some(fs) = p.file_stem().and_then(|s| s.to_str()) else {
                return false;
            };
            // STRICT: file_stem must equal stem exactly. `foo.srt` -> stem is
            // `foo`. A previously-renamed `foo.FWXL-Transcribe.U.v2_en.srt`
            // has stem `foo.FWXL-Transcribe.U.v2_en` and is correctly rejected.
            // Belt-and-suspenders: also reject any stem containing a known
            // canonical prefix, even if somehow `fs == stem`.
            fs == stem && !has_known_prefix(fs)
        })
        .collect()
}

/// Post-rename sanity check: after finalize, there should be ZERO raw-stem
/// outputs left. Any that remain indicate a rename failure the user needs to
/// know about (permissions, disk full, transient lock we couldn't retry
/// through). Returns the offending paths.
pub fn verify_no_raw_outputs_remain(dir: &Path, stem: &str) -> Vec<PathBuf> {
    find_raw_whisper_outputs(dir, stem)
}

/// Rename all sidecar files (json/lrc/text/tsv/txt/vtt) whose stem matches
/// `source_stem` in `dir` to share the canonical rename applied to the SRT.
/// The new filename is `{stem}.{prefix}.{mode_tag}.{model}_{lang}.{ext}`.
/// Files that already have a known prefix are left untouched.
/// Returns the list of renamed pairs (old -> new) that actually moved.
pub fn rename_whisper_sidecars(
    dir: &Path,
    source_stem: &str,
    stem: &str,
    prefix: &str,
    mode_tag: &str,
    model: &str,
    lang: &str,
) -> Vec<(PathBuf, PathBuf)> {
    let mut moved = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return moved,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !WHISPER_SIDECAR_EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
            continue;
        }
        let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Only rename files whose stem exactly matches the source media stem.
        // Skip files that already carry a known prefix (avoid double-rename).
        if file_stem != source_stem || has_known_prefix(file_stem) {
            continue;
        }
        let new_name = format!("{stem}.{prefix}.{mode_tag}.{model}_{lang}.{ext}");
        let new_path = path.with_file_name(&new_name);
        if path == new_path {
            continue;
        }
        // Don't clobber an existing file of the same name.
        if new_path.exists() {
            continue;
        }
        if rename_with_retry(&path, &new_path).is_ok() {
            moved.push((path, new_path));
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_name() {
        let name = canonical_srt_name("MyVideo", "FWXL-Transcribe", "W", "large-v2", "en");
        assert_eq!(name, "MyVideo.FWXL-Transcribe.W.large-v2_en.srt");
    }

    #[test]
    fn test_has_known_prefix() {
        assert!(has_known_prefix("MyVideo.FWXL-Transcribe.W.large-v2_en.srt"));
        assert!(has_known_prefix("foo.Parakeet-Transcribe.P.v2-int8_en.srt"));
        assert!(!has_known_prefix("MyVideo.srt"));
        assert!(!has_known_prefix("large-v2.srt"));
    }

    /// Regression test for the batch-rename cascade bug. When a dir contains:
    ///   - foo.srt                                          (just produced, RAW)
    ///   - foo.FWXL-Transcribe.U.large-v2_en.srt            (from earlier run)
    ///   - foo.FWXL-Transcribe.U.large-v2_en.json           (from earlier run)
    /// strict discovery must return ONLY the raw foo.srt, so finalize renames
    /// the new run's output without clobbering the earlier run's canonical files.
    #[test]
    fn test_find_raw_whisper_outputs_excludes_canonical_siblings() {
        let tmp = std::env::temp_dir().join(format!("fwxl-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Raw new output
        std::fs::write(tmp.join("foo.srt"), "raw").unwrap();
        std::fs::write(tmp.join("foo.json"), "{}").unwrap();
        // Canonical from earlier run (must be preserved & excluded)
        std::fs::write(tmp.join("foo.FWXL-Transcribe.U.large-v2_en.srt"), "old").unwrap();
        std::fs::write(tmp.join("foo.FWXL-Transcribe.U.large-v2_en.json"), "{}").unwrap();
        // Unrelated file
        std::fs::write(tmp.join("bar.srt"), "other").unwrap();

        let raw = find_raw_whisper_outputs(&tmp, "foo");
        let names: Vec<String> = raw
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(raw.len(), 2, "got {names:?}");
        assert!(names.iter().any(|n| n == "foo.srt"));
        assert!(names.iter().any(|n| n == "foo.json"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_find_raw_excludes_starts_with_siblings() {
        // `fooBar.srt` must NOT match a search for stem "foo" (the old
        // starts_with matcher would have incorrectly included it).
        let tmp = std::env::temp_dir().join(format!("fwxl-test-sw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("foo.srt"), "").unwrap();
        std::fs::write(tmp.join("fooBar.srt"), "").unwrap();
        std::fs::write(tmp.join("foo_extra.srt"), "").unwrap();

        let raw = find_raw_whisper_outputs(&tmp, "foo");
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].file_name().unwrap(), "foo.srt");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_rename_with_retry_happy_path() {
        let tmp = std::env::temp_dir().join(format!("fwxl-test-rn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("a.srt");
        let dst = tmp.join("b.srt");
        std::fs::write(&src, "x").unwrap();
        rename_with_retry(&src, &dst).unwrap();
        assert!(!src.exists());
        assert!(dst.exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_is_embedding_file_diarize_speaker_txt() {
        // Real-world output from `--diarize pyannote_v3.1 --return_embeddings`:
        // <audio>.m4a_diarize_pyannote_v3.1_SPEAKER_<NN>.txt
        assert!(is_embedding_file(
            "phone_20260414.m4a_diarize_pyannote_v3.1_SPEAKER_00.txt"
        ));
        assert!(is_embedding_file(
            "phone_20260414.m4a_diarize_pyannote_v3.1_SPEAKER_03.txt"
        ));
        // Case-insensitive
        assert!(is_embedding_file(
            "Phone.MP3_Diarize_Pyannote_v3.1_Speaker_00.TXT"
        ));
    }

    #[test]
    fn test_is_embedding_file_numpy_variants() {
        assert!(is_embedding_file("phone.m4a.npy"));
        assert!(is_embedding_file("phone.m4a.NPZ"));
        assert!(is_embedding_file("phone.m4a.pkl"));
    }

    #[test]
    fn test_is_embedding_file_rejects_subtitles() {
        // Standard whisper sidecars must NOT be misidentified as embeddings.
        assert!(!is_embedding_file("phone.srt"));
        assert!(!is_embedding_file("phone.json"));
        assert!(!is_embedding_file("phone.txt"));
        assert!(!is_embedding_file("phone.vtt"));
        assert!(!is_embedding_file("phone.lrc"));
        assert!(!is_embedding_file("phone.tsv"));
        assert!(!is_embedding_file(""));
    }

    #[test]
    fn test_find_diarize_embeddings_picks_per_speaker_txt() {
        let tmp = std::env::temp_dir().join(format!("fwxl-test-emb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Audio + standard sidecars (must be ignored)
        std::fs::write(tmp.join("phone.m4a"), "").unwrap();
        std::fs::write(tmp.join("phone.srt"), "").unwrap();
        std::fs::write(tmp.join("phone.txt"), "").unwrap();
        // Diarize speaker outputs (must be picked up)
        std::fs::write(
            tmp.join("phone.m4a_diarize_pyannote_v3.1_SPEAKER_00.txt"),
            "",
        )
        .unwrap();
        std::fs::write(
            tmp.join("phone.m4a_diarize_pyannote_v3.1_SPEAKER_01.txt"),
            "",
        )
        .unwrap();
        // Different audio file's diarize output (must NOT be picked up)
        std::fs::write(
            tmp.join("other.m4a_diarize_pyannote_v3.1_SPEAKER_00.txt"),
            "",
        )
        .unwrap();

        let found = find_diarize_embeddings(&tmp, "phone.m4a");
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(found.len(), 2, "got {names:?}");
        assert!(names
            .iter()
            .any(|n| n == "phone.m4a_diarize_pyannote_v3.1_SPEAKER_00.txt"));
        assert!(names
            .iter()
            .any(|n| n == "phone.m4a_diarize_pyannote_v3.1_SPEAKER_01.txt"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_find_diarize_embeddings_does_not_match_prefix_only() {
        // Audio `phone.m4a` must NOT have `phone2.m4a_diarize_*` files attributed
        // to it. Without the extension-in-prefix check, a substring-based match
        // could cross-attribute speakers between similarly-named files.
        let tmp = std::env::temp_dir().join(format!("fwxl-test-pfx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("phone2.m4a_diarize_pyannote_v3.1_SPEAKER_00.txt"),
            "",
        )
        .unwrap();
        let found = find_diarize_embeddings(&tmp, "phone.m4a");
        assert_eq!(found.len(), 0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_resolve_embeddings_dir_sibling_of_source() {
        // The use_source_folder=true case: srt_dir is `<media>/source`,
        // embeddings go in the sibling `<media>/embeddings`.
        let srt_dir = PathBuf::from("D:/media/source");
        let emb = resolve_embeddings_dir(&srt_dir);
        assert_eq!(emb, PathBuf::from("D:/media/embeddings"));
    }

    #[test]
    fn test_resolve_embeddings_dir_sibling_of_source_case_insensitive() {
        let srt_dir = PathBuf::from("D:/media/Source");
        let emb = resolve_embeddings_dir(&srt_dir);
        assert_eq!(emb, PathBuf::from("D:/media/embeddings"));
    }

    #[test]
    fn test_resolve_embeddings_dir_subdir_when_not_source() {
        // No `source` segment: embeddings is just a subdir of srt_dir.
        let srt_dir = PathBuf::from("D:/output");
        let emb = resolve_embeddings_dir(&srt_dir);
        assert_eq!(emb, PathBuf::from("D:/output/embeddings"));
    }

    #[test]
    fn test_resolve_embeddings_dir_subdir_when_media_dir() {
        // use_source_folder=false, no output_dir -> srt_dir == media.dir.
        // Embeddings becomes a subdir of the media dir.
        let srt_dir = PathBuf::from("D:/Podcasts");
        let emb = resolve_embeddings_dir(&srt_dir);
        assert_eq!(emb, PathBuf::from("D:/Podcasts/embeddings"));
    }

    #[test]
    fn test_rename_with_retry_permanent_error_fails_fast() {
        // Non-existent source -> NotFound -> permanent -> no retries.
        let src = std::env::temp_dir().join("does-not-exist-fwxl-xyzzy.srt");
        let dst = std::env::temp_dir().join("neither-fwxl-xyzzy.srt");
        let t0 = std::time::Instant::now();
        let err = rename_with_retry(&src, &dst).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        // Fail-fast: should return well under the 5*200ms backoff budget.
        assert!(t0.elapsed() < std::time::Duration::from_millis(300));
    }
}
