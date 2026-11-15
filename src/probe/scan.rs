use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Scan a directory for existing SRT files matching a media file's stem.
pub fn find_existing_srts(dir: &Path, stem: &str) -> Vec<PathBuf> {
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

/// One SRT discovered for a media stem, paired with the index of the search
/// directory it came from (so the UI can label it `[SRCDIR]` vs `[SRT_DIR]`).
#[derive(Debug, Clone)]
pub struct SrtCandidate {
    pub path: PathBuf,
    pub origin_dir_index: usize,
}

/// Scan multiple directories for SRTs matching `stem`, deduping by canonical
/// path. The first directory in `dirs` "wins" for any duplicate, so callers
/// should pass the more authoritative location first (e.g. SRCDIR before
/// SRT_DIR when they happen to point at the same place).
///
/// This is the multi-location version of `find_existing_srts` -- the SRT
/// hardburn picker scans `media.dir` (SRCDIR, "beside the file") AND the
/// resolved output dir (SRT_DIR, typically `media.dir/source`) so the user
/// can hardcode any subtitle the project has produced or dropped in.
pub fn find_srts_in_dirs(dirs: &[PathBuf], stem: &str) -> Vec<SrtCandidate> {
    let mut out: Vec<SrtCandidate> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for (i, dir) in dirs.iter().enumerate() {
        for srt in find_existing_srts(dir, stem) {
            // Canonicalize to dedupe across dirs that resolve to the same
            // physical path (e.g. media.dir == srt_dir when use_source_folder
            // is false). Fall back to the raw path if canonicalize fails so
            // a transient I/O error doesn't drop legitimate matches.
            let key = std::fs::canonicalize(&srt).unwrap_or_else(|_| srt.clone());
            if seen.insert(key) {
                out.push(SrtCandidate {
                    path: srt,
                    origin_dir_index: i,
                });
            }
        }
    }
    out
}

/// Find an "English master" SRT for translation (the base SRT to translate from).
pub fn find_english_master(dir: &Path, stem: &str) -> Option<PathBuf> {
    let srts = find_existing_srts(dir, stem);

    // Prefer FWXL-Transcribe tagged files, then any SRT
    srts.iter()
        .find(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.contains("FWXL-Transcribe") || f.contains("Parakeet-Transcribe"))
        })
        .or_else(|| srts.first())
        .cloned()
}
