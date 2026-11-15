//! Persistent record of the most recent multi-item run.
//!
//! Saved to `%APPDATA%\fwxl-burn\last_run.json` when a queue starts and
//! rewritten after every item completes, fails, or is skipped -- so a crash,
//! an OOM cascade, or a plain Ctrl+C mid-queue leaves an on-disk record of
//! exactly which files were done and which weren't. The Start screen's
//! `[R] Redo last run` reloads it into a fresh paste queue (failed items
//! first), which is what makes "run the same batch again on a different
//! setting or model" a two-keystroke operation instead of re-pasting paths.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Pending,
    Done,
    Failed,
    /// Already had a canonical SRT for this mode+model; not re-transcribed.
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestItem {
    pub path: PathBuf,
    pub status: ItemStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub srt: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub created: String,
    pub updated: String,
    /// TranscriptionMode key char ("Q", "I", ...). Empty until the first
    /// pipeline launch fills it in (the queue is committed before settings
    /// are picked).
    #[serde(default)]
    pub mode_key: String,
    #[serde(default)]
    pub model: String,
    pub items: Vec<ManifestItem>,
}

impl RunManifest {
    pub fn new(files: &[PathBuf]) -> Self {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        Self {
            created: now.clone(),
            updated: now,
            mode_key: String::new(),
            model: String::new(),
            items: files
                .iter()
                .map(|p| ManifestItem {
                    path: p.clone(),
                    status: ItemStatus::Pending,
                    srt: None,
                    error: None,
                })
                .collect(),
        }
    }

    pub fn failed_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == ItemStatus::Failed)
            .count()
    }

    pub fn pending_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == ItemStatus::Pending)
            .count()
    }

    /// Paths to re-run: everything unfinished (Failed AND Pending) when any
    /// exist, otherwise every item. A cancelled or crashed batch leaves its
    /// tail as Pending -- redo must pick that tail up, not just the explicit
    /// failures. Non-existent paths (moved/deleted since the run) are dropped.
    pub fn redo_paths(&self) -> Vec<PathBuf> {
        let unfinished: Vec<PathBuf> = self
            .items
            .iter()
            .filter(|i| matches!(i.status, ItemStatus::Failed | ItemStatus::Pending))
            .map(|i| i.path.clone())
            .filter(|p| p.exists())
            .collect();
        if !unfinished.is_empty() {
            return unfinished;
        }
        self.items
            .iter()
            .map(|i| i.path.clone())
            .filter(|p| p.exists())
            .collect()
    }

    /// Persist to `last_run.json`. Failures are non-fatal by design -- the
    /// manifest is a convenience, never worth aborting a transcription over.
    pub fn save(&self) {
        let Some(path) = manifest_path() else { return };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
    }
}

/// `%APPDATA%\fwxl-burn\last_run.json` (same root the config search uses).
pub fn manifest_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("fwxl-burn").join("last_run.json"))
}

/// Load the saved manifest, if any.
pub fn load() -> Option<RunManifest> {
    let path = manifest_path()?;
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Update the status of the item matching `media_path` (path compared
/// component-wise after canonicalization when possible). Returns true when a
/// row was actually updated.
pub fn set_status(
    manifest: &mut RunManifest,
    media_path: &Path,
    status: ItemStatus,
    srt: Option<PathBuf>,
    error: Option<String>,
) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let target = canon(media_path);
    let Some(item) = manifest
        .items
        .iter_mut()
        .find(|i| canon(&i.path) == target)
    else {
        return false;
    };
    item.status = status;
    if srt.is_some() {
        item.srt = srt;
    }
    item.error = error;
    manifest.updated = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redo_prefers_failed_items() {
        let dir = std::env::temp_dir().join(format!("fwxl-manifest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.m4a");
        let b = dir.join("b.m4a");
        std::fs::write(&a, "").unwrap();
        std::fs::write(&b, "").unwrap();

        let mut m = RunManifest::new(&[a.clone(), b.clone()]);
        assert_eq!(m.redo_paths().len(), 2, "no failures -> everything");

        set_status(&mut m, &b, ItemStatus::Failed, None, Some("oom".into()));
        set_status(&mut m, &a, ItemStatus::Done, None, None);
        assert_eq!(m.failed_count(), 1);
        let redo = m.redo_paths();
        assert_eq!(redo.len(), 1);
        assert_eq!(redo[0], b);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn redo_includes_pending_after_interrupt() {
        // A cancelled/crashed batch leaves its tail Pending -- redo must
        // pick up BOTH the failures and the never-ran tail. (Real case: a
        // 1584-item run died at item 70 with 15 failed / 1513 pending; the
        // failed-only version would have silently dropped the 1513.)
        let dir = std::env::temp_dir().join(format!("fwxl-manifest3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.m4a");
        let b = dir.join("b.m4a");
        let c = dir.join("c.m4a");
        for p in [&a, &b, &c] {
            std::fs::write(p, "").unwrap();
        }

        let mut m = RunManifest::new(&[a.clone(), b.clone(), c.clone()]);
        set_status(&mut m, &a, ItemStatus::Done, None, None);
        set_status(&mut m, &b, ItemStatus::Failed, None, Some("disk full".into()));
        // c stays Pending.
        let redo = m.redo_paths();
        assert_eq!(redo.len(), 2, "failed + pending, got {redo:?}");
        assert!(redo.contains(&b));
        assert!(redo.contains(&c));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_status_matches_uncanonicalized_paths() {
        let dir = std::env::temp_dir().join(format!("fwxl-manifest2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.m4a");
        std::fs::write(&a, "").unwrap();

        // Manifest holds the raw path; the pipeline reports the canonical one.
        let mut m = RunManifest::new(&[a.clone()]);
        let canonical = std::fs::canonicalize(&a).unwrap();
        assert!(set_status(&mut m, &canonical, ItemStatus::Done, None, None));
        assert_eq!(m.items[0].status, ItemStatus::Done);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
