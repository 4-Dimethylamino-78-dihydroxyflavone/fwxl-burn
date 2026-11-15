//! Cross-platform discovery of the external tool binaries fwxl-burn drives
//! (ffmpeg, ffprobe, faster-whisper-xxl).
//!
//! Two rules keep this predictable:
//!
//! 1. A path the user **pinned** in the config is authoritative. If it does not
//!    exist the run fails naming that path -- it is never quietly swapped for
//!    some other copy found elsewhere, because a silent substitute is exactly
//!    the kind of fallback that hides a broken setup.
//! 2. A tool the config leaves **unset** is discovered by walking
//!    [`search_dirs`] in order and taking the first hit.
//!
//! Nothing here hardcodes `.exe`: names go through [`exe_name`], which appends
//! the platform's executable suffix (empty on Linux/macOS).

use std::env::consts::EXE_SUFFIX;
use std::path::{Path, PathBuf};

/// Folder name of an unpacked Faster-Whisper-XXL distribution. Its Windows
/// build ships ffmpeg/ffprobe alongside the transcriber, so a portable layout
/// with this folder next to fwxl-burn needs no configuration at all.
pub const XXL_BUNDLE_DIR: &str = "Faster-Whisper-XXL";

/// Platform file name for a tool stem: `ffmpeg` -> `ffmpeg.exe` on Windows,
/// unchanged elsewhere.
pub fn exe_name(stem: &str) -> String {
    format!("{stem}{EXE_SUFFIX}")
}

/// Everything discovery is allowed to look at, captured once so the search
/// itself is a pure function of this struct (and therefore unit-testable
/// without touching the real environment).
#[derive(Debug, Clone, Default)]
pub struct SearchContext {
    /// Directory holding the loaded config file, if one was loaded.
    pub config_dir: Option<PathBuf>,
    /// Directory holding the running fwxl-burn executable.
    pub exe_dir: Option<PathBuf>,
    /// Entries of the `PATH` environment variable, in order.
    pub path_dirs: Vec<PathBuf>,
}

impl SearchContext {
    /// Capture the real environment. `config_path` is the config *file*.
    pub fn from_env(config_path: Option<&Path>) -> Self {
        let config_dir = config_path
            .and_then(|p| std::fs::canonicalize(p).ok())
            .and_then(|p| p.parent().map(Path::to_path_buf));
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        let path_dirs = std::env::var_os("PATH")
            .map(|v| std::env::split_paths(&v).collect())
            .unwrap_or_default();
        Self {
            config_dir,
            exe_dir,
            path_dirs,
        }
    }
}

/// Directories searched, in priority order, for a tool the config does not
/// pin. `extra` carries tool-specific well-known install locations (e.g. the
/// folder Subtitle Edit downloads Faster-Whisper-XXL into).
///
/// The current working directory is deliberately NOT searched: a media folder
/// the user happens to `cd` into must never be able to supply the `ffmpeg`
/// that gets executed.
///
/// This ordering is the one real policy decision in the module: bundled
/// copies (beside the executable or the config) beat system-wide ones on
/// PATH. That favours reproducible portable folders -- the ffmpeg shipped
/// with a bundle is the one it was tested with -- at the cost of a stale
/// bundled copy shadowing a newer system install.
pub fn search_dirs(ctx: &SearchContext, extra: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for base in [&ctx.exe_dir, &ctx.config_dir].into_iter().flatten() {
        dirs.push(base.clone());
        dirs.push(base.join(XXL_BUNDLE_DIR));
    }
    dirs.extend(extra.iter().cloned());
    dirs.extend(ctx.path_dirs.iter().cloned());
    dirs
}

/// First existing `<dir>/<stem><exe suffix>` across [`search_dirs`]. Stems are
/// tried in the order given within each directory.
pub fn discover(stems: &[&str], ctx: &SearchContext, extra: &[PathBuf]) -> Option<PathBuf> {
    search_dirs(ctx, extra).into_iter().find_map(|dir| {
        stems
            .iter()
            .map(|stem| dir.join(exe_name(stem)))
            .find(|cand| cand.is_file())
    })
}

/// Resolve a path written in the config.
///
/// Absolute paths are taken as-is. Relative ones are tried against the
/// working directory first (the historical behaviour, so existing configs keep
/// working), then the config file's directory, then the executable's
/// directory. Returns `None` when no candidate exists.
pub fn resolve_configured(path: &Path, ctx: &SearchContext) -> Option<PathBuf> {
    if path.is_absolute() {
        return path.exists().then(|| path.to_path_buf());
    }
    if path.exists() {
        return Some(path.to_path_buf());
    }
    [&ctx.config_dir, &ctx.exe_dir]
        .into_iter()
        .flatten()
        .map(|base| base.join(path))
        .find(|cand| cand.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn exe_name_appends_platform_suffix() {
        let name = exe_name("ffmpeg");
        if cfg!(windows) {
            assert_eq!(name, "ffmpeg.exe");
        } else {
            assert_eq!(name, "ffmpeg");
        }
    }

    #[test]
    fn discover_finds_tool_on_path() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        touch(&bin.join(exe_name("ffprobe")));
        let ctx = SearchContext {
            path_dirs: vec![tmp.path().join("empty"), bin.clone()],
            ..Default::default()
        };
        assert_eq!(
            discover(&["ffprobe"], &ctx, &[]),
            Some(bin.join(exe_name("ffprobe")))
        );
        assert_eq!(discover(&["ffmpeg"], &ctx, &[]), None);
    }

    #[test]
    fn discover_sees_the_xxl_bundle_beside_the_executable() {
        let tmp = tempfile::tempdir().unwrap();
        let bundled = tmp
            .path()
            .join(XXL_BUNDLE_DIR)
            .join(exe_name("faster-whisper-xxl"));
        touch(&bundled);
        let ctx = SearchContext {
            exe_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        assert_eq!(discover(&["faster-whisper-xxl"], &ctx, &[]), Some(bundled));
    }

    #[test]
    fn discover_tries_stems_in_order_within_a_directory() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join(exe_name("alass")));
        touch(&tmp.path().join(exe_name("alass-cli")));
        let ctx = SearchContext {
            path_dirs: vec![tmp.path().to_path_buf()],
            ..Default::default()
        };
        assert_eq!(
            discover(&["alass-cli", "alass"], &ctx, &[]),
            Some(tmp.path().join(exe_name("alass-cli")))
        );
    }

    #[test]
    fn relative_config_path_falls_back_to_the_config_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let rel = Path::new("fwxl-locate-test-models").join("vad.onnx");
        touch(&tmp.path().join(&rel));
        let ctx = SearchContext {
            config_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        assert_eq!(resolve_configured(&rel, &ctx), Some(tmp.path().join(&rel)));
        assert_eq!(resolve_configured(Path::new("fwxl-no-such-file"), &ctx), None);
    }

    #[test]
    fn absolute_config_path_is_never_rebased() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("not-here.bin");
        let ctx = SearchContext {
            config_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        assert_eq!(resolve_configured(&missing, &ctx), None);
    }
}
