use crate::config::AppConfig;
use crate::error::ToolError;
use crate::tools::{ToolInvocation, run_tool, validate_output};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Convert a subtitle file to SRT format using SubtitleEdit.
pub async fn convert_to_srt(
    config: &AppConfig,
    input: &Path,
    output: &Path,
) -> Result<PathBuf, ToolError> {
    let se_path = config
        .tools
        .subtitle_edit
        .as_ref()
        .ok_or_else(|| ToolError::MissingOutput {
            tool: "SubtitleEdit".to_string(),
            description: "SubtitleEdit binary not configured".to_string(),
            path: PathBuf::from(crate::tools::locate::exe_name("SubtitleEdit")),
        })?;

    let inv = ToolInvocation {
        program: se_path.clone(),
        args: vec![
            OsString::from("/convert"),
            OsString::from(input),
            OsString::from("subrip"),
            OsString::from("/outputfilename:"),
            OsString::from(output),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(60)),
    };

    run_tool(&inv).await?;
    validate_output(output, "SubtitleEdit", "converted SRT")?;
    Ok(output.to_path_buf())
}
