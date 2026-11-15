use crate::config::AppConfig;
use crate::error::ToolError;
use crate::tools::{ToolInvocation, run_tool, validate_output};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Align a subtitle file against a reference using ALASS.
pub async fn align_subtitles(
    config: &AppConfig,
    reference: &Path,
    input_srt: &Path,
    output_srt: &Path,
) -> Result<PathBuf, ToolError> {
    let alass_path = config
        .tools
        .alass
        .as_ref()
        .ok_or_else(|| ToolError::MissingOutput {
            tool: "alass".to_string(),
            description: "ALASS binary not configured".to_string(),
            path: PathBuf::from(crate::tools::locate::exe_name("alass-cli")),
        })?;

    let inv = ToolInvocation {
        program: alass_path.clone(),
        args: vec![
            OsString::from(reference),
            OsString::from(input_srt),
            OsString::from(output_srt),
        ],
        working_dir: None,
        env_overrides: Vec::new(),
        timeout: Some(Duration::from_secs(120)),
    };

    run_tool(&inv).await?;
    validate_output(output_srt, "alass", "aligned subtitle")?;
    Ok(output_srt.to_path_buf())
}
