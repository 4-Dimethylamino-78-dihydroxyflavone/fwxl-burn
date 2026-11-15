mod asr;
mod chunk;
mod config;
mod error;
mod gpu;
mod media;
mod naming;
mod pipeline;
mod probe;
mod run_manifest;
mod srt;
mod tools;
mod tui;

use clap::Parser;
use std::path::{Path, PathBuf};

/// Render a path as an absolute, user-friendly string.
/// Canonicalizes when possible and strips the Windows `\\?\` long-path prefix.
fn display_abs(p: &Path) -> String {
    let s = std::fs::canonicalize(p)
        .map(|c| c.display().to_string())
        .unwrap_or_else(|_| p.display().to_string());
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

#[derive(Parser, Debug)]
#[command(name = "fwxl-burn", version, about)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Generate a default config file and exit.
    #[arg(long)]
    init: bool,

    /// Run in headless mode (no TUI, for scripting/CI).
    #[arg(long)]
    headless: bool,

    /// Input file or directory (headless mode).
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Transcription mode for headless mode (W, E, Q, I, U, Y, P, D, O, X).
    #[arg(short, long)]
    mode: Option<String>,

    /// Whisper model name for headless mode.
    #[arg(long)]
    model: Option<String>,

    /// Media files to queue (positional arguments).
    /// Dragging files onto the executable on Windows passes them this way.
    #[arg(trailing_var_arg = true)]
    files: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.init {
        return generate_default_config(&cli.config);
    }

    // A config file is optional: with none on disk the defaults compiled into
    // the binary are used, so a fresh download runs with zero setup.
    let (mut config, config_path) = match config::find_config(cli.config.as_deref()) {
        Ok(path) => {
            let cfg = config::load_config(&path)?;
            eprintln!("Loaded config from {}", path.display());
            (cfg, Some(path))
        }
        Err(config::ConfigError::NotFound { .. }) => {
            eprintln!(
                "No config file found; using built-in defaults. \
                 Run `fwxl-burn --init` to write one you can edit."
            );
            (config::builtin_config()?, None)
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Pin down every external tool once, up front: configured paths must
    // exist, unset ones are discovered beside the executable or on PATH.
    match config::resolve_tools(&mut config, config_path.as_deref()) {
        Ok(warnings) => {
            for w in &warnings {
                eprintln!("Note: {w}");
            }
        }
        Err(e) => {
            eprintln!("Setup error: {e}");
            std::process::exit(1);
        }
    }

    // Collect input files: from --input, from positional drag-drop args, or none (TUI prompts)
    let mut input_files: Vec<PathBuf> = cli.files.clone();
    if let Some(ref input) = cli.input {
        if !input_files.contains(input) {
            input_files.insert(0, input.clone());
        }
    }
    // Filter to files that actually exist
    input_files.retain(|p| p.exists());

    if cli.headless {
        run_headless(config, cli, &input_files).await?;
    } else {
        tui::run(config, input_files).await?;
    }

    Ok(())
}

async fn run_headless(config: config::AppConfig, cli: Cli, input_files: &[PathBuf]) -> anyhow::Result<()> {
    let input = input_files
        .first()
        .or(cli.input.as_ref())
        .ok_or_else(|| anyhow::anyhow!("--input is required in headless mode"))?
        .clone();

    let mode_str = cli.mode.unwrap_or_else(|| "U".to_string());
    let model = cli
        .model
        .unwrap_or_else(|| config.whisper.default_model.clone());

    // Parse mode
    let mode = match mode_str.to_uppercase().as_str() {
        "W" => pipeline::TranscriptionMode::TestWhisper,
        "E" => pipeline::TranscriptionMode::Extended,
        "Q" => pipeline::TranscriptionMode::Quality,
        "I" => pipeline::TranscriptionMode::Inclusive,
        "U" => pipeline::TranscriptionMode::Unfiltered,
        "Y" => pipeline::TranscriptionMode::Chunked,
        "P" => pipeline::TranscriptionMode::Parakeet,
        "D" => pipeline::TranscriptionMode::ParakeetDiarize,
        "O" => pipeline::TranscriptionMode::Omnilingual,
        "X" => pipeline::TranscriptionMode::Diagnostic,
        other => anyhow::bail!("Unknown mode: {other}. Valid: W, E, Q, I, U, Y, P, D, O, X"),
    };

    // Create media file
    let mut media_file = media::MediaFile::from_path(&input)?;
    eprintln!("Input: {} ({})", media_file.stem, media_file.extension);

    // Probe media. Repair is handled below, once the context exists; refusing
    // here would deny headless runs the same auto-repair the TUI gets.
    eprintln!("Probing media...");
    let probe_status = probe::probe_media_file(&config, &mut media_file).await?;
    if let Some(reason) = probe_status.reason() {
        eprintln!("Input container is unreadable: {reason}");
    }
    if let Some(dur) = media_file.duration {
        eprintln!("Duration: {:.1}s", dur.as_secs_f64());
    }
    eprintln!(
        "Embedded subtitle streams: {}",
        media_file.embedded_subs.len()
    );

    // Determine output directory
    let srt_dir = if config.use_source_folder {
        let dir = media_file.dir.join("source");
        std::fs::create_dir_all(&dir)?;
        dir
    } else if let Some(ref out) = config.output_dir {
        std::fs::create_dir_all(out)?;
        out.clone()
    } else {
        media_file.dir.clone()
    };

    let final_out_dir = config
        .output_dir
        .clone()
        .unwrap_or_else(|| media_file.dir.clone());
    std::fs::create_dir_all(&final_out_dir)?;

    let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M").to_string();

    let mut ctx = pipeline::PipelineContext {
        config: config.clone(),
        media: media_file,
        mode,
        selected_model: model,
        srt_dir,
        final_out_dir,
        timestamp,
        is_batch: false,
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    // Run pipeline
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    // Pre-flight repair, before any transcription work is scheduled.
    pipeline::repair::ensure_readable(&mut ctx, &event_tx).await?;
    let ctx = ctx;

    // Spawn event printer
    let printer = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                pipeline::PipelineEvent::PhaseStarted { name, phase_num, total_phases } => {
                    eprintln!("[{phase_num}/{total_phases}] {name}");
                }
                pipeline::PipelineEvent::Progress { percent, message } => {
                    eprintln!("  {percent:.1}% {message}");
                }
                pipeline::PipelineEvent::StderrLine(l) => {
                    eprintln!("  {l}");
                }
                pipeline::PipelineEvent::Warning(msg) => {
                    eprintln!("[WARN] {msg}");
                }
                pipeline::PipelineEvent::Error(msg) => {
                    eprintln!("[ERROR] {msg}");
                }
                pipeline::PipelineEvent::Completed(result) => {
                    if let Some(ref srt) = result.srt_file {
                        eprintln!("[OK] SRT: {}", display_abs(srt));
                    }
                    if let Some(ref vid) = result.hardcoded_video {
                        eprintln!("[OK] Video: {}", display_abs(vid));
                    }
                }
                _ => {}
            }
        }
    });

    if mode.uses_whisper() {
        match pipeline::whisper::run_whisper_pipeline(&ctx, event_tx).await {
            Ok(result) => {
                eprintln!("\nTranscription complete.");
                if let Some(ref srt) = result.srt_file {
                    eprintln!("Output: {}", display_abs(srt));
                }
                for warn in &result.warnings {
                    eprintln!("Warning: {warn}");
                }
            }
            Err(e) => eprintln!("\nTranscription failed: {e}"),
        }
    } else if mode.uses_parakeet() {
        if mode == pipeline::TranscriptionMode::ParakeetDiarize {
            eprintln!(
                "Warning: ParakeetDiarize (D) is no longer supported. \
                 Running plain Parakeet (P) instead."
            );
        }
        let parakeet_model = ctx
            .config
            .prkt
            .models
            .iter()
            .find(|m| m.name == ctx.selected_model)
            .or_else(|| ctx.config.prkt.models.iter().find(|m| !m.broken))
            .or_else(|| ctx.config.prkt.models.first())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no Parakeet models configured under [prkt.models]"))?;
        match pipeline::parakeet::run_parakeet_pipeline(&ctx, &parakeet_model, event_tx).await {
            Ok(result) => {
                eprintln!("\nTranscription complete.");
                if let Some(ref srt) = result.srt_file {
                    eprintln!("Output: {}", display_abs(srt));
                }
                for warn in &result.warnings {
                    eprintln!("Warning: {warn}");
                }
            }
            Err(e) => eprintln!("\nTranscription failed: {e}"),
        }
    } else if mode.uses_omnilingual() {
        let omni_model = ctx
            .config
            .omni
            .models
            .iter()
            .find(|m| m.name == ctx.selected_model)
            .or_else(|| ctx.config.omni.models.iter().find(|m| !m.broken))
            .or_else(|| ctx.config.omni.models.first())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no Omnilingual models configured under [omni.models]"))?;
        match pipeline::omnilingual::run_omnilingual_pipeline(&ctx, &omni_model, event_tx).await {
            Ok(result) => {
                eprintln!("\nTranscription complete.");
                if let Some(ref srt) = result.srt_file {
                    eprintln!("Output: {}", display_abs(srt));
                }
                for warn in &result.warnings {
                    eprintln!("Warning: {warn}");
                }
            }
            Err(e) => eprintln!("\nTranscription failed: {e}"),
        }
    } else {
        eprintln!("Mode {} not yet wired for headless execution", mode.mode_tag());
    }

    printer.abort();
    Ok(())
}

fn generate_default_config(target: &Option<PathBuf>) -> anyhow::Result<()> {
    let path = target
        .clone()
        .unwrap_or_else(|| PathBuf::from("fwxl-burn.toml"));

    if path.exists() {
        eprintln!("Config file already exists: {}", path.display());
        eprintln!("Delete it first if you want to regenerate.");
        std::process::exit(1);
    }

    std::fs::write(&path, config::DEFAULT_CONFIG_TOML)?;
    eprintln!("Generated default config: {}", path.display());
    eprintln!("Edit tool paths and settings, then run fwxl-burn normally.");

    Ok(())
}
