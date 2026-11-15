use crate::config::AppConfig;
use crate::media::MediaFile;
use crate::pipeline::{
    self, PipelineContext, PipelineEvent, PipelineResult, SubtitleStyle, TranscriptionMode,
};
use crate::run_manifest::{self, ItemStatus, RunManifest};
use std::cell::Cell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Which screen is currently displayed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Screen {
    Start,
    SubtitleSelection,
    ModelSelect,
    ParakeetSelect,
    OmniSelect,
    /// Lists every SRT discovered for the current media (SRCDIR + SRT_DIR),
    /// auto-shown between transcription completion and StyleSelect so the
    /// user can hardburn an existing subtitle, not just the freshly produced
    /// one. Also reachable from the batch Menu via `[H]`.
    SrtPicker,
    StyleSelect,
    TranslateMenu,
    Progress,
    LogViewer,
    Menu,
}

/// How the user is feeding input to the app. Controls whether the pipeline runs
/// once (File) or loops over a queue (BatchFolder, PasteQueue), and whether
/// the StyleSelect/Hardburn screens are reachable (File + video only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// Single file -- current behavior. Hardburn offered for video.
    File,
    /// Directory of media files. Iterated; hardburn skipped.
    BatchFolder,
    /// Multiple pasted paths (Gallery-dl "run" style). Iterated; hardburn skipped.
    PasteQueue,
}

/// One row in the batch summary shown on Menu after a queue finishes.
/// Accumulated by `drain_pipeline_events` (one per Completed event) so the
/// user can see EVERY transcribed/hardcoded output, not just the last.
#[derive(Debug, Clone)]
pub struct BatchItemSummary {
    pub stem: String,
    pub media_path: PathBuf,
    pub is_video: bool,
    pub srt_path: Option<PathBuf>,
    pub video_path: Option<PathBuf>,
    pub error: Option<String>,
}

/// State for the opt-in batch hardburn flow triggered by `[H]` on the Menu
/// after a multi-item queue finishes. Iterates `batch_results` indices
/// pointing at video items with verified SRTs; each iteration loads the
/// matching media, scans SRTs, and routes through SrtPicker -> StyleSelect
/// -> hardburn -> back to Menu when the queue empties.
#[derive(Debug, Clone)]
pub struct PendingHardburn {
    pub indices: Vec<usize>,
    pub current: usize,
}

/// Full TUI application state.
pub struct AppState {
    pub config: AppConfig,
    pub current_screen: Screen,

    // Input state
    pub input_path: String,
    pub input_cursor: usize, // byte offset into input_path for cursor position
    pub input_is_valid: bool,
    pub input_is_directory: bool,
    pub media_file: Option<MediaFile>,
    /// Files passed via CLI drag-and-drop OR built up by a paste-queue screen
    /// (multiple = batch mode).
    pub input_files: Vec<std::path::PathBuf>,
    /// Index into input_files for batch processing.
    pub batch_index: usize,
    /// Which top-level input flow is active. `None` = start screen mode picker
    /// still needs a choice. Drag-drop auto-sets this (single=File, multi=PasteQueue).
    pub input_mode: Option<InputMode>,
    /// Draft line being typed on the paste-queue subscreen (before Enter commits it).
    pub paste_queue_draft: String,

    // Selection state
    pub selected_mode: Option<TranscriptionMode>,
    pub selected_model: Option<String>,
    pub selected_style: Option<SubtitleStyle>,
    pub available_models: Vec<String>,
    pub parakeet_model_index: usize,
    /// Selection cursor for the Omnilingual model list.
    pub omni_model_index: usize,

    // List selection indices
    pub mode_list_index: usize,
    pub model_list_index: usize,
    pub style_list_index: usize,

    // Pipeline state
    pub pipeline_event_rx: Option<mpsc::UnboundedReceiver<PipelineEvent>>,
    pub pipeline_cancel: Option<CancellationToken>,
    pub pipeline_result: Option<PipelineResult>,
    pub current_phase: String,
    pub current_progress: f32,
    pub progress_message: String,
    pub pipeline_running: bool,
    /// The file the *currently running* pipeline is working on.
    ///
    /// `media_file` tracks what the UI is pointed at, which in a batch can
    /// already have advanced to the next item by the time a slow item's final
    /// event is drained. Attributing results by `media_file` therefore pinned
    /// one file's error onto another file's row. This is captured when the
    /// pipeline is spawned and is the authoritative owner of its events.
    pub pipeline_media: Option<MediaFile>,

    // Log
    pub log_lines: VecDeque<String>,
    pub log_scroll: usize,
    /// Inner height (rows) of the log pane at last render. Used by input handlers
    /// to cap `log_scroll` so the user cannot scroll past the top of the buffer.
    /// `Cell` so the immutable `render(&app)` path can still update it.
    pub log_viewport_height: Cell<usize>,

    // Derived display strings cached once when the result is set, so render
    // paths don't hit the filesystem (`canonicalize`, recursive dir scans) every
    // single frame. Populated by `set_pipeline_result`.
    pub srt_display_path: Option<String>,
    pub video_display_path: Option<String>,
    pub english_master_display: Option<String>,

    // SRT picker state (auto-shown between transcribe and StyleSelect, also
    // reachable from Menu [H]). Populated by `populate_available_srts`.
    pub available_srts: Vec<crate::probe::scan::SrtCandidate>,
    pub srt_picker_dirs: Vec<PathBuf>,
    pub srt_picker_index: usize,
    /// SRT that the user picked in SrtPicker -- consumed by `launch_hardburn_pipeline`.
    pub srt_for_hardburn: Option<PathBuf>,

    // Batch reporting + opt-in batch hardburn.
    /// Accumulated per-item results so the Menu can show every transcribed
    /// output for a multi-item queue, not just the last one set into
    /// `pipeline_result`.
    pub batch_results: Vec<BatchItemSummary>,
    /// Active iterator for the [H] batch hardburn flow.
    pub pending_hardburn: Option<PendingHardburn>,
    /// Scroll offset (rows from the top) for the Menu's batch results pane.
    pub menu_scroll: usize,
    /// Total rendered lines / inner height of the batch results pane at last
    /// render, so key handlers can clamp `menu_scroll`. Cells because the
    /// immutable render path updates them.
    pub menu_summary_lines: Cell<usize>,
    pub menu_viewport_height: Cell<usize>,

    // Run persistence.
    /// Live manifest for the current multi-item run; mirrored to
    /// `last_run.json` after every item so a crash preserves progress.
    pub run_manifest: Option<RunManifest>,
    /// Manifest loaded from disk at startup (a previous session's run),
    /// consumed by `[R] Redo last run` on the Start mode picker.
    pub saved_manifest: Option<RunManifest>,
    /// The full file list of the current run, before any [F] retry narrowed
    /// `input_files` down to the failures. `[A] Run again` restores from it.
    pub original_queue: Vec<PathBuf>,

    // Mouse state
    pub mouse_row: u16,
    pub mouse_col: u16,
    pub list_area: Option<ratatui::layout::Rect>,
}

impl AppState {
    pub fn new(config: AppConfig, input_files: Vec<std::path::PathBuf>) -> Self {
        // If exactly one file was drag-dropped, pre-fill the input path
        let (input_path, input_is_valid) = if input_files.len() == 1 {
            let p = input_files[0].to_string_lossy().to_string();
            (p, true)
        } else {
            (String::new(), false)
        };
        let input_is_directory = input_files.len() == 1 && input_files[0].is_dir();

        // Auto-route drag-drop: single file/dir picks its natural mode, multiple
        // files implicitly become a paste-queue so the user confirms + locks
        // settings once. Typing at the start screen (no drag-drop) leaves this
        // as None so the F/B/P picker appears.
        let input_mode = if input_files.len() == 1 {
            if input_is_directory {
                Some(InputMode::BatchFolder)
            } else {
                Some(InputMode::File)
            }
        } else if input_files.len() > 1 {
            Some(InputMode::PasteQueue)
        } else {
            None
        };

        let cursor = input_path.len();
        Self {
            config,
            current_screen: Screen::Start,

            input_path,
            input_cursor: cursor,
            input_is_valid,
            input_is_directory,
            media_file: None,
            input_files,
            batch_index: 0,
            input_mode,
            paste_queue_draft: String::new(),

            selected_mode: None,
            selected_model: None,
            selected_style: None,
            available_models: Vec::new(),
            parakeet_model_index: 0,
            omni_model_index: 0,

            mode_list_index: 0,
            model_list_index: 0,
            style_list_index: 0,

            pipeline_event_rx: None,
            pipeline_cancel: None,
            pipeline_result: None,
            current_phase: String::new(),
            current_progress: 0.0,
            progress_message: String::new(),
            pipeline_running: false,
            pipeline_media: None,

            log_lines: VecDeque::with_capacity(10000),
            log_scroll: 0,
            log_viewport_height: Cell::new(0),

            srt_display_path: None,
            video_display_path: None,
            english_master_display: None,

            available_srts: Vec::new(),
            srt_picker_dirs: Vec::new(),
            srt_picker_index: 0,
            srt_for_hardburn: None,

            batch_results: Vec::new(),
            pending_hardburn: None,
            menu_scroll: 0,
            menu_summary_lines: Cell::new(0),
            menu_viewport_height: Cell::new(0),

            run_manifest: None,
            saved_manifest: run_manifest::load(),
            original_queue: Vec::new(),

            mouse_row: 0,
            mouse_col: 0,
            list_area: None,
        }
    }

    /// "[i/N] name.ext" for queue runs, "name.ext" for single files: the file
    /// the running pipeline is on (the UI's selection when nothing runs).
    /// Previously the only place the current file's name appeared was deep in
    /// the log, which a 1,500-item batch scrolls off in seconds.
    pub fn current_file_indicator(&self) -> Option<String> {
        let media = self.pipeline_media.as_ref().or(self.media_file.as_ref())?;
        let name = media
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| media.stem.clone());
        if self.is_queue_mode() {
            Some(format!(
                "[{}/{}] {name}",
                self.batch_index + 1,
                self.input_files.len()
            ))
        } else {
            Some(name)
        }
    }

    /// Console window title reflecting what the app is doing right now.
    /// The event loop pushes it whenever it changes.
    pub fn window_title(&self) -> String {
        const BASE: &str = concat!("fwxl-burn v", env!("CARGO_PKG_VERSION"));
        if self.pipeline_running {
            if let Some(file) = self.current_file_indicator() {
                let mode = self
                    .selected_mode
                    .map(|m| m.key().to_string())
                    .unwrap_or_else(|| "?".to_string());
                let model = self.selected_model.as_deref().unwrap_or("default");
                let pct = self.current_progress.clamp(0.0, 100.0).round() as u32;
                return format!("{file}  |  {mode} / {model}  |  {pct}%  -  fwxl-burn");
            }
        }
        if self.current_screen == Screen::Menu && !self.batch_results.is_empty() {
            let failed = self.batch_results.iter().filter(|r| r.error.is_some()).count();
            let ok = self.batch_results.len() - failed;
            return format!("Done: {ok} ok / {failed} failed  -  fwxl-burn");
        }
        BASE.to_string()
    }

    /// Canonicalize a path for display, stripping the Windows `\\?\` prefix.
    /// Falls back to plain `display()` on error.
    fn display_abs(p: &std::path::Path) -> String {
        let s = std::fs::canonicalize(p)
            .map(|c| c.display().to_string())
            .unwrap_or_else(|_| p.display().to_string());
        s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
    }

    /// Store a pipeline result and pre-compute display strings ONCE
    /// (expensive filesystem calls should not happen in render paths).
    pub fn set_pipeline_result(&mut self, result: PipelineResult) {
        self.srt_display_path = result.srt_file.as_deref().map(Self::display_abs);
        self.video_display_path = result.hardcoded_video.as_deref().map(Self::display_abs);
        // Refresh master-SRT cache (used by the Translate menu).
        if let Some(ref media) = self.media_file {
            self.english_master_display = crate::probe::scan::find_english_master(&media.dir, &media.stem)
                .as_deref()
                .map(Self::display_abs);
        }
        self.pipeline_result = Some(result);
    }

    /// Maximum valid value of `log_scroll` given the current buffer and last known viewport height.
    /// Past this, scrolling further up shows the same top-of-log content with no visual change.
    pub fn max_log_scroll(&self) -> usize {
        self.log_lines
            .len()
            .saturating_sub(self.log_viewport_height.get())
    }

    /// Increment log_scroll toward the top of the buffer, clamped.
    pub fn scroll_log_up(&mut self) {
        let max = self.max_log_scroll();
        if self.log_scroll < max {
            self.log_scroll += 1;
        }
    }

    /// Decrement log_scroll toward the latest lines (bottom), clamped at 0.
    pub fn scroll_log_down(&mut self) {
        self.log_scroll = self.log_scroll.saturating_sub(1);
    }

    /// Drain pending pipeline events from the background task channel.
    pub fn drain_pipeline_events(&mut self) {
        // Collect events first to avoid borrow conflicts
        let events: Vec<PipelineEvent> = if let Some(ref mut rx) = self.pipeline_event_rx {
            let mut batch = Vec::new();
            while let Ok(event) = rx.try_recv() {
                batch.push(event);
            }
            batch
        } else {
            return;
        };

        for event in events {
            match &event {
                PipelineEvent::PhaseStarted { name, .. } => {
                    self.current_phase = name.clone();
                    self.current_progress = 0.0;
                    self.progress_message.clear();
                }
                PipelineEvent::Progress { percent, message } => {
                    self.current_progress = *percent;
                    self.progress_message = message.clone();
                }
                PipelineEvent::StdoutLine(line) | PipelineEvent::StderrLine(line) => {
                    self.push_log(line.clone());
                }
                PipelineEvent::Completed(result) => {
                    let has_srt = result.srt_file.is_some();
                    let has_new_video = result.hardcoded_video.is_some();
                    self.set_pipeline_result(result.clone());
                    self.pipeline_running = false;

                    // Per-item summary push so the Menu can show every
                    // transcribed/hardcoded output for a queue, not just the
                    // last result that overwrote pipeline_result.
                    self.record_batch_item(&result);

                    // Hardburn batch advancement: this Completed came from
                    // ffmpeg, not whisper, and we're iterating a [H]-launched
                    // queue of pre-transcribed videos. Move to the next.
                    if self.pending_hardburn.is_some() {
                        if self.advance_pending_hardburn() {
                            self.current_screen = Screen::SrtPicker;
                            continue;
                        }
                        self.pending_hardburn = None;
                        self.current_screen = Screen::Menu;
                        continue;
                    }

                    let is_native = self.selected_mode == Some(TranscriptionMode::Native);

                    // NATIVE EXTRACTION COMPLETED (any input mode): always
                    // pause for SrtPicker so the user can choose between the
                    // freshly extracted SRT and ANY on-disk SRT in SRCDIR /
                    // SRT_DIR. "Native" means "I'm using existing subtitles"
                    // -- the picker is what actually surfaces them. Without
                    // this, batch/paste-queue Native auto-advanced through
                    // every file with no chance for the user to pick a
                    // disk SRT for items where extraction returned nothing
                    // useful, and even single-file users got dumped at Menu
                    // with no probe of their SRCDIR / SRT_DIR contents.
                    if !has_new_video && is_native {
                        if let Some(media) = self.media_file.clone() {
                            self.populate_available_srts(&media);
                        }
                        self.current_screen = Screen::SrtPicker;
                        continue;
                    }

                    // NATIVE BATCH HARDBURN COMPLETED: advance to next item
                    // and re-run native extraction so item N+1 gets its own
                    // SrtPicker visit. (For non-Native batches this would
                    // have already auto-advanced via the generic path below
                    // BEFORE the Native check above had a chance to fire,
                    // because non-Native modes don't gate on SrtPicker.)
                    if has_new_video
                        && is_native
                        && self.has_next_in_queue()
                        && self.advance_to_next_queue_item()
                    {
                        let idx = self.batch_index + 1;
                        let total = self.input_files.len();
                        self.push_log(format!(
                            "[QUEUE] Native: advancing to item {idx}/{total}"
                        ));
                        self.launch_transcription_pipeline();
                        // A skip/recover sweep may have drained the queue
                        // without launching; it navigates to Menu itself.
                        if self.pipeline_running {
                            self.current_screen = Screen::Progress;
                        }
                        continue;
                    }

                    // Generic transcription queue advancement (Whisper /
                    // Parakeet): SrtPicker/StyleSelect/Hardburn are SKIPPED
                    // between queue items -- the rename to canonical name
                    // has already happened inside the pipeline. Opt-in batch
                    // hardburn is offered from the Menu after the queue
                    // ends. Native is excluded above because each item must
                    // pause for the picker.
                    if self.has_next_in_queue() && self.advance_to_next_queue_item() {
                        let idx = self.batch_index + 1;
                        let total = self.input_files.len();
                        self.push_log(format!("[QUEUE] Advancing to item {idx}/{total}"));
                        self.launch_transcription_pipeline();
                        if self.pipeline_running {
                            self.current_screen = Screen::Progress;
                        }
                        continue;
                    }

                    // Terminal state. Single-file with a video: scan SRCDIR
                    // and SRT_DIR, jump to SrtPicker so the user can pick
                    // ANY existing SRT to hardburn (not just the just-made
                    // one). This is the auto-show-before-StyleSelect path.
                    // Anything else -> Menu (which renders the batch summary
                    // when applicable, with [H] to opt into batch hardburn).
                    if !has_new_video && has_srt && self.should_offer_hardburn() {
                        if let Some(media) = self.media_file.clone() {
                            self.populate_available_srts(&media);
                        }
                        self.current_screen = Screen::SrtPicker;
                    } else {
                        self.current_screen = Screen::Menu;
                    }
                }
                PipelineEvent::Warning(msg) => {
                    self.push_log(format!("[WARN] {msg}"));
                }
                PipelineEvent::Error(msg) => {
                    self.push_log(format!("[ERROR] {msg}"));
                    self.pipeline_running = false;
                    self.record_batch_error(msg);

                    // Pending hardburn iteration: don't let one failed
                    // hardburn abort the whole [H] queue.
                    if self.pending_hardburn.is_some() {
                        if self.advance_pending_hardburn() {
                            self.current_screen = Screen::SrtPicker;
                            continue;
                        }
                        self.pending_hardburn = None;
                        self.current_screen = Screen::Menu;
                        continue;
                    }

                    // NATIVE FALLBACK: even when extraction fails (no text
                    // subs, ffmpeg crashed, etc), still scan SRCDIR / SRT_DIR
                    // and route to SrtPicker so the user can hardburn a
                    // disk SRT instead of being dumped at the Menu. Applies
                    // to single-file AND every queue item -- Native means
                    // "use existing subtitles", and the picker is the entry
                    // point to existing subtitles. Skip from the picker
                    // advances the queue (see srt_picker.rs).
                    if self.selected_mode == Some(TranscriptionMode::Native) {
                        if let Some(media) = self.media_file.clone() {
                            self.populate_available_srts(&media);
                        }
                        self.current_screen = Screen::SrtPicker;
                        continue;
                    }

                    // On error mid-queue, skip to next item rather than stopping
                    // the whole batch. The user can abort with Ctrl+C.
                    if self.has_next_in_queue() && self.advance_to_next_queue_item() {
                        let idx = self.batch_index + 1;
                        let total = self.input_files.len();
                        self.push_log(format!(
                            "[QUEUE] Skipping failed item; advancing to {idx}/{total}"
                        ));
                        self.launch_transcription_pipeline();
                        if self.pipeline_running {
                            self.current_screen = Screen::Progress;
                        }
                        continue;
                    }
                    self.current_screen = Screen::Menu;
                }
                PipelineEvent::PhaseCompleted { name, duration } => {
                    self.push_log(format!(
                        "[DONE] {name} ({:.1}s)",
                        duration.as_secs_f64()
                    ));
                }
            }
        }
    }

    /// True when we're iterating a multi-item queue (batch folder or paste queue).
    pub fn is_queue_mode(&self) -> bool {
        matches!(
            self.input_mode,
            Some(InputMode::BatchFolder) | Some(InputMode::PasteQueue)
        ) && self.input_files.len() > 1
    }

    /// True when at least one more queue item remains past `batch_index`.
    pub fn has_next_in_queue(&self) -> bool {
        self.is_queue_mode() && self.batch_index + 1 < self.input_files.len()
    }

    /// Hardburn is a surgical single-file operation. Skip it for batch/paste
    /// queues and for audio-only inputs regardless of mode.
    pub fn should_offer_hardburn(&self) -> bool {
        if self.is_queue_mode() {
            return false;
        }
        match self.media_file {
            Some(ref m) => m.is_video(),
            None => false,
        }
    }

    /// Advance to the next item in the queue: bump `batch_index`, reload the
    /// `MediaFile`, and clear transient per-item state. Settings (mode, model,
    /// language, style) are preserved so every queued item uses the same
    /// config the user chose on item 1. Returns false if the probe fails.
    pub fn advance_to_next_queue_item(&mut self) -> bool {
        if !self.has_next_in_queue() {
            return false;
        }
        self.batch_index += 1;
        let next_path = self.input_files[self.batch_index].clone();
        self.input_path = next_path.to_string_lossy().to_string();
        self.input_cursor = self.input_path.len();
        self.input_is_valid = true;
        self.input_is_directory = next_path.is_dir();

        match MediaFile::from_path(&next_path) {
            Ok(mut media) => {
                let config = self.config.clone();
                let probe_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(crate::probe::probe_media_file(&config, &mut media))
                });
                match probe_result {
                    Err(e) => self.push_log(format!("[WARN] Probe failed for {}: {e}", media.stem)),
                    Ok(status) => {
                        if let Some(reason) = status.reason() {
                            self.push_log(format!(
                                "[ERR ] {} is unreadable: {reason}",
                                media.stem
                            ));
                        }
                    }
                }
                self.media_file = Some(media);
            }
            Err(e) => {
                self.push_log(format!("[ERROR] Failed to load next queue item: {e}"));
                return false;
            }
        }

        // Per-item state reset (keep logs rolling across the queue so the user
        // can scroll back through earlier items).
        self.pipeline_result = None;
        self.srt_display_path = None;
        self.video_display_path = None;
        self.english_master_display = None;
        self.srt_for_hardburn = None;
        self.available_srts.clear();
        self.srt_picker_dirs.clear();
        self.srt_picker_index = 0;
        self.current_phase.clear();
        self.current_progress = 0.0;
        self.progress_message.clear();
        true
    }

    /// Spawn the transcription pipeline as a background tokio task.
    ///
    /// May not actually launch: for multi-item queues, a pre-flight sweep can
    /// skip items already transcribed with this mode+model (or recover them
    /// from a quarantine dir) and advance past them. When that sweep drains
    /// the whole queue, this returns with `pipeline_running == false` and the
    /// screen already set to Menu -- callers must gate their "navigate to
    /// Progress" on `pipeline_running`.
    pub fn launch_transcription_pipeline(&mut self) {
        let mode = match self.selected_mode {
            Some(m) => m,
            None => {
                self.push_log("[ERROR] No mode selected".to_string());
                return;
            }
        };
        let model = self
            .selected_model
            .clone()
            .unwrap_or_else(|| self.config.whisper.default_model.clone());

        // Stamp settings into the persistent manifest (the queue is committed
        // to `last_run.json` before mode/model are picked).
        if let Some(m) = self.run_manifest.as_mut() {
            let key = mode.key().to_string();
            if m.mode_key != key || m.model != model {
                m.mode_key = key;
                m.model = model.clone();
                m.save();
            }
        }

        // Pre-flight skip/recover sweep for queues. Advances `media_file`
        // past pre-finished items; false means nothing is left to run.
        if self.is_queue_mode() && !self.preflight_queue_item(mode, &model) {
            self.push_log("[QUEUE] All remaining items were already transcribed".to_string());
            self.current_screen = Screen::Menu;
            return;
        }

        let media = match self.media_file.clone() {
            Some(m) => m,
            None => {
                self.push_log("[ERROR] No media file loaded".to_string());
                return;
            }
        };

        let config = self.config.clone();
        let cancel = CancellationToken::new();
        self.pipeline_cancel = Some(cancel.clone());

        let srt_dir = self.resolve_srt_dir(&media);
        let final_out_dir = config
            .output_dir
            .clone()
            .unwrap_or_else(|| media.dir.clone());
        let _ = std::fs::create_dir_all(&srt_dir);
        let _ = std::fs::create_dir_all(&final_out_dir);

        let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M").to_string();

        // Pin result/error attribution to this file for the life of the run.
        // Recorded before any repair step re-points the context, so the batch
        // row stays keyed to the file the user actually queued.
        self.pipeline_media = Some(media.clone());

        let ctx = PipelineContext {
            config,
            media,
            mode,
            selected_model: model,
            srt_dir,
            final_out_dir,
            timestamp,
            is_batch: false,
            cancel,
        };

        let (tx, rx) = mpsc::unbounded_channel();
        self.pipeline_event_rx = Some(rx);
        self.pipeline_running = true;
        self.current_phase = "Starting...".to_string();
        self.current_progress = 0.0;
        // Don't clear log_lines here -- when iterating a batch/paste queue, the
        // user should be able to scroll back through earlier items. Logs are
        // reset explicitly by `reset_for_new_file` on Menu -> Restart.

        // Spawn the pipeline on the tokio runtime
        tokio::spawn(async move {
            // Pre-flight: an unreadable container is repaired (to a separate
            // file) before any transcription starts, rather than being
            // discovered an hour later as an empty output directory.
            let mut ctx = ctx;
            if let Err(e) = pipeline::repair::ensure_readable(&mut ctx, &tx).await {
                let _ = tx.send(PipelineEvent::Error(format!("{e}")));
                return;
            }
            let ctx = ctx;

            let result = if mode.uses_whisper() {
                pipeline::whisper::run_whisper_pipeline(&ctx, tx.clone()).await
            } else if mode == TranscriptionMode::Native {
                // Extract the best embedded text subtitle to an SRT, then
                // drain_pipeline_events routes the Completed result through
                // the SrtPicker just like the Whisper/Parakeet flows do.
                pipeline::native::run_native_pipeline(&ctx, tx.clone()).await
            } else if mode.uses_parakeet() {
                // Honor the model the user picked in ParakeetSelect; fall
                // back to the first non-broken model in config order.
                let selected_name = ctx.selected_model.as_str();
                let parakeet_model = ctx
                    .config
                    .prkt
                    .models
                    .iter()
                    .find(|m| m.name == selected_name)
                    .or_else(|| ctx.config.prkt.models.iter().find(|m| !m.broken))
                    .or_else(|| ctx.config.prkt.models.first())
                    .cloned()
                    .expect("at least one Parakeet model configured");
                pipeline::parakeet::run_parakeet_pipeline(&ctx, &parakeet_model, tx.clone()).await
            } else if mode.uses_omnilingual() {
                // Honor the model the user picked in OmniSelect; fall back
                // to the first non-broken model in config order.
                let selected_name = ctx.selected_model.as_str();
                let omni_model = ctx
                    .config
                    .omni
                    .models
                    .iter()
                    .find(|m| m.name == selected_name)
                    .or_else(|| ctx.config.omni.models.iter().find(|m| !m.broken))
                    .or_else(|| ctx.config.omni.models.first())
                    .cloned();
                match omni_model {
                    Some(m) => {
                        pipeline::omnilingual::run_omnilingual_pipeline(&ctx, &m, tx.clone()).await
                    }
                    None => Err(crate::error::PipelineError::ModeUnavailable {
                        mode: "Omnilingual".to_string(),
                        reason: "no models configured under [omni.models]".to_string(),
                    }),
                }
            } else {
                // Custom / Translate / Diagnostic / OCR -- not yet wired
                // through this entry point. Surface a clear error rather
                // than silently doing nothing.
                Err(crate::error::PipelineError::ModeUnavailable {
                    mode: format!("{:?}", mode),
                    reason: "Mode is not wired through launch_transcription_pipeline yet"
                        .to_string(),
                })
            };

            match result {
                Ok(res) => {
                    let _ = tx.send(PipelineEvent::Completed(res));
                }
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Error(format!("{e}")));
                }
            }
        });
    }

    /// Spawn the hardburn pipeline as a background tokio task. Prefers the
    /// SRT the user picked in `SrtPicker`, falling back to whatever SRT the
    /// most recent transcription produced. Verifies the chosen SRT exists,
    /// has bytes, and parses to >=1 timing entry BEFORE handing it to ffmpeg
    /// -- a 0-byte or malformed SRT would otherwise burn a perfectly mute
    /// video for the user.
    pub fn launch_hardburn_pipeline(&mut self) {
        let media = match self.media_file.clone() {
            Some(m) => m,
            None => return,
        };
        let srt_path = match self
            .srt_for_hardburn
            .clone()
            .or_else(|| self.pipeline_result.as_ref().and_then(|r| r.srt_file.clone()))
        {
            Some(p) => p,
            None => {
                self.push_log("[ERROR] No SRT file to hardburn".to_string());
                return;
            }
        };

        // Pre-flight verification: if this fails we abort BEFORE spawning
        // ffmpeg so the user gets a clear error message instead of a silent
        // mute video burn.
        let v = crate::srt::verify_srt(&srt_path);
        if !v.valid {
            self.push_log(format!(
                "[ERROR] Refusing to hardburn invalid SRT: {} ({})",
                srt_path.display(),
                v.message
            ));
            self.current_screen = Screen::SrtPicker;
            return;
        }
        self.push_log(format!(
            "[OK] Verified SRT for hardburn: {} ({})",
            srt_path.display(),
            v.message
        ));

        let style = match self.selected_style {
            Some(s) => s,
            None => return,
        };

        let config = self.config.clone();
        let cancel = CancellationToken::new();
        self.pipeline_cancel = Some(cancel.clone());

        let final_out_dir = config
            .output_dir
            .clone()
            .unwrap_or_else(|| media.dir.clone());
        let _ = std::fs::create_dir_all(&final_out_dir);
        let timestamp = chrono::Local::now().format("%Y-%m-%d-%H%M").to_string();
        let srt_dir = srt_path
            .parent()
            .unwrap_or(&media.dir)
            .to_path_buf();

        let prev_result = self.pipeline_result.clone();
        self.pipeline_media = Some(media.clone());

        let ctx = PipelineContext {
            config,
            media,
            mode: self.selected_mode.unwrap_or(TranscriptionMode::Unfiltered),
            selected_model: self
                .selected_model
                .clone()
                .unwrap_or_else(|| "large-v2".to_string()),
            srt_dir,
            final_out_dir,
            timestamp,
            is_batch: false,
            cancel,
        };

        let (tx, rx) = mpsc::unbounded_channel();
        self.pipeline_event_rx = Some(rx);
        self.pipeline_running = true;
        self.current_phase = "Starting hardburn...".to_string();
        self.current_progress = 0.0;

        tokio::spawn(async move {
            match pipeline::hardburn::run_hardburn(&ctx, &srt_path, style, tx.clone()).await {
                Ok(video_path) => {
                    let mut result = prev_result.unwrap_or(PipelineResult {
                        srt_file: None,
                        hardcoded_video: None,
                        lang_code: String::new(),
                        mode_tag: String::new(),
                        warnings: Vec::new(),
                    });
                    result.hardcoded_video = Some(video_path);
                    let _ = tx.send(PipelineEvent::Completed(result));
                }
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Error(format!("Hardburn failed: {e}")));
                }
            }
        });
    }

    /// Cancel any running pipeline.
    pub fn cancel_pipeline(&mut self) {
        if let Some(ref cancel) = self.pipeline_cancel {
            cancel.cancel();
        }
        self.pipeline_running = false;
    }

    /// Resolve the SRT output directory based on config.
    /// `pub(crate)` so the SrtPicker screen can label `[SRT_DIR]` with the
    /// same path the pipeline writes to.
    pub(crate) fn resolve_srt_dir(&self, media: &MediaFile) -> PathBuf {
        if self.config.use_source_folder {
            let dir = media.dir.join("source");
            let _ = std::fs::create_dir_all(&dir);
            dir
        } else if let Some(ref out) = self.config.output_dir {
            let _ = std::fs::create_dir_all(out);
            out.clone()
        } else {
            media.dir.clone()
        }
    }

    /// Scan SRCDIR (`media.dir`, beside the file) and SRT_DIR (the resolved
    /// output dir; typically `media.dir/source` when `use_source_folder` is
    /// true). Dedupes by canonical path so the two-locations-pointing-at-
    /// the-same-place case doesn't show every SRT twice. Pre-selects the
    /// just-produced SRT (from `pipeline_result`) when present, else the
    /// first entry. Reuses `find_existing_srts` -- the same scanner the
    /// Translate flow uses for its English-master detection.
    pub fn populate_available_srts(&mut self, media: &MediaFile) {
        let srt_dir = self.resolve_srt_dir(media);

        // SRCDIR first so duplicates resolve in favor of the user-visible
        // location next to the media file.
        let mut dirs: Vec<PathBuf> = vec![media.dir.clone()];
        if srt_dir != media.dir {
            dirs.push(srt_dir);
        }

        let mut found = crate::probe::scan::find_srts_in_dirs(&dirs, &media.stem);

        // Sort: the just-produced SRT first (so Enter on entry hardburns the
        // freshly transcribed file with zero extra keystrokes), then by
        // filename for stable ordering.
        let pending = self
            .pipeline_result
            .as_ref()
            .and_then(|r| r.srt_file.clone());
        found.sort_by(|a, b| {
            let a_pending = pending.as_ref().is_some_and(|p| paths_equal(p, &a.path));
            let b_pending = pending.as_ref().is_some_and(|p| paths_equal(p, &b.path));
            match (a_pending, b_pending) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.path.file_name().cmp(&b.path.file_name()),
            }
        });

        self.available_srts = found;
        self.srt_picker_dirs = dirs;
        self.srt_picker_index = 0;
    }

    /// The media file a just-drained pipeline event belongs to.
    ///
    /// Prefers the file the running pipeline was started for; falls back to the
    /// UI's current selection for events raised outside a pipeline run.
    fn owning_media(&self) -> Option<MediaFile> {
        self.pipeline_media
            .as_ref()
            .or(self.media_file.as_ref())
            .cloned()
    }

    /// Append (or update) a batch summary row for a Completed pipeline event.
    /// Hardburn Completed events arrive AFTER a transcription Completed for
    /// the same media file -- when the just-arrived event is a hardburn (it
    /// carries `hardcoded_video`), update the existing matching row in place
    /// rather than appending a duplicate.
    fn record_batch_item(&mut self, result: &PipelineResult) {
        let media = match self.owning_media() {
            Some(m) => m,
            None => return,
        };
        self.update_manifest(
            &media.path,
            ItemStatus::Done,
            result.srt_file.clone(),
            None,
        );

        // Match by canonical media path so a hardburn done on the same media
        // updates the row created by the transcription pass.
        let matches_existing = |row: &BatchItemSummary| paths_equal(&row.media_path, &media.path);
        if let Some(existing) = self.batch_results.iter_mut().find(|r| matches_existing(r)) {
            if result.srt_file.is_some() && existing.srt_path.is_none() {
                existing.srt_path = result.srt_file.clone();
            }
            if let Some(ref v) = result.hardcoded_video {
                existing.video_path = Some(v.clone());
            }
            existing.error = None;
            return;
        }

        self.batch_results.push(BatchItemSummary {
            stem: media.stem.clone(),
            media_path: media.path.clone(),
            is_video: media.is_video(),
            srt_path: result.srt_file.clone(),
            video_path: result.hardcoded_video.clone(),
            error: None,
        });
    }

    /// Record a pipeline failure for the current media so the batch summary
    /// shows what didn't make it. Called from the Error branch of
    /// `drain_pipeline_events`.
    fn record_batch_error(&mut self, err: &str) {
        let media = match self.owning_media() {
            Some(m) => m,
            None => return,
        };
        self.update_manifest(
            &media.path,
            ItemStatus::Failed,
            None,
            Some(err.to_string()),
        );
        if let Some(existing) = self
            .batch_results
            .iter_mut()
            .find(|r| paths_equal(&r.media_path, &media.path))
        {
            existing.error = Some(err.to_string());
            return;
        }
        self.batch_results.push(BatchItemSummary {
            stem: media.stem.clone(),
            media_path: media.path.clone(),
            is_video: media.is_video(),
            srt_path: None,
            video_path: None,
            error: Some(err.to_string()),
        });
    }

    /// Build the list of batch_results indices that are eligible for
    /// hardburn (video media + verified SRT) and start the iterator. Returns
    /// false if nothing eligible.
    pub fn start_pending_hardburn(&mut self) -> bool {
        let indices: Vec<usize> = self
            .batch_results
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                r.is_video
                    && r.video_path.is_none() // not yet hardburned
                    && r.srt_path
                        .as_ref()
                        .is_some_and(|p| crate::srt::verify_srt(p).valid)
            })
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            return false;
        }
        self.pending_hardburn = Some(PendingHardburn { indices, current: 0 });
        self.load_pending_hardburn_current()
    }

    /// Step from the just-finished hardburn to the next eligible batch item.
    /// Returns true if another item was loaded (ready for SrtPicker).
    fn advance_pending_hardburn(&mut self) -> bool {
        let next_pos = match self.pending_hardburn.as_mut() {
            Some(ph) => {
                ph.current += 1;
                ph.current
            }
            None => return false,
        };
        let len = self
            .pending_hardburn
            .as_ref()
            .map(|ph| ph.indices.len())
            .unwrap_or(0);
        if next_pos >= len {
            return false;
        }
        self.load_pending_hardburn_current()
    }

    /// Load the media for the current `pending_hardburn` index and populate
    /// the SrtPicker. Returns false if the row doesn't have a usable media
    /// path on disk anymore.
    fn load_pending_hardburn_current(&mut self) -> bool {
        let (idx, total, current_pos) = {
            let ph = match self.pending_hardburn.as_ref() {
                Some(ph) => ph,
                None => return false,
            };
            (
                *ph.indices.get(ph.current).unwrap_or(&usize::MAX),
                ph.indices.len(),
                ph.current,
            )
        };
        let row = match self.batch_results.get(idx) {
            Some(r) => r.clone(),
            None => return false,
        };
        match MediaFile::from_path(&row.media_path) {
            Ok(mut media) => {
                let config = self.config.clone();
                let probe_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(crate::probe::probe_media_file(&config, &mut media))
                });
                match probe_result {
                    Err(e) => self.push_log(format!("[WARN] Probe failed for {}: {e}", media.stem)),
                    Ok(status) => {
                        if let Some(reason) = status.reason() {
                            self.push_log(format!(
                                "[ERR ] {} is unreadable: {reason}",
                                media.stem
                            ));
                        }
                    }
                }
                self.media_file = Some(media.clone());
                // Pre-fill pipeline_result with the SRT we already have so
                // SrtPicker auto-selects it and launch_hardburn_pipeline has
                // a sane fallback.
                self.pipeline_result = Some(PipelineResult {
                    srt_file: row.srt_path.clone(),
                    hardcoded_video: None,
                    lang_code: String::new(),
                    mode_tag: String::new(),
                    warnings: Vec::new(),
                });
                self.srt_display_path = row
                    .srt_path
                    .as_deref()
                    .map(Self::display_abs);
                self.video_display_path = None;
                self.populate_available_srts(&media);
                self.push_log(format!(
                    "[BATCH-HARDBURN] {}/{}: {}",
                    current_pos + 1,
                    total,
                    media.stem
                ));
                true
            }
            Err(e) => {
                self.push_log(format!(
                    "[ERROR] Cannot reload media for batch hardburn: {e}"
                ));
                false
            }
        }
    }

    /// The (prefix, tag) a canonical SRT would carry for `mode`, when the
    /// mode's output IS a deterministic function of (stem, mode, model).
    /// Modes without that property (Native, Custom, Translate, Diagnostic)
    /// return None and are never skip-checked.
    fn skip_identity(&self, mode: TranscriptionMode) -> Option<(String, &'static str)> {
        let n = &self.config.naming;
        if mode.uses_whisper() {
            Some((n.transcribe_prefix.clone(), mode.mode_tag()))
        } else if mode.uses_parakeet() {
            Some((n.parakeet_prefix.clone(), "P"))
        } else if mode.uses_omnilingual() {
            Some((n.omnilingual_prefix.clone(), "O"))
        } else {
            None
        }
    }

    /// Pre-flight for the current queue item: skip it when a canonical SRT
    /// for this exact mode+model already exists (the on-disk names are the
    /// transcription index), or rebuild it from a `.fwxl-recovery` folder a
    /// failed run left behind. Keeps advancing until an item genuinely needs
    /// transcription. Returns false when the sweep drained the queue.
    fn preflight_queue_item(&mut self, mode: TranscriptionMode, model: &str) -> bool {
        if !self.config.batch.skip_existing && !self.config.batch.recover_quarantine {
            return true;
        }
        let Some((prefix, tag)) = self.skip_identity(mode) else {
            return true;
        };
        loop {
            let media = match self.media_file.clone() {
                Some(m) => m,
                None => return true,
            };
            let srt_dir = self.resolve_srt_dir(&media);

            if self.config.batch.skip_existing {
                if let Some(existing) =
                    crate::naming::find_canonical_srt(&srt_dir, &media.stem, &prefix, tag, model)
                {
                    self.push_log(format!(
                        "[SKIP] {} -- already transcribed with {tag}/{model} ({})",
                        media.stem,
                        existing
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    ));
                    self.note_prefinished_item(&media, existing, ItemStatus::Skipped);
                    if self.has_next_in_queue() && self.advance_to_next_queue_item() {
                        continue;
                    }
                    return false;
                }
            }

            // Quarantine recovery only makes sense for whisper-family runs --
            // the in-process engines fail without leaving a recovery folder.
            if self.config.batch.recover_quarantine && mode.uses_whisper() {
                if let Some(rec) =
                    crate::pipeline::salvage::recover_from_quarantine(&srt_dir, &media.stem)
                {
                    let lang = rec.lang.clone().unwrap_or_else(|| "en".to_string());
                    let dest = srt_dir.join(crate::naming::canonical_srt_name(
                        &media.stem,
                        &prefix,
                        tag,
                        model,
                        &lang,
                    ));
                    match crate::srt::write_srt_file(&rec.entries, &dest) {
                        Ok(()) => {
                            self.push_log(format!(
                                "[RECOVER] {} -- rebuilt {} entries from {} in {}",
                                media.stem,
                                rec.entries.len(),
                                rec.source.label(),
                                rec.dir.display()
                            ));
                            self.note_prefinished_item(&media, dest, ItemStatus::Done);
                            if self.has_next_in_queue() && self.advance_to_next_queue_item() {
                                continue;
                            }
                            return false;
                        }
                        Err(e) => {
                            self.push_log(format!(
                                "[WARN] Could not write recovered SRT for {}: {e}",
                                media.stem
                            ));
                        }
                    }
                }
            }

            return true;
        }
    }

    /// Record a queue item that was finished WITHOUT running a pipeline
    /// (skip-existing hit, or quarantine recovery) so the Menu summary and
    /// the persistent manifest both list it.
    fn note_prefinished_item(&mut self, media: &MediaFile, srt: PathBuf, status: ItemStatus) {
        if let Some(existing) = self
            .batch_results
            .iter_mut()
            .find(|r| paths_equal(&r.media_path, &media.path))
        {
            if existing.srt_path.is_none() {
                existing.srt_path = Some(srt.clone());
            }
            existing.error = None;
        } else {
            self.batch_results.push(BatchItemSummary {
                stem: media.stem.clone(),
                media_path: media.path.clone(),
                is_video: media.is_video(),
                srt_path: Some(srt.clone()),
                video_path: None,
                error: None,
            });
        }
        self.update_manifest(&media.path, status, Some(srt), None);
    }

    /// Update + persist the live run manifest for one media path. No-op when
    /// no manifest is active (single-file runs).
    fn update_manifest(
        &mut self,
        media_path: &Path,
        status: ItemStatus,
        srt: Option<PathBuf>,
        error: Option<String>,
    ) {
        if let Some(m) = self.run_manifest.as_mut() {
            if run_manifest::set_status(m, media_path, status, srt, error) {
                m.save();
            }
        }
    }

    /// Probe `input_files[0]` into `media_file` and reset per-item state,
    /// leaving settings (mode/model/style) untouched. Used by the Menu's
    /// retry/requeue actions to restart a queue from its front.
    pub fn load_queue_front(&mut self) -> bool {
        let Some(first) = self.input_files.first().cloned() else {
            return false;
        };
        self.batch_index = 0;
        self.input_path = first.to_string_lossy().to_string();
        self.input_cursor = self.input_path.len();
        self.input_is_valid = true;
        self.input_is_directory = first.is_dir();
        match MediaFile::from_path(&first) {
            Ok(mut media) => {
                let config = self.config.clone();
                let probe_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(crate::probe::probe_media_file(&config, &mut media))
                });
                match probe_result {
                    Err(e) => self.push_log(format!("[WARN] Probe failed for {}: {e}", media.stem)),
                    Ok(status) => {
                        if let Some(reason) = status.reason() {
                            self.push_log(format!("[ERR ] {} is unreadable: {reason}", media.stem));
                        }
                    }
                }
                self.media_file = Some(media);
            }
            Err(e) => {
                self.push_log(format!("[ERROR] Failed to load queue item: {e}"));
                return false;
            }
        }
        self.pipeline_result = None;
        self.srt_display_path = None;
        self.video_display_path = None;
        self.english_master_display = None;
        self.srt_for_hardburn = None;
        self.available_srts.clear();
        self.srt_picker_dirs.clear();
        self.srt_picker_index = 0;
        self.current_phase.clear();
        self.current_progress = 0.0;
        self.progress_message.clear();
        true
    }

    /// Media paths of every batch row that ended in an error.
    pub fn failed_batch_paths(&self) -> Vec<PathBuf> {
        self.batch_results
            .iter()
            .filter(|r| r.error.is_some())
            .map(|r| r.media_path.clone())
            .collect()
    }

    /// `[F]` on the Menu: re-queue only the failed rows, same settings, and
    /// launch immediately. The live manifest is kept, so retried items flip
    /// from failed to done in `last_run.json` as they succeed.
    pub fn retry_failed_items(&mut self) -> bool {
        let failed = self.failed_batch_paths();
        if failed.is_empty() || self.selected_mode.is_none() {
            return false;
        }
        if self.input_mode.is_none() {
            self.input_mode = Some(InputMode::PasteQueue);
        }
        self.push_log(format!(
            "[RETRY] Re-running {} failed item(s) with the same settings",
            failed.len()
        ));
        self.input_files = failed;
        self.menu_scroll = 0;
        self.load_queue_front()
    }

    /// `[A]` on the Menu: restore the full original queue so the user can
    /// pick a DIFFERENT mode/model and run the same file list again. Starts
    /// a fresh manifest run.
    pub fn requeue_all_items(&mut self) -> bool {
        let mut all = if self.original_queue.is_empty() {
            self.batch_results
                .iter()
                .map(|r| r.media_path.clone())
                .collect::<Vec<_>>()
        } else {
            self.original_queue.clone()
        };
        all.retain(|p| p.exists());
        if all.is_empty() {
            return false;
        }
        if self.input_mode.is_none() {
            self.input_mode = Some(InputMode::PasteQueue);
        }
        self.push_log(format!(
            "[REDO] Re-queued {} item(s) -- pick a mode and model for the new pass",
            all.len()
        ));
        self.input_files = all;
        self.original_queue = self.input_files.clone();
        self.batch_results.clear();
        self.menu_scroll = 0;
        self.selected_mode = None;
        self.selected_model = None;
        let manifest = RunManifest::new(&self.input_files);
        manifest.save();
        self.run_manifest = Some(manifest);
        self.load_queue_front()
    }

    pub fn push_log(&mut self, msg: String) {
        self.log_lines.push_back(msg);
        if self.log_lines.len() > 10000 {
            self.log_lines.pop_front();
        }
    }

    /// Check if a mouse position is within a given rect, and return the row offset within it.
    pub fn mouse_list_index(&self, row: u16, area: ratatui::layout::Rect) -> Option<usize> {
        // Account for border (1 row top)
        if row > area.y && row < area.y + area.height - 1 && self.mouse_col >= area.x && self.mouse_col < area.x + area.width {
            Some((row - area.y - 1) as usize)
        } else {
            None
        }
    }
}

/// Compare two paths by canonical form when possible, falling back to a
/// straight `==` when canonicalization isn't available (e.g. one path no
/// longer exists). Used to dedupe SRTs across SRCDIR/SRT_DIR and to match
/// `batch_results` rows by media path.
fn paths_equal(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}
