use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph, Wrap};

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    // Decide layout up front. When there are >= 2 batch rows we expand the
    // results pane to ~60% of the screen so the user can actually see every
    // transcribed/hardcoded path. Single-file flow keeps the original
    // 7-row pane (matches what the previous Menu rendered).
    let multi = app.batch_results.len() >= 2;
    let (results_constraint, options_constraint) = if multi {
        (Constraint::Min(8), Constraint::Length(8))
    } else {
        (Constraint::Length(7), Constraint::Min(1))
    };

    let chunks = Layout::vertical([
        Constraint::Length(3),     // Title
        results_constraint,        // Result summary
        options_constraint,        // Options list
        Constraint::Length(2),     // Help
    ])
    .split(area);

    let title_text = if multi {
        format!(" Complete -- {} files processed ", app.batch_results.len())
    } else {
        " Complete ".to_string()
    };
    let title = Paragraph::new(title_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Green)),
        );
    frame.render_widget(title, chunks[0]);

    // Result summary panel.
    let summary_widget = if multi {
        render_batch_summary(app, chunks[1])
    } else {
        render_single_summary(app)
    };
    frame.render_widget(summary_widget, chunks[1]);

    // Options list. The [H] hardburn entry is dynamic:
    //   - hidden if no media we could hardburn (no SRT, or non-video)
    //   - shown for single-file (re-pick SRT for the just-finished file)
    //   - shown for batch with at least one video that has a verified SRT
    //     and is not yet hardburned
    let h_eligible = hardburn_eligible(app);
    let failed_count = app.failed_batch_paths().len();
    let f_eligible = failed_count > 0 && app.selected_mode.is_some();
    let a_eligible = !app.original_queue.is_empty() || app.batch_results.len() >= 2;
    let mut items: Vec<ListItem> = Vec::new();
    items.push(ListItem::new(Line::from(vec![
        Span::styled(
            " [R] ",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ),
        Span::raw("Restart - Process another file"),
    ])));
    if f_eligible {
        items.push(ListItem::new(Line::from(vec![
            Span::styled(
                " [F] ",
                Style::default()
                    .fg(Color::LightRed)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                "Retry failed - re-run {failed_count} failed item(s), same settings"
            )),
        ])));
    }
    if a_eligible {
        items.push(ListItem::new(Line::from(vec![
            Span::styled(
                " [A] ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("Run again - same queue, pick a different mode/model"),
        ])));
    }
    if h_eligible {
        let label = if multi {
            "Hardburn batch videos - opt-in, picks SRT per file"
        } else {
            "Hardburn - re-pick SRT and burn into video"
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled(
                " [H] ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(label),
        ])));
    }
    items.push(ListItem::new(Line::from(vec![
        Span::styled(
            " [T] ",
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        ),
        Span::raw("Translate - Translate the output SRT"),
    ])));
    items.push(ListItem::new(Line::from(vec![
        Span::styled(
            " [L] ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("Log - View full process log"),
    ])));
    items.push(ListItem::new(Line::from(vec![
        Span::styled(
            " [Q] ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::raw("Quit"),
    ])));

    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Options ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(list, chunks[2]);

    let mut help_parts: Vec<String> = vec!["R: Restart".to_string()];
    if h_eligible {
        help_parts.push("H: Hardburn".to_string());
    }
    if f_eligible {
        help_parts.push("F: Retry failed".to_string());
    }
    if a_eligible {
        help_parts.push("A: Run again".to_string());
    }
    help_parts.push("T: Translate".to_string());
    help_parts.push("L: Log".to_string());
    help_parts.push("Q: Quit".to_string());
    if multi {
        help_parts.push("Up/Down: Scroll results".to_string());
    }
    let help_text = help_parts.join(" | ");
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[3]);
}

/// Single-file results pane (the original Menu). Shows the SRT and any
/// hardcoded video for the most recent pipeline_result.
fn render_single_summary(app: &AppState) -> Paragraph<'static> {
    let summary = if let Some(ref result) = app.pipeline_result {
        let mut lines = Vec::new();
        if let Some(ref srt_path) = app.srt_display_path {
            lines.push(Line::from(vec![Span::styled(
                " SRT: ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(Line::from(vec![Span::styled(
                format!("   {srt_path}"),
                Style::default().fg(Color::White),
            )]));
        }
        if let Some(ref vid_path) = app.video_display_path {
            lines.push(Line::from(vec![Span::styled(
                " Video: ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )]));
            lines.push(Line::from(vec![Span::styled(
                format!("   {vid_path}"),
                Style::default().fg(Color::White),
            )]));
        }
        for warn in &result.warnings {
            lines.push(Line::styled(
                format!(" [WARN] {warn}"),
                Style::default().fg(Color::Yellow),
            ));
        }
        if lines.is_empty() {
            lines.push(Line::styled(
                " No output produced",
                Style::default().fg(Color::Red),
            ));
        }
        Text::from(lines)
    } else {
        Text::styled(" No results", Style::default().fg(Color::DarkGray))
    };
    Paragraph::new(summary).wrap(Wrap { trim: false }).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Results ")
            .border_style(Style::default().fg(Color::Green)),
    )
}

/// Batch results pane: lists every item the queue processed with its SRT
/// path, hardcoded video path (if any), and any error. This is the fix for
/// "we display the last file processed instead of every transcribed/hardcoded
/// output" -- the user sees each batch item individually now.
///
/// Scrollable: content past the pane height used to be silently clipped, so
/// a long queue's tail simply never appeared. `menu_scroll` (Up/Down/wheel)
/// pans the pane; the title shows position when there's more than fits.
fn render_batch_summary(app: &AppState, area: Rect) -> Paragraph<'static> {
    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in app.batch_results.iter().enumerate() {
        let badge = if row.error.is_some() {
            "[FAIL]"
        } else if row.video_path.is_some() {
            "[BURNED]"
        } else if row.srt_path.is_some() {
            "[SRT]"
        } else {
            "[?]"
        };
        let badge_color = match badge {
            "[FAIL]" => Color::Red,
            "[BURNED]" => Color::Magenta,
            "[SRT]" => Color::Cyan,
            _ => Color::DarkGray,
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:>2}. ", i + 1),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("{badge} "),
                Style::default()
                    .fg(badge_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(row.stem.clone(), Style::default().fg(Color::White)),
        ]));
        if let Some(ref err) = row.error {
            lines.push(Line::from(vec![Span::styled(
                format!("       error: {err}"),
                Style::default().fg(Color::Red),
            )]));
            continue;
        }
        if let Some(ref srt) = row.srt_path {
            lines.push(Line::from(vec![Span::styled(
                format!("       SRT:   {}", srt.display()),
                Style::default().fg(Color::Gray),
            )]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                "       SRT:   (none)",
                Style::default().fg(Color::DarkGray),
            )]));
        }
        if let Some(ref vid) = row.video_path {
            lines.push(Line::from(vec![Span::styled(
                format!("       Video: {}", vid.display()),
                Style::default().fg(Color::Gray),
            )]));
        }
    }
    if lines.is_empty() {
        lines.push(Line::styled(
            " No batch results",
            Style::default().fg(Color::DarkGray),
        ));
    }

    // Estimate RENDERED rows (post-wrap): Paragraph::scroll offsets in
    // wrapped-row space, so clamping against the logical line count would
    // strand the last rows out of reach whenever a long path wraps.
    let inner_w = (area.width.saturating_sub(2) as usize).max(1);
    let total_rows: usize = lines
        .iter()
        .map(|l| {
            let w = l.width();
            if w == 0 { 1 } else { w.div_ceil(inner_w) }
        })
        .sum();
    let viewport = area.height.saturating_sub(2) as usize;
    app.menu_summary_lines.set(total_rows);
    app.menu_viewport_height.set(viewport);
    let max_scroll = total_rows.saturating_sub(viewport);
    let scroll = app.menu_scroll.min(max_scroll);

    let title = if total_rows > viewport {
        format!(
            " Batch Results ({}) -- rows {}-{}/{} (Up/Down) ",
            app.batch_results.len(),
            scroll + 1,
            (scroll + viewport).min(total_rows),
            total_rows
        )
    } else {
        format!(" Batch Results ({}) ", app.batch_results.len())
    };

    Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .scroll((scroll as u16, 0))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(title)
                .border_style(Style::default().fg(Color::Green)),
        )
}

/// Whether `[H]` should be offered. Single-file: any video with a usable SRT.
/// Batch: at least one row that's a video with an SRT and no hardcoded video
/// yet. Uses a CHEAP existence + size check here -- this runs every render
/// frame, so a full SRT parse per item per frame would be wasteful. The
/// actual hardburn gate (SrtPicker commit + launch_hardburn_pipeline) does
/// the full `verify_srt`.
fn hardburn_eligible(app: &AppState) -> bool {
    if app.batch_results.len() >= 2 {
        return app
            .batch_results
            .iter()
            .any(|r| r.is_video && r.video_path.is_none() && srt_looks_usable(r.srt_path.as_deref()));
    }
    let media_is_video = app.media_file.as_ref().is_some_and(|m| m.is_video());
    let has_srt = app
        .pipeline_result
        .as_ref()
        .and_then(|r| r.srt_file.as_deref())
        .is_some_and(|p| srt_looks_usable(Some(p)));
    media_is_video && has_srt
}

/// Filesystem-only sanity check: does the file exist and have any bytes?
/// Cheap enough for per-frame use; not authoritative.
fn srt_looks_usable(path: Option<&std::path::Path>) -> bool {
    let p = match path {
        Some(p) => p,
        None => return false,
    };
    std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false)
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    match key.code {
        KeyCode::Char('r') | KeyCode::Char('R') => {
            reset_for_new_file(app);
            ScreenAction::Navigate(Screen::Start)
        }
        KeyCode::Char('h') | KeyCode::Char('H') => trigger_hardburn(app),
        // Retry only the failed rows with the settings already selected.
        KeyCode::Char('f') | KeyCode::Char('F') => {
            if app.retry_failed_items() {
                ScreenAction::LaunchPipeline
            } else {
                ScreenAction::None
            }
        }
        // Re-queue the full original list and go pick new settings.
        KeyCode::Char('a') | KeyCode::Char('A') => {
            if app.requeue_all_items() {
                ScreenAction::Navigate(Screen::SubtitleSelection)
            } else {
                ScreenAction::None
            }
        }
        // Scroll the batch results pane.
        KeyCode::Up => {
            app.menu_scroll = app.menu_scroll.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            let max = app
                .menu_summary_lines
                .get()
                .saturating_sub(app.menu_viewport_height.get());
            if app.menu_scroll < max {
                app.menu_scroll += 1;
            }
            ScreenAction::None
        }
        KeyCode::PageUp => {
            let page = app.menu_viewport_height.get().max(1);
            app.menu_scroll = app.menu_scroll.saturating_sub(page);
            ScreenAction::None
        }
        KeyCode::PageDown => {
            let page = app.menu_viewport_height.get().max(1);
            let max = app
                .menu_summary_lines
                .get()
                .saturating_sub(app.menu_viewport_height.get());
            app.menu_scroll = (app.menu_scroll + page).min(max);
            ScreenAction::None
        }
        KeyCode::Char('t') | KeyCode::Char('T') => {
            app.mode_list_index = 0; // reset language selection
            ScreenAction::Navigate(Screen::TranslateMenu)
        }
        KeyCode::Char('l') | KeyCode::Char('L') => {
            app.log_scroll = 0;
            ScreenAction::Navigate(Screen::LogViewer)
        }
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => ScreenAction::Quit,
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, _row: u16, _col: u16) -> ScreenAction {
    // Click-to-select on the dynamic options list is fragile (the [H] row
    // appears/disappears, shifting indices). Keep clicks as a no-op for now;
    // the keys (R/H/T/L/Q) are documented in the help line.
    ScreenAction::None
}

/// `[H]`: opt-in hardburn. Single-file: re-enter SrtPicker against the
/// existing media (so the user can swap the SRT). Batch: build the pending
/// hardburn queue, jump to the SrtPicker for item 1.
fn trigger_hardburn(app: &mut AppState) -> ScreenAction {
    if !hardburn_eligible(app) {
        return ScreenAction::None;
    }
    if app.batch_results.len() >= 2 {
        if !app.start_pending_hardburn() {
            app.push_log(
                "[WARN] No batch items eligible for hardburn (need video + verified SRT)"
                    .to_string(),
            );
            return ScreenAction::None;
        }
        return ScreenAction::Navigate(Screen::SrtPicker);
    }
    // Single-file: re-pick SRT for the file currently in `media_file`.
    if let Some(media) = app.media_file.clone() {
        app.populate_available_srts(&media);
        return ScreenAction::Navigate(Screen::SrtPicker);
    }
    ScreenAction::None
}

fn reset_for_new_file(app: &mut AppState) {
    app.input_path.clear();
    app.input_cursor = 0;
    app.input_is_valid = false;
    app.input_is_directory = false;
    app.media_file = None;
    app.input_files.clear();
    app.batch_index = 0;
    app.input_mode = None;
    app.paste_queue_draft.clear();
    app.selected_mode = None;
    app.selected_model = None;
    app.selected_style = None;
    app.pipeline_result = None;
    app.srt_display_path = None;
    app.video_display_path = None;
    app.english_master_display = None;
    app.available_srts.clear();
    app.srt_picker_dirs.clear();
    app.srt_picker_index = 0;
    app.srt_for_hardburn = None;
    app.batch_results.clear();
    app.pending_hardburn = None;
    app.menu_scroll = 0;
    app.original_queue.clear();
    // The finished run's manifest becomes the "last run" the Start screen's
    // [R] can redo -- last_run.json already holds it on disk.
    if let Some(m) = app.run_manifest.take() {
        app.saved_manifest = Some(m);
    }
    app.current_phase.clear();
    app.current_progress = 0.0;
    app.progress_message.clear();
    app.mode_list_index = 0;
    app.model_list_index = 0;
    app.style_list_index = 0;
    app.log_lines.clear();
    app.log_scroll = 0;
}
