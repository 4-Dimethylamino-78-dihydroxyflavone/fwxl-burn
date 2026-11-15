use crate::media::{iter_media_files, MediaFile};
use crate::tui::app::{AppState, InputMode, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph, Wrap};
use std::path::PathBuf;

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    // Top-level routing: the user first picks F/B/P (unless drag-drop already
    // set the mode), then the matching input screen renders.
    match app.input_mode {
        None => {
            render_mode_picker(frame, app, area);
            return;
        }
        Some(InputMode::PasteQueue) if app.media_file.is_none() => {
            render_paste_queue(frame, app, area);
            return;
        }
        _ => {}
    }

    let chunks = Layout::vertical([
        Constraint::Length(3), // Banner
        Constraint::Length(3), // Input
        Constraint::Length(5), // Status
        Constraint::Min(1),   // Help
    ])
    .split(area);

    // Banner
    let banner = Paragraph::new(concat!(
        " FWXL-BURN v",
        env!("CARGO_PKG_VERSION"),
        "  --  Transcription & Subtitle Hardcoding "
    ))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(banner, chunks[0]);

    // Input field with visible cursor
    let input_style = if app.input_is_valid {
        Style::default().fg(Color::Green)
    } else if app.input_path.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::Red)
    };

    let border_style = if app.input_is_valid {
        Style::default().fg(Color::Green)
    } else if app.input_path.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::Red)
    };

    // Build the input line with a visible cursor at the cursor position
    let display_line = if app.input_path.is_empty() {
        Line::styled(
            "Type/paste a path, or drag a file onto this window...",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        let cursor = app.input_cursor.min(app.input_path.len());
        let before = &app.input_path[..cursor];
        let after = &app.input_path[cursor..];
        let cursor_char = if cursor < app.input_path.len() {
            // Cursor is on a character -- highlight it
            let c = app.input_path[cursor..].chars().next().unwrap();
            let char_len = c.len_utf8();
            let on = &app.input_path[cursor..cursor + char_len];
            let rest = &app.input_path[cursor + char_len..];
            return render_input_with_cursor(frame, chunks[1], before, on, rest, input_style, border_style);
        } else {
            // Cursor at end -- show block
            "\u{2588}"
        };

        Line::from(vec![
            Span::styled(before.to_string(), input_style),
            Span::styled(cursor_char, Style::default().fg(Color::White).bg(Color::Gray)),
            Span::styled(after.to_string(), input_style),
        ])
    };

    let input = Paragraph::new(display_line).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Input Path ")
            .border_style(border_style),
    );
    frame.render_widget(input, chunks[1]);

    // Status
    render_status(frame, app, chunks[2]);

    // Help
    let help_text = if app.media_file.is_some() {
        "Enter: Continue | Backspace: Clear | Ctrl+C: Quit"
    } else if app.input_is_valid {
        "Enter: Probe & continue | Arrows: Edit | Ctrl+C: Quit"
    } else {
        "Type/paste a path | Arrows: Move cursor | Home/End | Ctrl+C: Quit"
    };
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[3]);
}

fn render_input_with_cursor(
    frame: &mut Frame,
    area: Rect,
    before: &str,
    on_cursor: &str,
    after: &str,
    text_style: Style,
    border_style: Style,
) {
    let line = Line::from(vec![
        Span::styled(before.to_string(), text_style),
        Span::styled(on_cursor.to_string(), Style::default().fg(Color::Black).bg(Color::White)),
        Span::styled(after.to_string(), text_style),
    ]);
    let input = Paragraph::new(line).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Input Path ")
            .border_style(border_style),
    );
    frame.render_widget(input, area);
}

fn render_status(frame: &mut Frame, app: &AppState, area: Rect) {
    let status_text = if let Some(ref media) = app.media_file {
        let dur_str = media
            .duration
            .map(|d| {
                let secs = d.as_secs();
                if secs >= 3600 {
                    format!("{}h{}m{}s", secs / 3600, (secs % 3600) / 60, secs % 60)
                } else if secs >= 60 {
                    format!("{}m{}s", secs / 60, secs % 60)
                } else {
                    format!("{}s", secs)
                }
            })
            .unwrap_or_else(|| "?".to_string());
        let subs = media.embedded_subs.len();
        let sub_info = if subs > 0 {
            format!("{subs} embedded sub streams")
        } else {
            "no embedded subs".to_string()
        };
        format!(" {} | .{} | {} | {}", media.stem, media.extension, dur_str, sub_info)
    } else if app.input_is_valid {
        if app.input_is_directory {
            " Directory detected -- will use batch mode".to_string()
        } else {
            " File found -- press Enter to probe and continue".to_string()
        }
    } else if app.input_path.is_empty() {
        " Waiting for input...".to_string()
    } else {
        " Path not found -- check the path and try again".to_string()
    };

    let status_style = if app.media_file.is_some() {
        Style::default().fg(Color::Green)
    } else if app.input_is_valid {
        Style::default().fg(Color::Yellow)
    } else if app.input_path.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::Red)
    };

    let border_color = if app.media_file.is_some() {
        Color::Green
    } else if app.input_is_valid {
        Color::Yellow
    } else if app.input_path.is_empty() {
        Color::DarkGray
    } else {
        Color::Red
    };
    let status = Paragraph::new(status_text).style(status_style).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Status ")
            .border_style(Style::default().fg(border_color)),
    );
    frame.render_widget(status, area);
}

/// Mode picker shown on first entry to the Start screen (when no drag-drop
/// has pre-selected a mode). Keys F/B/P pick between single-file, batch
/// folder, and paste-queue flows -- unifying what used to be three implicit
/// code paths into one explicit user choice.
fn render_mode_picker(frame: &mut Frame, app: &AppState, area: Rect) {
    let chunks = Layout::vertical([
        Constraint::Length(3), // Banner
        Constraint::Min(1),    // Three cards
        Constraint::Length(2), // Help
    ])
    .split(area);

    let banner = Paragraph::new(" FWXL-BURN  --  Choose input mode ")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(banner, chunks[0]);

    let cards = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
    ])
    .split(chunks[1]);

    let card = |key: &str, title: &str, blurb: &str, color: Color| -> Paragraph<'static> {
        let mut lines = vec![
            Line::from(vec![Span::styled(
                format!(" [{key}] "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )]),
            Line::from(Span::styled(
                title.to_string(),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];
        // One Line per blurb line: a `\n` embedded inside a single Span is
        // NOT a line break to ratatui -- it rendered as the sentences mashed
        // together ("one file.Hardburn") overflowing the card border.
        for part in blurb.split('\n') {
            lines.push(Line::from(Span::styled(
                part.to_string(),
                Style::default().fg(Color::Gray),
            )));
        }
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true })
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(color)),
            )
    };

    frame.render_widget(
        card(
            "F",
            "Single File",
            "Transcribe one file.\nHardburn offered for video.",
            Color::Green,
        ),
        cards[0],
    );
    frame.render_widget(
        card(
            "B",
            "Batch Folder",
            "Process every media file\nin a folder, in order.",
            Color::Yellow,
        ),
        cards[1],
    );
    frame.render_widget(
        card(
            "P",
            "Paste Queue",
            "Paste many paths, type RUN.\nSame settings for all.",
            Color::Magenta,
        ),
        cards[2],
    );

    let help_text = match app.saved_manifest.as_ref() {
        Some(m) => {
            let unfinished = m.failed_count() + m.pending_count();
            let redo_note = if unfinished > 0 {
                format!("R: Redo last run ({unfinished}/{} unfinished)", m.items.len())
            } else {
                format!("R: Redo last run ({} item(s), all done)", m.items.len())
            };
            format!("F: File  |  B: Batch folder  |  P: Paste queue  |  {redo_note}  |  Esc: Quit")
        }
        None => "F: File  |  B: Batch folder  |  P: Paste queue  |  Esc: Quit".to_string(),
    };
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}

fn handle_mode_picker_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    match key.code {
        KeyCode::Char('f') | KeyCode::Char('F') => {
            app.input_mode = Some(InputMode::File);
            ScreenAction::None
        }
        KeyCode::Char('b') | KeyCode::Char('B') => {
            app.input_mode = Some(InputMode::BatchFolder);
            ScreenAction::None
        }
        KeyCode::Char('p') | KeyCode::Char('P') => {
            app.input_mode = Some(InputMode::PasteQueue);
            ScreenAction::None
        }
        // Redo last run: reload the persisted manifest's file list into a
        // fresh paste queue (failed items only when any failed, else all).
        // The user reviews the list, types RUN, and picks mode/model as
        // usual -- which is exactly what makes "same batch, different
        // settings" work without re-pasting anything.
        KeyCode::Char('r') | KeyCode::Char('R') => {
            if let Some(m) = app.saved_manifest.clone() {
                let paths = m.redo_paths();
                if paths.is_empty() {
                    app.push_log(
                        "[WARN] Last run's files no longer exist on disk".to_string(),
                    );
                } else {
                    let failed = m.failed_count();
                    let pending = m.pending_count();
                    let scope = if failed + pending > 0 {
                        format!("{failed} failed + {pending} never ran")
                    } else {
                        "all items".to_string()
                    };
                    app.push_log(format!(
                        "[REDO] Loaded {} item(s) from last run ({scope}) -- review, then type RUN",
                        paths.len()
                    ));
                    app.input_mode = Some(InputMode::PasteQueue);
                    app.input_files = paths;
                }
            }
            ScreenAction::None
        }
        KeyCode::Esc => ScreenAction::Quit,
        _ => ScreenAction::None,
    }
}

/// Paste-queue subscreen (Gallery-dl style: paste paths, type RUN to begin).
/// Listed entries accumulate in `app.input_files`; the current line being
/// typed lives in `app.paste_queue_draft`. Multi-path pastes (from clipboard)
/// add each path as its own entry; single-path pastes append to the draft so
/// the user can review before committing with Enter.
fn render_paste_queue(frame: &mut Frame, app: &AppState, area: Rect) {
    let chunks = Layout::vertical([
        Constraint::Length(3), // Banner
        Constraint::Min(1),    // Queue list
        Constraint::Length(3), // Draft input
        Constraint::Length(3), // Status
        Constraint::Length(2), // Help
    ])
    .split(area);

    let banner = Paragraph::new(" Paste Queue  --  type paths, RUN to begin ")
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Magenta)),
        );
    frame.render_widget(banner, chunks[0]);

    // Keep the NEWEST entries visible. A ratatui List clips at the pane
    // bottom, so a long paste session used to silently hide everything past
    // the first screenful -- the user pastes item 40 and sees no change.
    // Instead, drop the oldest rows behind a one-line "... N earlier ..."
    // indicator so the tail of the queue (what was just added) always shows.
    let inner_height = chunks[1].height.saturating_sub(2) as usize;
    let total = app.input_files.len();
    let hidden = if inner_height > 0 && total > inner_height {
        total - inner_height.saturating_sub(1).max(1)
    } else {
        0
    };
    let mut items: Vec<ListItem> = Vec::new();
    if hidden > 0 {
        items.push(ListItem::new(Line::from(Span::styled(
            format!("      ... {hidden} earlier item(s) above ..."),
            Style::default().fg(Color::DarkGray),
        ))));
    }
    items.extend(app.input_files.iter().enumerate().skip(hidden).map(|(i, p)| {
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        let dir = p
            .parent()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        ListItem::new(Line::from(vec![
            Span::styled(
                format!(" {: >2}. ", i + 1),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(name, Style::default().fg(Color::White)),
            Span::styled(format!("  [{dir}]"), Style::default().fg(Color::DarkGray)),
        ]))
    }));
    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(format!(" Queue ({}) ", app.input_files.len()))
            .border_style(Style::default().fg(Color::Magenta)),
    );
    frame.render_widget(list, chunks[1]);

    let draft_display = if app.paste_queue_draft.is_empty() {
        Line::styled(
            "Type or paste a path, then Enter to add",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Line::from(vec![
            Span::styled(app.paste_queue_draft.clone(), Style::default().fg(Color::Green)),
            Span::styled("\u{2588}", Style::default().fg(Color::White).bg(Color::Gray)),
        ])
    };
    let draft = Paragraph::new(draft_display).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Next entry ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(draft, chunks[2]);

    let status_text = if app.input_files.is_empty() {
        " Add at least one path before typing RUN".to_string()
    } else {
        format!(
            " {} item(s) queued -- Enter RUN (or press Enter on an empty line) to begin",
            app.input_files.len()
        )
    };
    let status_color = if app.input_files.is_empty() {
        Color::DarkGray
    } else {
        Color::Green
    };
    let status = Paragraph::new(status_text)
        .style(Style::default().fg(status_color))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Status ")
                .border_style(Style::default().fg(status_color)),
        );
    frame.render_widget(status, chunks[3]);

    let help = Paragraph::new(
        "Ctrl+V: paste clipboard  |  Enter: add  |  type RUN: begin  |  Del: remove  |  Esc: back",
    )
    .alignment(Alignment::Center)
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[4]);
}

fn handle_paste_queue_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    // Ctrl+V: read the system clipboard directly. Crossterm's bracketed-paste
    // events don't fire in legacy Windows conhost (which is what opens on
    // double-click), so we need an explicit path that works everywhere.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V'))
    {
        if let Some(text) = read_clipboard() {
            return handle_paste_queue_paste(app, &text);
        }
        app.push_log("[WARN] Clipboard is empty or inaccessible".to_string());
        return ScreenAction::None;
    }

    match key.code {
        KeyCode::Enter => {
            let draft_raw = app.paste_queue_draft.trim().to_string();
            let is_run_sentinel = draft_raw.eq_ignore_ascii_case("run");
            let empty_with_entries = draft_raw.is_empty() && !app.input_files.is_empty();

            if is_run_sentinel || empty_with_entries {
                app.paste_queue_draft.clear();
                return start_paste_queue_pipeline(app);
            }

            if !draft_raw.is_empty() {
                let path = PathBuf::from(draft_raw.trim_matches('"'));
                if path.exists() {
                    app.input_files.push(path);
                }
                // Silently ignore invalid paths -- the "Queue" counter
                // doesn't advance, which is the feedback. No error modal.
                app.paste_queue_draft.clear();
            }
            ScreenAction::None
        }
        KeyCode::Backspace => {
            if app.paste_queue_draft.is_empty() {
                // Convenience: Backspace on an empty draft pops the last queued entry.
                app.input_files.pop();
            } else {
                // Pop one UTF-8 char off the end.
                let last_boundary = app
                    .paste_queue_draft
                    .char_indices()
                    .next_back()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                app.paste_queue_draft.truncate(last_boundary);
            }
            ScreenAction::None
        }
        KeyCode::Delete => {
            app.input_files.pop();
            ScreenAction::None
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.paste_queue_draft.clear();
            ScreenAction::None
        }
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input_files.clear();
            app.paste_queue_draft.clear();
            ScreenAction::None
        }
        KeyCode::Char(c) => {
            app.paste_queue_draft.push(c);
            ScreenAction::None
        }
        KeyCode::Esc => {
            app.input_mode = None;
            app.input_files.clear();
            app.paste_queue_draft.clear();
            ScreenAction::None
        }
        _ => ScreenAction::None,
    }
}

fn handle_paste_queue_paste(app: &mut AppState, text: &str) -> ScreenAction {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return ScreenAction::None;
    }
    let paths = split_pasted_paths(cleaned);

    if paths.len() > 1 {
        // Clipboard contained multiple paths -- commit each that exists.
        for p in paths {
            let pb = PathBuf::from(p.trim().trim_matches('"'));
            if pb.exists() {
                app.input_files.push(pb);
            }
        }
    } else {
        // Single line -- append to the draft so the user sees what they pasted.
        let trimmed = cleaned.trim_matches('"');
        app.paste_queue_draft.push_str(trimmed);
    }
    ScreenAction::None
}

/// Finalize the paste queue: load item 0 as the active media_file and navigate
/// to SubtitleSelection. Subsequent items are advanced by `drain_pipeline_events`
/// after each pipeline completion.
fn start_paste_queue_pipeline(app: &mut AppState) -> ScreenAction {
    if app.input_files.is_empty() {
        return ScreenAction::None;
    }
    app.batch_index = 0;
    commit_queue_run(app);
    let first = app.input_files[0].clone();
    app.input_path = first.to_string_lossy().to_string();
    app.input_cursor = app.input_path.len();
    app.input_is_valid = true;
    app.input_is_directory = false;
    probe_input(app);
    if app.media_file.is_some() {
        ScreenAction::Navigate(Screen::SubtitleSelection)
    } else {
        ScreenAction::None
    }
}

/// Expand a batch-folder path into `input_files` (enumerating media files) and
/// proceed to SubtitleSelection with item 0 loaded. Subsequent items advance
/// through the same queue-runner as paste-queue.
fn start_batch_folder(app: &mut AppState) -> ScreenAction {
    let dir = PathBuf::from(&app.input_path);
    let files = match iter_media_files(&dir) {
        Ok(files) => files,
        Err(e) => {
            app.push_log(format!("[ERROR] Could not read folder: {e}"));
            return ScreenAction::None;
        }
    };
    if files.is_empty() {
        app.push_log(format!(
            "[ERROR] No media files found in folder: {}",
            dir.display()
        ));
        return ScreenAction::None;
    }
    app.input_files = files;
    app.batch_index = 0;
    commit_queue_run(app);
    let first = app.input_files[0].clone();
    app.input_path = first.to_string_lossy().to_string();
    app.input_cursor = app.input_path.len();
    app.input_is_valid = true;
    app.input_is_directory = false;
    probe_input(app);
    if app.media_file.is_some() {
        ScreenAction::Navigate(Screen::SubtitleSelection)
    } else {
        ScreenAction::None
    }
}

/// Record the committed queue: remember the full file list for `[A] Run
/// again`, and start a persistent manifest run (multi-item queues only) so
/// `last_run.json` tracks per-item outcomes from the very first file.
fn commit_queue_run(app: &mut AppState) {
    app.original_queue = app.input_files.clone();
    if app.input_files.len() > 1 {
        let manifest = crate::run_manifest::RunManifest::new(&app.input_files);
        manifest.save();
        app.run_manifest = Some(manifest);
    }
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    // Route based on top-level input mode before falling through to the
    // classic single-path input handler.
    match app.input_mode {
        None => return handle_mode_picker_key(app, key),
        Some(InputMode::PasteQueue) if app.media_file.is_none() => {
            return handle_paste_queue_key(app, key);
        }
        _ => {}
    }

    // Ctrl+V fallback for the single-file/batch-folder input (legacy conhost
    // doesn't emit bracketed-paste events).
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V'))
    {
        if let Some(text) = read_clipboard() {
            return handle_paste(app, &text);
        }
        app.push_log("[WARN] Clipboard is empty or inaccessible".to_string());
        return ScreenAction::None;
    }

    match key.code {
        KeyCode::Enter => {
            if app.media_file.is_some() {
                ScreenAction::Navigate(Screen::SubtitleSelection)
            } else if app.input_is_valid {
                // Batch folder: enumerate media files into input_files BEFORE
                // probing so the queue runner can iterate them. Without this,
                // whisper-xxl would receive a directory and do its own opaque
                // iteration -- we'd lose per-item progress.
                if matches!(app.input_mode, Some(InputMode::BatchFolder))
                    && app.input_is_directory
                {
                    return start_batch_folder(app);
                }
                probe_input(app);
                if app.media_file.is_some() {
                    ScreenAction::Navigate(Screen::SubtitleSelection)
                } else {
                    ScreenAction::None
                }
            } else {
                ScreenAction::None
            }
        }

        // -- Cursor movement --
        KeyCode::Left => {
            if app.input_cursor > 0 {
                // Move back one char (handle UTF-8)
                let prev = app.input_path[..app.input_cursor]
                    .char_indices()
                    .next_back()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                app.input_cursor = prev;
            }
            ScreenAction::None
        }
        KeyCode::Right => {
            if app.input_cursor < app.input_path.len() {
                let next = app.input_path[app.input_cursor..]
                    .chars()
                    .next()
                    .map(|c| app.input_cursor + c.len_utf8())
                    .unwrap_or(app.input_path.len());
                app.input_cursor = next;
            }
            ScreenAction::None
        }
        KeyCode::Home => {
            app.input_cursor = 0;
            ScreenAction::None
        }
        KeyCode::End => {
            app.input_cursor = app.input_path.len();
            ScreenAction::None
        }

        // Ctrl+A: select all / clear
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.input_path.clear();
            app.input_cursor = 0;
            app.input_is_valid = false;
            app.media_file = None;
            ScreenAction::None
        }

        // Regular character: insert at cursor
        KeyCode::Char(c) => {
            app.input_path.insert(app.input_cursor, c);
            app.input_cursor += c.len_utf8();
            validate_path(app);
            app.media_file = None;
            ScreenAction::None
        }

        // Backspace: delete char before cursor
        KeyCode::Backspace => {
            if app.media_file.is_some() {
                // Clear everything
                app.input_path.clear();
                app.input_cursor = 0;
                app.input_is_valid = false;
                app.media_file = None;
                app.input_files.clear();
            } else if app.input_cursor > 0 {
                // Find the previous char boundary
                let prev = app.input_path[..app.input_cursor]
                    .char_indices()
                    .next_back()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                app.input_path.drain(prev..app.input_cursor);
                app.input_cursor = prev;
                validate_path(app);
            }
            ScreenAction::None
        }

        // Delete: delete char at cursor
        KeyCode::Delete => {
            if app.input_cursor < app.input_path.len() {
                let c = app.input_path[app.input_cursor..].chars().next().unwrap();
                app.input_path.drain(app.input_cursor..app.input_cursor + c.len_utf8());
                validate_path(app);
            } else {
                // At end: clear all
                app.input_path.clear();
                app.input_cursor = 0;
                app.input_is_valid = false;
                app.media_file = None;
                app.input_files.clear();
            }
            ScreenAction::None
        }

        KeyCode::Esc => {
            // Back out to the mode picker. The user can hit Esc again there to quit.
            app.input_mode = None;
            app.input_path.clear();
            app.input_cursor = 0;
            app.input_is_valid = false;
            app.input_is_directory = false;
            app.media_file = None;
            app.input_files.clear();
            ScreenAction::None
        }
        _ => ScreenAction::None,
    }
}

/// Handle a paste event (bracketed paste).
pub fn handle_paste(app: &mut AppState, text: &str) -> ScreenAction {
    if matches!(app.input_mode, Some(InputMode::PasteQueue)) && app.media_file.is_none() {
        return handle_paste_queue_paste(app, text);
    }
    let cleaned = text.trim().trim_matches('"').trim().to_string();
    let paths = split_pasted_paths(&cleaned);

    if paths.len() > 1 {
        let valid: Vec<std::path::PathBuf> = paths
            .iter()
            .map(|p| std::path::PathBuf::from(p.trim().trim_matches('"')))
            .filter(|p| p.exists())
            .collect();
        if valid.is_empty() {
            app.input_path = cleaned;
            app.input_cursor = app.input_path.len();
            validate_path(app);
        } else {
            // Multi-path paste in a single-file/batch-folder context: upgrade
            // transparently to the paste-queue screen so the user sees the list
            // and can confirm with RUN. Clears any half-typed single path.
            app.input_mode = Some(InputMode::PasteQueue);
            app.input_files = valid;
            app.input_path.clear();
            app.input_cursor = 0;
            app.media_file = None;
        }
    } else {
        // Single path: replace input entirely
        app.input_path = cleaned;
        app.input_cursor = app.input_path.len();
        validate_path(app);
        app.media_file = None;
    }

    ScreenAction::None
}

fn validate_path(app: &mut AppState) {
    // Only strip surrounding quotes, NEVER trim whitespace --
    // paths like "D:\SD Card\..." have internal spaces that must be preserved.
    // Trailing spaces are harmless; trim() was killing them mid-typing.
    let needs_strip = app.input_path.starts_with('"') || app.input_path.ends_with('"');
    if needs_strip {
        let stripped = app.input_path.trim_matches('"').to_string();
        let was_cursor = app.input_cursor;
        app.input_path = stripped;
        app.input_cursor = was_cursor.min(app.input_path.len());
    }
    let path = std::path::Path::new(&app.input_path);
    app.input_is_valid = path.exists();
    app.input_is_directory = path.is_dir();
}

fn probe_input(app: &mut AppState) {
    let path = std::path::Path::new(&app.input_path);
    match MediaFile::from_path(path) {
        Ok(mut media) => {
            let config = app.config.clone();
            // block_in_place allows blocking inside the tokio async runtime
            // without the "cannot start a runtime from within a runtime" panic
            let probe_result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(crate::probe::probe_media_file(&config, &mut media))
            });
            match probe_result {
                Ok(status) => {
                    match status.reason() {
                        Some(reason) => app.push_log(format!(
                            "[ERR ] {} is unreadable: {reason}",
                            media.stem
                        )),
                        None => app.push_log(format!(
                            "Probed: {} ({} subs, duration {:?})",
                            media.stem,
                            media.embedded_subs.len(),
                            media.duration
                        )),
                    }
                    app.media_file = Some(media);
                }
                Err(e) => {
                    app.push_log(format!("[WARN] Probe failed: {e} -- continuing anyway"));
                    app.media_file = Some(media);
                }
            }
        }
        Err(e) => {
            app.push_log(format!("[ERROR] Failed to load: {e}"));
        }
    }
}

/// Split a pasted string that may contain multiple Windows paths.
///
/// Three splitting rules, in order:
///   1. Newlines (CR/LF) -- the common case for `dir /b` output or Notepad lists.
///   2. Drive-letter boundaries (`X:\...Y:\...`) -- clipboard pastes from
///      Explorer's "Copy as path" for multiple selections.
///   3. Quoted-path whitespace boundaries -- `"C:\a.mkv" "D:\b.mkv"`.
///
/// Results are trimmed, de-quoted, and de-duped. Empty entries are dropped.
fn split_pasted_paths(text: &str) -> Vec<String> {
    // Phase 1: if there are any newlines, that's the dominant delimiter. Split
    // on them first, then hand each line to the drive-letter splitter in case
    // a single line still contains multiple concatenated paths.
    let has_newline = text.contains('\n') || text.contains('\r');
    if has_newline {
        let mut out = Vec::new();
        for line in text.split(|c| c == '\n' || c == '\r') {
            // Pre-strip surrounding quotes BEFORE the drive-boundary scan.
            // Otherwise a leading `"` at position 0 makes prev-char at i=1 look
            // like a boundary trigger (prev != \ / :), and `"D:\x` splits into
            // [`"`, `D:\x`]. Stripping first keeps `D:\x` whole.
            let stripped = line.trim().trim_matches('"').trim();
            for p in split_by_drive_boundary(stripped) {
                let trimmed = p.trim().trim_matches('"').trim().to_string();
                if !trimmed.is_empty() {
                    out.push(trimmed);
                }
            }
        }
        return dedupe_preserve_order(out);
    }

    // Phase 2: no newlines -- look for `"..."  "..."` quoted-with-whitespace.
    // This only triggers when the string begins with `"` so we don't mangle
    // unquoted paths that happen to contain spaces.
    if text.trim_start().starts_with('"') {
        let quoted: Vec<String> = text
            .split('"')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if quoted.len() >= 2 {
            return dedupe_preserve_order(quoted);
        }
    }

    // Phase 2b (Linux/macOS): terminals there hand over dropped files in
    // shell form -- either single-quoted (`'/a/my file.mkv' '/a/b.mkv'`) or
    // with backslash-escaped spaces (`/a/my\ file.mkv /a/b.mkv`). Neither
    // applies on Windows, where `\` is the path separator and `'` is an
    // ordinary filename character.
    if !cfg!(windows) {
        if let Some(paths) = split_shell_style(text) {
            return dedupe_preserve_order(paths);
        }
    }

    // Phase 3: fall back to the original drive-letter boundary splitter.
    let raw = split_by_drive_boundary(text);
    let cleaned: Vec<String> = raw
        .into_iter()
        .map(|s| s.trim().trim_matches('"').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    dedupe_preserve_order(cleaned)
}

/// Split shell-style pasted paths: whitespace separates paths unless it is
/// inside single quotes or escaped with a backslash.
///
/// Returns `None` when the text uses neither convention, so that a plain
/// unquoted path containing spaces (`/a/my file.mkv`, as copied from a file
/// manager) is left whole for the caller instead of being cut at each space.
fn split_shell_style(text: &str) -> Option<Vec<String>> {
    let text = text.trim();
    if !text.starts_with('\'') && !text.contains("\\ ") {
        return None;
    }
    let mut paths = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '\'' => in_quote = !in_quote,
            // Outside quotes a backslash makes the next character literal.
            '\\' if !in_quote => current.extend(chars.next()),
            c if c.is_whitespace() && !in_quote => {
                if !current.is_empty() {
                    paths.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        paths.push(current);
    }
    Some(paths)
}

/// Find boundaries like `...X:\...Y:\...` and split there.
fn split_by_drive_boundary(text: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i > 0
            && i + 2 < chars.len()
            && chars[i].is_ascii_alphabetic()
            && chars[i + 1] == ':'
            && (chars[i + 2] == '\\' || chars[i + 2] == '/')
        {
            let prev = chars[i - 1];
            if prev != '\\' && prev != '/' && prev != ':' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    paths.push(trimmed);
                }
                current = String::new();
            }
        }
        current.push(chars[i]);
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        paths.push(trimmed);
    }
    paths
}

fn dedupe_preserve_order(xs: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(xs.len());
    for x in xs {
        if seen.insert(x.clone()) {
            out.push(x);
        }
    }
    out
}

/// Read text from the system clipboard. Returns None on empty clipboard or
/// error (clipboard driver unavailable, non-text content, etc.).
///
/// Fallback path for legacy Windows conhost, which does not fire crossterm
/// bracketed-paste events even with `EnableBracketedPaste`. Without this,
/// Ctrl+V in a double-clicked exe just types a literal 'v'.
fn read_clipboard() -> Option<String> {
    match arboard::Clipboard::new() {
        Ok(mut cb) => cb.get_text().ok().filter(|s| !s.is_empty()),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_single_path() {
        let paths = split_pasted_paths(r"D:\Videos\movie.mkv");
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_split_two_paths() {
        let paths = split_pasted_paths(r"D:\Videos\movie1.mkvD:\Videos\movie2.mkv");
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_split_paths_with_spaces() {
        let paths = split_pasted_paths(r"D:\My Videos\Star Trek S03E03.mkv C:\Users\test\file.mp4");
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_split_newline_separated() {
        let text = "D:\\a.mkv\r\nD:\\b.mkv\nD:\\c.mkv";
        let paths = split_pasted_paths(text);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], r"D:\a.mkv");
        assert_eq!(paths[2], r"D:\c.mkv");
    }

    #[test]
    fn test_split_quoted_whitespace() {
        let text = r#""D:\My Videos\a.mkv" "D:\My Videos\b.mkv""#;
        let paths = split_pasted_paths(text);
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("a.mkv"));
        assert!(paths[1].ends_with("b.mkv"));
    }

    #[test]
    fn test_split_dedupes() {
        let text = "D:\\a.mkv\nD:\\a.mkv\nD:\\b.mkv";
        let paths = split_pasted_paths(text);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_split_newline_with_trailing_blank() {
        let text = "D:\\a.mkv\n\n";
        let paths = split_pasted_paths(text);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_split_quoted_per_line() {
        // The real-world format: "copy as path" from Explorer multi-select, or
        // a list pasted from a text editor with one quoted path per line.
        let text = "\"D:\\Music\\a.m4a\"\r\n\"D:\\Music\\b.m4a\"\r\n\"D:\\Music\\c.m4a\"";
        let paths = split_pasted_paths(text);
        assert_eq!(paths.len(), 3, "got {paths:?}");
        assert_eq!(paths[0], r"D:\Music\a.m4a");
        assert_eq!(paths[2], r"D:\Music\c.m4a");
    }

    // `split_shell_style` is called directly so these run on every OS, not
    // only where `split_pasted_paths` routes to it.
    #[test]
    fn test_shell_style_single_quoted() {
        let paths = split_shell_style("'/media/My Videos/a.mkv' '/media/b.mkv'").unwrap();
        assert_eq!(paths, ["/media/My Videos/a.mkv", "/media/b.mkv"]);
    }

    #[test]
    fn test_shell_style_escaped_spaces() {
        let paths = split_shell_style(r"/media/My\ Videos/a.mkv /media/b.mkv").unwrap();
        assert_eq!(paths, ["/media/My Videos/a.mkv", "/media/b.mkv"]);
    }

    #[test]
    fn test_shell_style_leaves_plain_paths_alone() {
        // No quoting or escaping in play: must not be cut at the space.
        assert_eq!(split_shell_style("/media/My Videos/a.mkv"), None);
    }
}
