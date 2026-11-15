use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph, Wrap};
use std::path::Path;

/// Layout: title(3) + scanned dirs(5) -> list starts at row 8.
const LIST_TOP: u16 = 8;

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3),  // Title
        Constraint::Length(5),  // Scanned locations (SRCDIR + SRT_DIR)
        Constraint::Min(1),     // SRT list (+ Skip option)
        Constraint::Length(3),  // Live verification of highlighted SRT
        Constraint::Length(2),  // Help
    ])
    .split(area);

    // Title
    let title_text = if let Some(ref m) = app.media_file {
        format!(" Pick SRT to Hardburn -- {} ", m.stem)
    } else {
        " Pick SRT to Hardburn ".to_string()
    };
    let title = Paragraph::new(title_text)
        .alignment(Alignment::Center)
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(title, chunks[0]);

    // Scanned locations panel: makes "where are we looking?" explicit so the
    // user understands why a particular SRT shows up (and where to drop new
    // ones). The first dir is always SRCDIR (beside the file), the second
    // is SRT_DIR (the configured output dir, often `media.dir/source`).
    let mut dir_lines: Vec<Line> = Vec::new();
    for (i, d) in app.srt_picker_dirs.iter().enumerate() {
        let label = origin_label(i);
        dir_lines.push(Line::from(vec![
            Span::styled(
                format!(" {label} "),
                Style::default()
                    .fg(label_color(i))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(d.display().to_string(), Style::default().fg(Color::Gray)),
        ]));
    }
    if dir_lines.is_empty() {
        dir_lines.push(Line::styled(
            " (no media loaded -- nothing to scan)",
            Style::default().fg(Color::DarkGray),
        ));
    }
    let dirs_panel = Paragraph::new(dir_lines).wrap(Wrap { trim: false }).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Scanned Locations ")
            .border_style(Style::default().fg(Color::Yellow)),
    );
    frame.render_widget(dirs_panel, chunks[1]);

    // SRT list
    let pending_srt = app
        .pipeline_result
        .as_ref()
        .and_then(|r| r.srt_file.clone());

    let mut items: Vec<ListItem> = app
        .available_srts
        .iter()
        .enumerate()
        .map(|(i, cand)| {
            let is_selected = i == app.srt_picker_index;
            let is_pending = pending_srt
                .as_ref()
                .is_some_and(|p| same_path(p, &cand.path));
            let label = origin_label(cand.origin_dir_index);
            let label_col = label_color(cand.origin_dir_index);
            let name = cand
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string();
            let recommended = if is_pending {
                "  (just transcribed -- default)"
            } else {
                ""
            };

            let name_style = if is_selected {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else if is_pending {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Gray)
            };

            let line = Line::from(vec![
                Span::styled(
                    format!(" {label} "),
                    Style::default()
                        .fg(label_col)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(name, name_style),
                Span::styled(
                    recommended.to_string(),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]);
            let item = ListItem::new(line);
            if is_selected {
                item.style(Style::default().bg(Color::DarkGray))
            } else {
                item
            }
        })
        .collect();

    // Skip option last in the list -- selectable via [S] or by navigating to
    // it. Maps to `srt_for_hardburn = None` and routes to Menu.
    let skip_idx = app.available_srts.len();
    let skip_selected = app.srt_picker_index == skip_idx;
    items.push(
        ListItem::new(Line::from(vec![
            Span::styled(
                " [S] ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Skip hardburn -- keep SRT only",
                Style::default().fg(if skip_selected {
                    Color::White
                } else {
                    Color::DarkGray
                }),
            ),
        ]))
        .style(if skip_selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        }),
    );

    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(format!(
                " Available Subtitles ({}) ",
                app.available_srts.len()
            ))
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(list, chunks[2]);

    // Verification panel for the currently-highlighted SRT. Calls verify_srt
    // on every move; that's full file read + parse but only on user motion,
    // not per frame, and SRTs are tiny.
    let verify_line = if app.srt_picker_index < app.available_srts.len() {
        let path = &app.available_srts[app.srt_picker_index].path;
        let v = crate::srt::verify_srt(path);
        let (col, prefix) = if v.valid {
            (Color::Green, "[OK]")
        } else {
            (Color::Red, "[INVALID]")
        };
        Line::from(vec![
            Span::styled(
                format!(" {prefix} "),
                Style::default().fg(col).add_modifier(Modifier::BOLD),
            ),
            Span::styled(v.message, Style::default().fg(col)),
        ])
    } else {
        Line::styled(
            " No SRT selected -- skipping hardburn keeps the transcription only",
            Style::default().fg(Color::DarkGray),
        )
    };
    let verify_panel = Paragraph::new(verify_line)
        .wrap(Wrap { trim: false })
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Verification ")
                .border_style(Style::default().fg(Color::Green)),
        );
    frame.render_widget(verify_panel, chunks[3]);

    let help = Paragraph::new(
        "Up/Down/Scroll: Navigate  |  Enter: Use SRT  |  S: Skip  |  Esc: Back",
    )
    .alignment(Alignment::Center)
    .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[4]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    let total = app.available_srts.len() + 1; // +1 for Skip

    match key.code {
        KeyCode::Up => {
            app.srt_picker_index = app.srt_picker_index.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            if app.srt_picker_index + 1 < total {
                app.srt_picker_index += 1;
            }
            ScreenAction::None
        }
        KeyCode::Home => {
            app.srt_picker_index = 0;
            ScreenAction::None
        }
        KeyCode::End => {
            app.srt_picker_index = total.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Enter => commit_selection(app),
        KeyCode::Char('s') | KeyCode::Char('S') => {
            app.srt_for_hardburn = None;
            ScreenAction::Navigate(Screen::Menu)
        }
        KeyCode::Esc => ScreenAction::Navigate(Screen::Menu),
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    let total = app.available_srts.len() + 1;
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < total {
            app.srt_picker_index = idx;
            return commit_selection(app);
        }
    }
    ScreenAction::None
}

/// Apply the currently-highlighted choice. Verifies the SRT one more time
/// (the verification panel is informational; this is the actual gate) so
/// even a click-through can't slip an invalid SRT into the hardburn flow.
fn commit_selection(app: &mut AppState) -> ScreenAction {
    if app.srt_picker_index >= app.available_srts.len() {
        // Skip -- no hardburn.
        app.srt_for_hardburn = None;
        return ScreenAction::Navigate(Screen::Menu);
    }
    let chosen = app.available_srts[app.srt_picker_index].path.clone();
    let v = crate::srt::verify_srt(&chosen);
    if !v.valid {
        app.push_log(format!(
            "[ERROR] Cannot use this SRT for hardburn: {} ({})",
            chosen.display(),
            v.message
        ));
        return ScreenAction::None;
    }
    app.push_log(format!(
        "[OK] Selected SRT for hardburn: {} ({})",
        chosen.display(),
        v.message
    ));
    app.srt_for_hardburn = Some(chosen);
    ScreenAction::Navigate(Screen::StyleSelect)
}

fn origin_label(i: usize) -> &'static str {
    match i {
        0 => "[SRCDIR]",
        1 => "[SRT_DIR]",
        _ => "[OTHER]",
    }
}

fn label_color(i: usize) -> Color {
    match i {
        0 => Color::Cyan,
        1 => Color::Magenta,
        _ => Color::Gray,
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}
