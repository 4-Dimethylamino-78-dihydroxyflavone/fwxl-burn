use crate::pipeline::SubtitleStyle;
use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

const LIST_TOP: u16 = 6; // title(3) + srt info(3)

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3), // Title
        Constraint::Length(3), // SRT info
        Constraint::Min(1),   // Style list
        Constraint::Length(2), // Help
    ])
    .split(area);

    let title = Paragraph::new(" Select Hardcoding Style ")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Magenta)),
        );
    frame.render_widget(title, chunks[0]);

    // Show the SRT that's about to be hardburned. Prefers the user's
    // explicit pick from SrtPicker (`srt_for_hardburn`); falls back to the
    // most recent transcription's output. The path text is canonicalized at
    // pick time so render is filesystem-free.
    let srt_info = if let Some(ref chosen) = app.srt_for_hardburn {
        format!(" SRT to burn: {}", chosen.display())
    } else if let Some(ref result) = app.pipeline_result {
        if let Some(ref srt_path) = app.srt_display_path {
            format!(" SRT ({}): {}", result.lang_code, srt_path)
        } else {
            " No SRT produced".to_string()
        }
    } else {
        " Select a style or skip hardburn".to_string()
    };
    let info = Paragraph::new(srt_info)
        .style(Style::default().fg(Color::Green))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Hardburn Source ")
                .border_style(Style::default().fg(Color::Green)),
        );
    frame.render_widget(info, chunks[1]);

    let styles = SubtitleStyle::all();
    let mut items: Vec<ListItem> = styles
        .iter()
        .enumerate()
        .map(|(i, style)| {
            let is_selected = i == app.style_list_index;
            let key_color = match style {
                SubtitleStyle::Youtube => Color::Red,
                SubtitleStyle::YoutubeFlow => Color::LightRed,
                SubtitleStyle::Netflix => Color::Red,
                SubtitleStyle::Minimal => Color::Cyan,
                SubtitleStyle::Default => Color::White,
            };
            let line = Line::from(vec![
                Span::styled(
                    format!(" [{}] ", style.key()),
                    Style::default().fg(key_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    style.label().to_string(),
                    if is_selected {
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    },
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

    // Skip option
    let skip_selected = app.style_list_index == styles.len();
    items.push(
        ListItem::new(Line::from(vec![
            Span::styled(
                " [S] ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Skip hardburn - keep SRT only",
                Style::default().fg(if skip_selected { Color::White } else { Color::DarkGray }),
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
            .title(" Styles ")
            .border_style(Style::default().fg(Color::Magenta)),
    );
    frame.render_widget(list, chunks[2]);

    let help = Paragraph::new("Up/Down: Navigate | Enter/Click: Select | S: Skip | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[3]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    let styles = SubtitleStyle::all();
    let total = styles.len() + 1;

    match key.code {
        KeyCode::Up => {
            app.style_list_index = app.style_list_index.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            if app.style_list_index < total - 1 {
                app.style_list_index += 1;
            }
            ScreenAction::None
        }
        KeyCode::Enter => {
            if app.style_list_index < styles.len() {
                app.selected_style = Some(styles[app.style_list_index]);
                ScreenAction::LaunchHardburn
            } else {
                ScreenAction::Navigate(Screen::Menu)
            }
        }
        KeyCode::Char('s') | KeyCode::Char('S') => ScreenAction::Navigate(Screen::Menu),
        // Digit hotkeys come from `SubtitleStyle::key()`, so adding a style
        // to the enum is enough to give it a key here.
        KeyCode::Char(c) => match styles.iter().find(|s| s.key() == c) {
            Some(style) => {
                app.selected_style = Some(*style);
                ScreenAction::LaunchHardburn
            }
            None => ScreenAction::None,
        },
        // Esc backs up to the picker so the user can choose a different SRT
        // (or Skip) without rerunning the transcription pipeline.
        KeyCode::Esc => ScreenAction::Navigate(Screen::SrtPicker),
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    let styles = SubtitleStyle::all();
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < styles.len() {
            app.style_list_index = idx;
            app.selected_style = Some(styles[idx]);
            return ScreenAction::LaunchHardburn;
        } else if idx == styles.len() {
            return ScreenAction::Navigate(Screen::Menu);
        }
    }
    ScreenAction::None
}
