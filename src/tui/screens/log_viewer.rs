use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3), // Title
        Constraint::Min(1),   // Log
        Constraint::Length(2), // Help
    ])
    .split(area);

    let line_count = app.log_lines.len();
    let title = Paragraph::new(format!(" Log Viewer ({line_count} lines) "))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(title, chunks[0]);

    let log_height = chunks[1].height.saturating_sub(2) as usize;
    app.log_viewport_height.set(log_height);
    let max_scroll = line_count.saturating_sub(log_height);
    let effective_scroll = app.log_scroll.min(max_scroll);
    let skip = if effective_scroll == 0 {
        line_count.saturating_sub(log_height)
    } else {
        line_count.saturating_sub(log_height + effective_scroll)
    };

    let items: Vec<ListItem> = app
        .log_lines
        .iter()
        .skip(skip)
        .take(log_height)
        .map(|line| {
            let style = if line.starts_with("[ERROR]") {
                Style::default().fg(Color::Red)
            } else if line.starts_with("[WARN]") {
                Style::default().fg(Color::Yellow)
            } else if line.starts_with("[DONE]") {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(line.as_str()).style(style)
        })
        .collect();

    let log = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Full Log ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(log, chunks[1]);

    let help = Paragraph::new("Up/Down/Scroll: Navigate | Home/End: Jump | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    match key.code {
        KeyCode::Up => {
            app.scroll_log_up();
            ScreenAction::None
        }
        KeyCode::Down => {
            app.scroll_log_down();
            ScreenAction::None
        }
        KeyCode::Home => {
            // Jump to top of log (oldest lines).
            app.log_scroll = app.max_log_scroll();
            ScreenAction::None
        }
        KeyCode::End => {
            app.log_scroll = 0;
            ScreenAction::None
        }
        KeyCode::Esc => {
            app.log_scroll = 0;
            if app.pipeline_running {
                ScreenAction::Navigate(Screen::Progress)
            } else {
                ScreenAction::Navigate(Screen::Menu)
            }
        }
        _ => ScreenAction::None,
    }
}
