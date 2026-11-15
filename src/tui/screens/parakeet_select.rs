use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

const LIST_TOP: u16 = 3;

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .split(area);

    let title = Paragraph::new(" Select Parakeet Model [Experimental] ")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Red)),
        );
    frame.render_widget(title, chunks[0]);

    let items: Vec<ListItem> = app
        .config
        .prkt
        .models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let is_selected = i == app.parakeet_model_index;
            let broken_tag = if model.broken { " [BROKEN]" } else { "" };
            let style = if model.broken {
                Style::default().fg(Color::DarkGray)
            } else if is_selected {
                Style::default()
                    .fg(Color::White)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(format!(" {}{broken_tag}", model.display)).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Parakeet Models ")
            .border_style(Style::default().fg(Color::Red)),
    );
    frame.render_widget(list, chunks[1]);

    let help = Paragraph::new("Up/Down: Navigate | Enter/Click: Select & Start | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    let count = app.config.prkt.models.len();

    match key.code {
        KeyCode::Up => {
            app.parakeet_model_index = app.parakeet_model_index.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            if app.parakeet_model_index < count.saturating_sub(1) {
                app.parakeet_model_index += 1;
            }
            ScreenAction::None
        }
        KeyCode::Enter => confirm_parakeet(app),
        KeyCode::Esc => ScreenAction::Navigate(Screen::SubtitleSelection),
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < app.config.prkt.models.len() {
            app.parakeet_model_index = idx;
            return confirm_parakeet(app);
        }
    }
    ScreenAction::None
}

fn confirm_parakeet(app: &mut AppState) -> ScreenAction {
    if let Some(model) = app.config.prkt.models.get(app.parakeet_model_index) {
        if model.broken {
            app.push_log(format!(
                "[WARN] Model {} is marked as broken. Select another.",
                model.name
            ));
            return ScreenAction::None;
        }
        app.selected_model = Some(model.name.clone());
    }
    ScreenAction::LaunchPipeline
}
