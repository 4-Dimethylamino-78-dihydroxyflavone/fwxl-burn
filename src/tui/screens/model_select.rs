use crate::tools::whisper_xxl;
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

    let title = Paragraph::new(" Select Whisper Model ")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Yellow)),
        );
    frame.render_widget(title, chunks[0]);

    let items: Vec<ListItem> = app
        .available_models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let is_selected = i == app.model_list_index;
            let is_default = model == &app.config.whisper.default_model;
            let suffix = if is_default { " (default)" } else { "" };
            let style = if is_selected {
                Style::default()
                    .fg(Color::White)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(format!(" {model}{suffix}")).style(style)
        })
        .collect();

    if items.is_empty() {
        let msg = Paragraph::new(" No models found in _models/ directory. Using default.")
            .style(Style::default().fg(Color::Yellow))
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Models ")
                    .border_style(Style::default().fg(Color::Yellow)),
            );
        frame.render_widget(msg, chunks[1]);
    } else {
        let list = List::new(items).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Models ")
                .border_style(Style::default().fg(Color::Yellow)),
        );
        frame.render_widget(list, chunks[1]);
    }

    let help = Paragraph::new("Up/Down/Scroll: Navigate | Enter/Click: Select & Start | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    ensure_models_loaded(app);

    match key.code {
        KeyCode::Up => {
            app.model_list_index = app.model_list_index.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            let max = app.available_models.len().saturating_sub(1);
            if app.model_list_index < max {
                app.model_list_index += 1;
            }
            ScreenAction::None
        }
        KeyCode::Enter => confirm_model(app),
        KeyCode::Esc => ScreenAction::Navigate(Screen::SubtitleSelection),
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    ensure_models_loaded(app);
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < app.available_models.len() {
            app.model_list_index = idx;
            return confirm_model(app);
        }
    }
    ScreenAction::None
}

fn confirm_model(app: &mut AppState) -> ScreenAction {
    if let Some(model) = app.available_models.get(app.model_list_index) {
        app.selected_model = Some(model.clone());
    } else {
        app.selected_model = Some(app.config.whisper.default_model.clone());
    }
    // Launch the pipeline
    ScreenAction::LaunchPipeline
}

fn ensure_models_loaded(app: &mut AppState) {
    if app.available_models.is_empty() {
        app.available_models = whisper_xxl::list_available_models(&app.config);
        if app.available_models.is_empty() {
            app.available_models
                .push(app.config.whisper.default_model.clone());
        }
        app.model_list_index = app
            .available_models
            .iter()
            .position(|m| m == &app.config.whisper.default_model)
            .unwrap_or(0);
    }
}
