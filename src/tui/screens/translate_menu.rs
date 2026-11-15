use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

const LANGUAGES: &[(&str, &str)] = &[
    ("th", "Thai"),
    ("ja", "Japanese"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("vi", "Vietnamese"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
];

const LIST_TOP: u16 = 6;

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3), // Title
        Constraint::Length(3), // English master info
        Constraint::Min(1),   // Language list
        Constraint::Length(2), // Help
    ])
    .split(area);

    let title = Paragraph::new(" Translate - Select Target Language ")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Blue)),
        );
    frame.render_widget(title, chunks[0]);

    // English master info -- uses cached value from `set_pipeline_result` so we
    // don't rescan the filesystem every render frame.
    let master_info = if app.media_file.is_some() {
        match &app.english_master_display {
            Some(abs) => format!(" English master: {abs}"),
            None => " No English SRT found. Transcribe first.".to_string(),
        }
    } else {
        " No media loaded".to_string()
    };
    let master = Paragraph::new(master_info).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Source SRT ")
            .border_style(Style::default().fg(Color::Blue)),
    );
    frame.render_widget(master, chunks[1]);

    let items: Vec<ListItem> = LANGUAGES
        .iter()
        .enumerate()
        .map(|(i, (code, name))| {
            let is_selected = i == app.mode_list_index; // reuse mode_list_index
            let style = if is_selected {
                Style::default()
                    .fg(Color::White)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(format!(" [{code}] {name}")).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Languages ")
            .border_style(Style::default().fg(Color::Blue)),
    );
    frame.render_widget(list, chunks[2]);

    let help = Paragraph::new("Up/Down: Navigate | Enter/Click: Translate | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[3]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    let count = LANGUAGES.len();

    match key.code {
        KeyCode::Up => {
            app.mode_list_index = app.mode_list_index.saturating_sub(1);
            ScreenAction::None
        }
        KeyCode::Down => {
            if app.mode_list_index < count - 1 {
                app.mode_list_index += 1;
            }
            ScreenAction::None
        }
        KeyCode::Enter => {
            // TODO: launch translation pipeline
            app.push_log(format!(
                "Translation to {} not yet wired",
                LANGUAGES[app.mode_list_index].1
            ));
            ScreenAction::Navigate(Screen::Menu)
        }
        KeyCode::Esc => ScreenAction::Navigate(Screen::SubtitleSelection),
        _ => ScreenAction::None,
    }
}

pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < LANGUAGES.len() {
            app.mode_list_index = idx;
            // Same as Enter for now
            app.push_log(format!(
                "Translation to {} not yet wired",
                LANGUAGES[idx].1
            ));
            return ScreenAction::Navigate(Screen::Menu);
        }
    }
    ScreenAction::None
}
