use crate::pipeline::TranscriptionMode;
use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

/// Layout constant: title=3, list area starts at row 3
const LIST_TOP: u16 = 3;

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(3), // Title
        Constraint::Min(1),   // Mode list
        Constraint::Length(2), // Help
    ])
    .split(area);

    // Title with media name
    let title_text = if let Some(ref media) = app.media_file {
        format!(" Select Mode - {} ", media.stem)
    } else {
        " Select Transcription Mode ".to_string()
    };
    let title = Paragraph::new(title_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(title, chunks[0]);

    // Mode list with CMD-matching colors
    let modes = TranscriptionMode::all();
    let items: Vec<ListItem> = modes
        .iter()
        .enumerate()
        .map(|(i, mode)| {
            let is_selected = i == app.mode_list_index;
            let key_color = mode_key_color(mode);
            let line = Line::from(vec![
                Span::styled(
                    format!(" [{}] ", mode.key()),
                    Style::default().fg(key_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    mode.label().to_string(),
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

    let list = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Modes ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(list, chunks[1]);

    let help = Paragraph::new("Up/Down/Scroll: Navigate | Enter/Click: Select | Shortcut keys: N,C,W,E,Q,I,T,Y,U,P,O,X | Esc: Back")
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[2]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    let modes = TranscriptionMode::all();
    let count = modes.len();

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
        KeyCode::Enter => select_mode(app, modes[app.mode_list_index]),
        KeyCode::Char(c) => {
            let upper = c.to_ascii_uppercase();
            if let Some(mode) = modes.iter().find(|m| m.key() == upper) {
                select_mode(app, *mode)
            } else {
                ScreenAction::None
            }
        }
        KeyCode::Esc => ScreenAction::Navigate(Screen::Start),
        _ => ScreenAction::None,
    }
}

/// Handle mouse click on the mode list.
pub fn handle_click(app: &mut AppState, row: u16, _col: u16) -> ScreenAction {
    let modes = TranscriptionMode::all();
    // List area starts after title (3 rows) + 1 for border
    let list_row = row.checked_sub(LIST_TOP + 1);
    if let Some(idx) = list_row {
        let idx = idx as usize;
        if idx < modes.len() {
            app.mode_list_index = idx;
            return select_mode(app, modes[idx]);
        }
    }
    ScreenAction::None
}

fn select_mode(app: &mut AppState, mode: TranscriptionMode) -> ScreenAction {
    app.selected_mode = Some(mode);
    if mode.uses_whisper() || mode == TranscriptionMode::Diagnostic {
        ScreenAction::Navigate(Screen::ModelSelect)
    } else if mode.uses_parakeet() {
        ScreenAction::Navigate(Screen::ParakeetSelect)
    } else if mode.uses_omnilingual() {
        ScreenAction::Navigate(Screen::OmniSelect)
    } else if mode == TranscriptionMode::Translate {
        ScreenAction::Navigate(Screen::TranslateMenu)
    } else if mode == TranscriptionMode::Native {
        // Native: just LaunchPipeline. The pre-flight check we used to do
        // here ("does media have a text subtitle stream?") was checking
        // item 0 only -- which silently blocked the user in paste-queue /
        // batch-folder mode whenever item 0 happened to have no text subs
        // even though items 1..N did. The pipeline itself surfaces a clear
        // [ERROR] per item if extraction has nothing to work with, and
        // drain_pipeline_events already handles per-item errors by
        // advancing the queue. Pre-flighting here was strictly a regression.
        //
        // Single-file flow with no text subs: pipeline emits Error, lands
        // on Menu with the reason in the log + batch_results.
        // Queue flow: each item is tried independently.
        ScreenAction::LaunchPipeline
    } else {
        // Custom: route through the SrtPicker against on-disk SRTs. The
        // user dropped a subtitle file directly, so there's nothing to
        // transcribe -- just let them confirm which SRT to burn.
        if let Some(media) = app.media_file.clone() {
            app.populate_available_srts(&media);
        }
        ScreenAction::Navigate(Screen::SrtPicker)
    }
}

fn mode_key_color(mode: &TranscriptionMode) -> Color {
    match mode {
        TranscriptionMode::Native => Color::Green,
        TranscriptionMode::Custom => Color::Green,
        TranscriptionMode::TestWhisper => Color::Yellow,
        TranscriptionMode::Extended => Color::Yellow,
        TranscriptionMode::Quality => Color::Magenta,
        TranscriptionMode::Inclusive => Color::LightMagenta,
        TranscriptionMode::Translate => Color::Blue,
        TranscriptionMode::Chunked => Color::Blue,
        TranscriptionMode::Unfiltered => Color::Cyan,
        TranscriptionMode::Parakeet => Color::Red,
        TranscriptionMode::ParakeetDiarize => Color::Red,
        TranscriptionMode::Omnilingual => Color::Magenta,
        TranscriptionMode::Diagnostic => Color::LightRed,
    }
}
