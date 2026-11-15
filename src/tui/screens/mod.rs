pub mod start;
pub mod subtitle_selection;
pub mod model_select;
pub mod omni_select;
pub mod parakeet_select;
pub mod srt_picker;
pub mod style_select;
pub mod translate_menu;
pub mod progress;
pub mod log_viewer;
pub mod menu;

use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyEvent, MouseEvent, MouseEventKind, MouseButton};
use ratatui::prelude::*;

/// Render the current screen.
pub fn render(frame: &mut Frame, app: &AppState) {
    match app.current_screen {
        Screen::Start => start::render(frame, app),
        Screen::SubtitleSelection => subtitle_selection::render(frame, app),
        Screen::ModelSelect => model_select::render(frame, app),
        Screen::ParakeetSelect => parakeet_select::render(frame, app),
        Screen::OmniSelect => omni_select::render(frame, app),
        Screen::SrtPicker => srt_picker::render(frame, app),
        Screen::StyleSelect => style_select::render(frame, app),
        Screen::TranslateMenu => translate_menu::render(frame, app),
        Screen::Progress => progress::render(frame, app),
        Screen::LogViewer => log_viewer::render(frame, app),
        Screen::Menu => menu::render(frame, app),
    }
}

/// Handle a key event for the current screen.
pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    match app.current_screen {
        Screen::Start => start::handle_key(app, key),
        Screen::SubtitleSelection => subtitle_selection::handle_key(app, key),
        Screen::ModelSelect => model_select::handle_key(app, key),
        Screen::ParakeetSelect => parakeet_select::handle_key(app, key),
        Screen::OmniSelect => omni_select::handle_key(app, key),
        Screen::SrtPicker => srt_picker::handle_key(app, key),
        Screen::StyleSelect => style_select::handle_key(app, key),
        Screen::TranslateMenu => translate_menu::handle_key(app, key),
        Screen::Progress => progress::handle_key(app, key),
        Screen::LogViewer => log_viewer::handle_key(app, key),
        Screen::Menu => menu::handle_key(app, key),
    }
}

/// Handle a mouse event. Updates cursor tracking and delegates to screens.
pub fn handle_mouse(app: &mut AppState, mouse: MouseEvent) -> ScreenAction {
    // Always track cursor position for hover effects
    match mouse.kind {
        MouseEventKind::Moved | MouseEventKind::Down(_) | MouseEventKind::Drag(_) => {
            app.mouse_row = mouse.row;
            app.mouse_col = mouse.column;
        }
        _ => {}
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            match app.current_screen {
                Screen::SubtitleSelection => subtitle_selection::handle_click(app, mouse.row, mouse.column),
                Screen::ModelSelect => model_select::handle_click(app, mouse.row, mouse.column),
                Screen::ParakeetSelect => parakeet_select::handle_click(app, mouse.row, mouse.column),
                Screen::OmniSelect => omni_select::handle_click(app, mouse.row, mouse.column),
                Screen::SrtPicker => srt_picker::handle_click(app, mouse.row, mouse.column),
                Screen::StyleSelect => style_select::handle_click(app, mouse.row, mouse.column),
                Screen::TranslateMenu => translate_menu::handle_click(app, mouse.row, mouse.column),
                Screen::Menu => menu::handle_click(app, mouse.row, mouse.column),
                _ => ScreenAction::None,
            }
        }
        MouseEventKind::ScrollUp => {
            match app.current_screen {
                // Wheel up on a log pane reveals OLDER entries (scroll toward top).
                // log_scroll counts rows above the bottom, so we INCREMENT it here.
                Screen::Progress | Screen::LogViewer => {
                    app.scroll_log_up();
                    ScreenAction::None
                }
                Screen::SubtitleSelection => {
                    if app.mode_list_index > 0 {
                        app.mode_list_index -= 1;
                    }
                    ScreenAction::None
                }
                Screen::ModelSelect => {
                    if app.model_list_index > 0 {
                        app.model_list_index -= 1;
                    }
                    ScreenAction::None
                }
                Screen::SrtPicker => {
                    if app.srt_picker_index > 0 {
                        app.srt_picker_index -= 1;
                    }
                    ScreenAction::None
                }
                Screen::Menu => {
                    app.menu_scroll = app.menu_scroll.saturating_sub(1);
                    ScreenAction::None
                }
                _ => ScreenAction::None,
            }
        }
        MouseEventKind::ScrollDown => {
            match app.current_screen {
                // Wheel down reveals NEWER entries (back toward bottom).
                Screen::Progress | Screen::LogViewer => {
                    app.scroll_log_down();
                    ScreenAction::None
                }
                Screen::SubtitleSelection => {
                    let max = crate::pipeline::TranscriptionMode::all().len().saturating_sub(1);
                    if app.mode_list_index < max {
                        app.mode_list_index += 1;
                    }
                    ScreenAction::None
                }
                Screen::ModelSelect => {
                    let max = app.available_models.len().saturating_sub(1);
                    if app.model_list_index < max {
                        app.model_list_index += 1;
                    }
                    ScreenAction::None
                }
                Screen::SrtPicker => {
                    // +1 here: list contains every SRT plus the trailing
                    // "Skip" pseudo-row, both reachable with the wheel.
                    let max = app.available_srts.len(); // = len + 1 - 1
                    if app.srt_picker_index < max {
                        app.srt_picker_index += 1;
                    }
                    ScreenAction::None
                }
                Screen::Menu => {
                    let max = app
                        .menu_summary_lines
                        .get()
                        .saturating_sub(app.menu_viewport_height.get());
                    if app.menu_scroll < max {
                        app.menu_scroll += 1;
                    }
                    ScreenAction::None
                }
                _ => ScreenAction::None,
            }
        }
        _ => ScreenAction::None,
    }
}
