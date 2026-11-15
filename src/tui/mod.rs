pub mod app;
pub mod app_event;
pub mod screens;
pub mod widgets;

use crate::config::AppConfig;
use crate::error::AppError;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
    },
};
use ratatui::prelude::*;
use std::io;
use std::path::PathBuf;

use self::app::{AppState, Screen};

/// Actions that screen handlers can return.
pub enum ScreenAction {
    None,
    Quit,
    Navigate(Screen),
    LaunchPipeline,
    LaunchHardburn,
}

/// Initialize the terminal and run the TUI event loop.
pub async fn run(config: AppConfig, input_files: Vec<PathBuf>) -> Result<(), AppError> {
    enable_raw_mode().map_err(|e| AppError::Tui(format!("Failed to enable raw mode: {e}")))?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )
    .map_err(|e| AppError::Tui(format!("Failed to enter alternate screen: {e}")))?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)
        .map_err(|e| AppError::Tui(format!("Failed to create terminal: {e}")))?;

    let result = run_event_loop(&mut terminal, config, input_files).await;

    // Restore terminal
    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste
    )
    .ok();
    terminal.show_cursor().ok();

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    config: AppConfig,
    input_files: Vec<PathBuf>,
) -> Result<(), AppError> {
    let mut app = AppState::new(config, input_files);
    let mut last_title = String::new();

    loop {
        // Keep the console title in sync with the current file / progress.
        // Only re-set when it changes; the whisper child can no longer
        // clobber it (it runs in its own hidden console).
        let title = app.window_title();
        if title != last_title {
            let _ = execute!(terminal.backend_mut(), SetTitle(title.as_str()));
            last_title = title;
        }

        terminal
            .draw(|frame| {
                screens::render(frame, &app);
            })
            .map_err(|e| AppError::Tui(format!("Draw error: {e}")))?;

        // Poll for events with 50ms timeout (allows pipeline event draining)
        if crossterm::event::poll(std::time::Duration::from_millis(50))
            .map_err(|e| AppError::Tui(format!("Event poll error: {e}")))?
        {
            let event = crossterm::event::read()
                .map_err(|e| AppError::Tui(format!("Event read error: {e}")))?;

            let action = match event {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        ScreenAction::None
                    } else if key.modifiers.contains(KeyModifiers::CONTROL)
                        && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('q'))
                    {
                        app.cancel_pipeline();
                        ScreenAction::Quit
                    } else {
                        screens::handle_key(&mut app, key)
                    }
                }
                // Crossterm paste event (bracketed paste mode)
                Event::Paste(text) => {
                    if app.current_screen == Screen::Start {
                        screens::start::handle_paste(&mut app, &text)
                    } else {
                        ScreenAction::None
                    }
                }
                Event::Mouse(mouse) => screens::handle_mouse(&mut app, mouse),
                Event::Resize(_, _) => ScreenAction::None,
                _ => ScreenAction::None,
            };

            match action {
                ScreenAction::None => {}
                ScreenAction::Quit => return Ok(()),
                ScreenAction::Navigate(screen) => app.current_screen = screen,
                ScreenAction::LaunchPipeline => {
                    app.launch_transcription_pipeline();
                    // The launch can decline to run: skip/recover sweeps may
                    // finish the whole queue (it navigates to Menu itself),
                    // and setup errors log + stay put. Only a real launch
                    // should land on the Progress screen.
                    if app.pipeline_running {
                        app.current_screen = Screen::Progress;
                    }
                }
                ScreenAction::LaunchHardburn => {
                    app.launch_hardburn_pipeline();
                    app.current_screen = Screen::Progress;
                }
            }
        }

        app.drain_pipeline_events();
    }
}
