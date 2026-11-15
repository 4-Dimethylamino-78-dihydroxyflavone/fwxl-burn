use crate::tui::app::{AppState, Screen};
use crate::tui::ScreenAction;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, List, ListItem, Paragraph};

pub fn render(frame: &mut Frame, app: &AppState) {
    let area = frame.area();

    let chunks = Layout::vertical([
        Constraint::Length(4), // Current file + phase
        Constraint::Length(3), // Progress bar
        Constraint::Length(3), // Progress message
        Constraint::Min(1),   // Live log
        Constraint::Length(2), // Help
    ])
    .split(area);

    // Current file + phase. The file line answers "what is it working on
    // right now?" -- before, the filename only ever appeared in the log.
    let (phase_fg, border_fg) = if app.pipeline_running {
        (Color::Yellow, Color::Yellow)
    } else {
        (Color::Green, Color::Green)
    };
    let file_line = match app.current_file_indicator() {
        Some(indicator) => Line::from(vec![
            Span::styled("Now: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                indicator,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]),
        None => Line::from(""),
    };
    let phase_text = if app.pipeline_running {
        app.current_phase.clone()
    } else {
        "Pipeline complete".to_string()
    };
    let phase_line = Line::styled(
        phase_text,
        Style::default().fg(phase_fg).add_modifier(Modifier::BOLD),
    );
    let block_title = if app.pipeline_running {
        " Now transcribing "
    } else {
        " Phase "
    };
    let phase = Paragraph::new(vec![file_line, phase_line])
        .alignment(Alignment::Center)
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(block_title)
                .border_style(Style::default().fg(border_fg)),
        );
    frame.render_widget(phase, chunks[0]);

    // Progress bar -- pip-style: a green hash fill over a dim dash track,
    // with the percentage OUTSIDE the bar. The old Gauge painted its label
    // on top of the fill/track boundary, where the numbers had almost no
    // contrast against the blue fill.
    let pct = app.current_progress.clamp(0.0, 100.0);
    let label = format!(" {pct:5.1}% ");
    let inner_w = chunks[1].width.saturating_sub(2) as usize;
    let bar_w = inner_w.saturating_sub(label.chars().count() + 2); // [ + ]
    let filled = (((bar_w as f32) * pct / 100.0).round() as usize).min(bar_w);
    let bar_color = if pct >= 100.0 {
        Color::LightGreen
    } else {
        Color::Green
    };
    let bar_line = Line::from(vec![
        Span::styled("[", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "#".repeat(filled),
            Style::default().fg(bar_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "-".repeat(bar_w.saturating_sub(filled)),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("]", Style::default().fg(Color::DarkGray)),
        Span::styled(
            label,
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
    ]);
    let progress = Paragraph::new(bar_line).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Progress ")
            .border_style(Style::default().fg(Color::Green)),
    );
    frame.render_widget(progress, chunks[1]);

    // Progress message
    let msg = Paragraph::new(app.progress_message.as_str())
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Status ")
                .border_style(Style::default().fg(Color::Gray)),
        );
    frame.render_widget(msg, chunks[2]);

    // Live log (auto-scroll to bottom unless user scrolled)
    let log_height = chunks[3].height.saturating_sub(2) as usize;
    // Record viewport height so key handlers can cap scroll past the top.
    app.log_viewport_height.set(log_height);
    let total_lines = app.log_lines.len();
    // Cap the effective scroll so we never "scroll past" the top of the buffer.
    let max_scroll = total_lines.saturating_sub(log_height);
    let effective_scroll = app.log_scroll.min(max_scroll);
    let skip = if effective_scroll == 0 {
        total_lines.saturating_sub(log_height)
    } else {
        total_lines.saturating_sub(log_height + effective_scroll)
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
            } else if line.starts_with("[FALLBACK]") {
                Style::default().fg(Color::Magenta)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(line.as_str()).style(style)
        })
        .collect();

    let log = List::new(items).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" Output ")
            .border_style(Style::default().fg(Color::Cyan)),
    );
    frame.render_widget(log, chunks[3]);

    // Help
    let help_text = if app.pipeline_running {
        "Ctrl+C: Cancel | Scroll: Browse log"
    } else {
        "Press any key to continue..."
    };
    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(help, chunks[4]);
}

pub fn handle_key(app: &mut AppState, key: KeyEvent) -> ScreenAction {
    if app.pipeline_running {
        // While running, only allow log scrolling
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
            _ => ScreenAction::None,
        }
    } else {
        // Pipeline done, any key continues
        // The drain_pipeline_events already navigated to StyleSelect or Menu
        ScreenAction::None
    }
}
