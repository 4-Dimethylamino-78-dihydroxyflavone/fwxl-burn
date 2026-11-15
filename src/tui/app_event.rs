use crate::pipeline::PipelineEvent;

/// Events flowing through the TUI system.
#[derive(Debug)]
pub enum AppEvent {
    /// Keyboard/terminal input (handled by crossterm directly).
    /// Pipeline events from background transcription/hardburn.
    Pipeline(PipelineEvent),
    /// A tick for periodic UI refresh.
    Tick,
}
