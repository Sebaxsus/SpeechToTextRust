// TODO: implement whisper-rs FullParams-based transcription (see CLAUDE.local.md: Fase 2 —
// Whisper — must run inside tokio::task::spawn_blocking, never tokio::spawn).
pub struct WhisperRunner;

impl WhisperRunner {
    pub fn new(_model_path: &str) -> anyhow::Result<Self> {
        todo!("WhisperRunner::new is not implemented yet")
    }

    pub fn transcribe_chunk(&self, _samples: &[f32]) -> anyhow::Result<String> {
        todo!("WhisperRunner::transcribe_chunk is not implemented yet")
    }
}
