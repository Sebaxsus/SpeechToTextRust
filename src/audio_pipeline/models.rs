use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobMetadata {
    pub job_id: String,
    pub audio_path: String,
    pub transcript_path: String,
    pub checkpoint_path: String,
    pub status: JobStatus,
    pub created_at: String,
    /// Marca si Fase 2/3 (Whisper + persistencia) terminó bien, independientemente de si Fase 4
    /// (embeddings) todavía está en curso o falló — permite que un cliente sepa si ya tiene
    /// sentido pedir `GET /api/jobs/{job_id}/transcript` sin esperar a `status == Completed`.
    /// `#[serde(default)]`: un `job.json` escrito antes de agregar este campo debe seguir
    /// deserializando (default `false`) en vez de romper `load_job`.
    #[serde(default)]
    pub transcript_ready: bool,
}

pub struct AudioChunk {
    pub index: usize,
    pub start_sec: f32,
    pub end_sec: f32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub chunk: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    /// Media de `ln(token_probability())` sobre todos los tokens del chunk — confianza de
    /// Whisper, poblada en `WhisperRunner::transcribe_chunk` (ver CLAUDE.local.md).
    pub avg_logprob: f32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Checkpoint {
    pub last_chunk: usize,
    pub processed_seconds: f32,
}
