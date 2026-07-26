use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
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
}

pub struct AudioChunk {
    pub index: usize,
    pub start_sec: f32,
    pub end_sec: f32,
    pub samples: Vec<f32>,
}

#[derive(Serialize)]
pub struct TranscriptEntry {
    pub chunk: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Checkpoint {
    pub last_chunk: usize,
    pub processed_seconds: f32,
}
