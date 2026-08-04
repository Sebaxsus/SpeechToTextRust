use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// Estado del resumen por audio (map-reduce vía `rag::summary::generate_summary`), disparado
/// como tarea independiente después de `JobStatus::Completed` (ver
/// `handlers::audio_handler::lanzar_generacion_resumen`). Bookkeeping chico en `job.json` — el
/// contenido en sí vive en el sidecar `summary.txt` (`JobMetadata::summary_path()`), mismo
/// criterio que separa `metrics.jsonl` de `job.json`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum SummaryStatus {
    #[default]
    NotStarted,
    Generating,
    Ready,
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
    /// Timestamps de fase (epoch-segundos, mismo formato que `created_at`), agregados 2026-07-31
    /// para poder medir "tiempo de ejecución entre tareas" a nivel macro: duración Whisper =
    /// `transcript_ready_at - processing_started_at`; duración embeddings = `completed_at -
    /// transcript_ready_at`. `#[serde(default)]` para que un `job.json` viejo sin estos campos
    /// siga deserializando (mismo criterio que `transcript_ready`).
    #[serde(default)]
    pub processing_started_at: Option<String>,
    #[serde(default)]
    pub transcript_ready_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    /// Ver `SummaryStatus`. `#[serde(default)]` para que un `job.json` viejo (de antes de esta
    /// feature) siga deserializando con `NotStarted`.
    #[serde(default)]
    pub summary_status: SummaryStatus,
    /// Nombre de archivo original del multipart (`campo.file_name()`), guardado como dato plano
    /// para mostrar un "título" en el cliente web — **nunca** se usa para construir una ruta de
    /// disco (el archivo real siempre vive en `audio_path`, derivado del `job_id`), así que no
    /// reintroduce el riesgo de path traversal que Fase 1 ya evitó para este mismo valor.
    /// `None` si el cliente no mandó ningún `filename` en el multipart. `#[serde(default)]` para
    /// que un `job.json` viejo siga deserializando.
    #[serde(default)]
    pub original_filename: Option<String>,
    /// Duración total del audio en segundos, medida con `ffprobe` sobre el archivo ya escrito a
    /// disco, justo antes de responder `202` en `recibir_y_procesar_audio` — es una llamada corta
    /// (mismo binario que ya usa el fallback de `decoder.rs`), así que no vale la pena moverla a
    /// background solo por eso. `None` si `ffprobe` falla (no bloquea ni falla el upload por
    /// esto) o para `job.json` viejos sin este campo.
    #[serde(default)]
    pub duration_seconds: Option<f32>,
    /// Título representativo elegido por el usuario (campo de texto opcional `title` del
    /// multipart de `POST /api/upload-audio`), guardado tal cual — nunca se usa para construir
    /// una ruta de disco, mismo criterio que `original_filename`. `None` si el cliente no mandó
    /// el campo; el fallback (`title ?? original_filename ?? job_id`) es responsabilidad de quien
    /// consume `JobSummary` (el cliente web), no se resuelve acá para no perder la distinción
    /// entre "el usuario no puso título" y "el usuario puso un título vacío". `#[serde(default)]`
    /// para que un `job.json` viejo siga deserializando.
    #[serde(default)]
    pub title: Option<String>,
}

impl JobMetadata {
    /// Deriva la ruta de `metrics.jsonl` a partir de `transcript_path` en vez de agregar un
    /// campo nuevo a `JobMetadata`/migrar `job.json` — evita duplicar la derivación entre
    /// `pipeline.rs` (que escribe el archivo) y `jobs_handler.rs` (que lo lee para el endpoint
    /// `GET /api/jobs/{job_id}/metrics`).
    pub fn metrics_path(&self) -> PathBuf {
        std::path::Path::new(&self.transcript_path).with_file_name("metrics.jsonl")
    }

    /// Deriva la ruta de `summary.txt` — mismo criterio que `metrics_path()`. Texto plano UTF-8
    /// (no JSON): es un blob único escrito una vez cuando `summary_status` pasa a `Ready`, no un
    /// log incremental como `transcript.jsonl`/`metrics.jsonl`.
    pub fn summary_path(&self) -> PathBuf {
        std::path::Path::new(&self.transcript_path).with_file_name("summary.txt")
    }
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

/// Una línea de `metrics.jsonl` (diagnóstico del pipeline, separado de `transcript.jsonl`/el
/// payload de Qdrant a propósito — ver CLAUDE.local.md, sección del logger de métricas). Persiste
/// tiempos por etapa y señales de calidad de whisper-rs que hoy se calculan pero se descartaban.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkMetrics {
    pub chunk: usize,
    pub start: f32,
    pub end: f32,
    pub decode_ms: u64,
    pub whisper_ms: u64,
    pub persist_ms: u64,
    pub total_ms: u64,
    pub text_len: usize,
    pub avg_logprob: f32,
    /// `exp(avg_logprob)` — confianza aproximada en el rango 0-1, más legible que el log-espacio.
    pub score: f32,
    pub no_speech_prob: f32,
    /// Ver `whisper_runner::ChunkTranscription::entropy` — misma fórmula que usa whisper.cpp
    /// internamente, recalculada sobre el chunk ya finalizado.
    pub entropy: f32,
    pub segment_count: usize,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Checkpoint {
    pub last_chunk: usize,
    pub processed_seconds: f32,
}
