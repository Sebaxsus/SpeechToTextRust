//! Endpoints REST de solo lectura sobre jobs (`GET /api/jobs`, `GET /api/jobs/{job_id}`,
//! `GET /api/jobs/{job_id}/transcript`, `GET /api/jobs/{job_id}/metrics`) — wrappers finos sobre
//! `audio_pipeline::job`, pensados para que un cliente web no tenga que hablar el protocolo MCP
//! completo solo para leer estado. Montados bajo el mismo middleware de bearer token que `/mcp`
//! (ver `router.rs`), porque exponen el mismo contenido transcrito sensible.

use std::io::{BufRead, BufReader};

use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::Serialize;

use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::job::{list_jobs, load_job};
use crate::audio_pipeline::models::{ChunkMetrics, JobMetadata, JobStatus, TranscriptEntry};

/// Subconjunto público de `JobMetadata` — nunca expone `audio_path`/`transcript_path`/
/// `checkpoint_path` (rutas de disco del servidor, estructura interna que un cliente remoto no
/// necesita).
#[derive(Debug, Serialize)]
pub struct JobSummary {
    pub job_id: String,
    pub status: JobStatus,
    pub transcript_ready: bool,
    pub created_at: String,
    /// Timestamps de fase (epoch-segundos, `None` si esa fase todavía no ocurrió) — permiten
    /// calcular "tiempo de ejecución entre tareas": duración Whisper =
    /// `transcript_ready_at - processing_started_at`; duración embeddings = `completed_at -
    /// transcript_ready_at` (ver CLAUDE.local.md: logger de métricas).
    pub processing_started_at: Option<String>,
    pub transcript_ready_at: Option<String>,
    pub completed_at: Option<String>,
}

impl From<&JobMetadata> for JobSummary {
    fn from(metadata: &JobMetadata) -> Self {
        Self {
            job_id: metadata.job_id.clone(),
            status: metadata.status,
            transcript_ready: metadata.transcript_ready,
            created_at: metadata.created_at.clone(),
            processing_started_at: metadata.processing_started_at.clone(),
            transcript_ready_at: metadata.transcript_ready_at.clone(),
            completed_at: metadata.completed_at.clone(),
        }
    }
}

/// `GET /api/jobs` — listado de todos los jobs, equivalente REST de la tool MCP `list_audios`.
/// Reusa `audio_pipeline::job::list_jobs` (mismo `read_dir` + `load_job` que la tool MCP).
pub async fn listar_jobs_handler() -> impl IntoResponse {
    match list_jobs() {
        Ok(jobs) => {
            let resumen: Vec<JobSummary> = jobs.iter().map(JobSummary::from).collect();
            Json(resumen).into_response()
        }
        Err(e) => {
            tracing::error!("Error listando jobs: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    #[serde(flatten)]
    pub summary: JobSummary,
    /// Último chunk procesado y segundos ya transcritos, leídos de `checkpoint.json` —
    /// `(0, 0.0)` si el job todavía no generó ningún checkpoint (recién creado, o en cola detrás
    /// de otro job pesado).
    pub last_chunk: usize,
    pub processed_seconds: f32,
}

/// `GET /api/jobs/{job_id}` — status + progreso de un job puntual. `404` si `job_id` no es un
/// UUID válido o no existe (mismo criterio que `POST /api/jobs/{job_id}/resume`); `200` siempre
/// que el job exista, sea cual sea su `status`.
pub async fn obtener_job_handler(Path(job_id): Path<String>) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    // Sin checkpoint todavía (job recién creado o aún en cola) no es un error — CheckpointManager
    // ya devuelve el default (0, 0.0) en ese caso (ver checkpoint.rs).
    let checkpoint =
        match CheckpointManager::new(&metadata.checkpoint_path).and_then(|mut cm| cm.load()) {
            Ok(cp) => cp,
            Err(e) => {
                tracing::error!("No se pudo leer el checkpoint del job '{job_id}': {e}");
                Default::default()
            }
        };

    Json(JobStatusResponse {
        summary: JobSummary::from(&metadata),
        last_chunk: checkpoint.last_chunk,
        processed_seconds: checkpoint.processed_seconds,
    })
    .into_response()
}

#[derive(Debug, Serialize)]
pub struct TranscriptResponse {
    pub status: JobStatus,
    pub transcript_ready: bool,
    pub entries: Vec<TranscriptEntry>,
}

/// `GET /api/jobs/{job_id}/transcript` — contenido de `transcript.jsonl`. Siempre `200` mientras
/// el job exista (`404` solo si no existe/UUID inválido, mismo criterio que el resto de estos
/// endpoints) — el cliente decide qué hacer mirando `status`/`transcript_ready` en el body, no el
/// código HTTP (ver docs/TODO.md: contrato decidido para contenido vacío/parcial). Si
/// `transcript.jsonl` todavía no existe en disco (pipeline no arrancó a escribir), `entries` es
/// `[]`.
pub async fn obtener_transcript_handler(Path(job_id): Path<String>) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    let entries = if std::path::Path::new(&metadata.transcript_path).exists() {
        match leer_jsonl_entries::<TranscriptEntry>(&metadata.transcript_path) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::error!("No se pudo leer el transcript del job '{job_id}': {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Error interno del servidor",
                )
                    .into_response();
            }
        }
    } else {
        Vec::new()
    };

    Json(TranscriptResponse {
        status: metadata.status,
        transcript_ready: metadata.transcript_ready,
        entries,
    })
    .into_response()
}

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub entries: Vec<ChunkMetrics>,
}

/// `GET /api/jobs/{job_id}/metrics` — contenido de `metrics.jsonl` (tiempos por etapa + señales
/// de calidad de whisper-rs por chunk, ver `CLAUDE.local.md`: logger de métricas). Mismo criterio
/// que `obtener_transcript_handler`: `404` solo si el job no existe/UUID inválido, `200` con
/// `entries: []` si el pipeline todavía no escribió ningún chunk.
pub async fn obtener_metricas_handler(Path(job_id): Path<String>) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    let metrics_path = metadata.metrics_path();
    let entries = if metrics_path.exists() {
        match leer_jsonl_entries::<ChunkMetrics>(&metrics_path.to_string_lossy()) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::error!("No se pudo leer las métricas del job '{job_id}': {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Error interno del servidor",
                )
                    .into_response();
            }
        }
    } else {
        Vec::new()
    };

    Json(MetricsResponse { entries }).into_response()
}

/// Lee un `.jsonl` línea por línea (streaming, `BufReader::lines()`, nunca `fs::read_to_string`
/// completo — mismo estilo que `embeddings.rs::run_embedding_phase`). Genérico sobre `T` para
/// reusarlo entre `transcript.jsonl` (`TranscriptEntry`) y `metrics.jsonl` (`ChunkMetrics`).
fn leer_jsonl_entries<T: serde::de::DeserializeOwned>(path: &str) -> anyhow::Result<Vec<T>> {
    let file = std::fs::File::open(path)?;
    let mut entries = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        entries.push(serde_json::from_str(&line)?);
    }
    Ok(entries)
}
