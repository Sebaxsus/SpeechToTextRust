//! Endpoints REST de solo lectura sobre jobs (`GET /api/jobs`, `GET /api/jobs/{job_id}`,
//! `GET /api/jobs/{job_id}/transcript`) — wrappers finos sobre `audio_pipeline::job`, pensados
//! para que un cliente web no tenga que hablar el protocolo MCP completo solo para leer estado.
//! Montados bajo el mismo middleware de bearer token que `/mcp` (ver `router.rs`), porque
//! exponen el mismo contenido transcrito sensible.

use std::io::{BufRead, BufReader};

use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::Serialize;

use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::job::{list_jobs, load_job};
use crate::audio_pipeline::models::{JobMetadata, JobStatus, TranscriptEntry};

/// Subconjunto público de `JobMetadata` — nunca expone `audio_path`/`transcript_path`/
/// `checkpoint_path` (rutas de disco del servidor, estructura interna que un cliente remoto no
/// necesita).
#[derive(Debug, Serialize)]
pub struct JobSummary {
    pub job_id: String,
    pub status: JobStatus,
    pub transcript_ready: bool,
    pub created_at: String,
}

impl From<&JobMetadata> for JobSummary {
    fn from(metadata: &JobMetadata) -> Self {
        Self {
            job_id: metadata.job_id.clone(),
            status: metadata.status,
            transcript_ready: metadata.transcript_ready,
            created_at: metadata.created_at.clone(),
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
            eprintln!("Error listando jobs: {e}");
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
            eprintln!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    // Sin checkpoint todavía (job recién creado o aún en cola) no es un error — CheckpointManager
    // ya devuelve el default (0, 0.0) en ese caso (ver checkpoint.rs).
    let checkpoint =
        match CheckpointManager::new(&metadata.checkpoint_path).and_then(|mut cm| cm.load()) {
            Ok(cp) => cp,
            Err(e) => {
                eprintln!("No se pudo leer el checkpoint del job '{job_id}': {e}");
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
            eprintln!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    let entries = if std::path::Path::new(&metadata.transcript_path).exists() {
        match leer_transcript_entries(&metadata.transcript_path) {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!("No se pudo leer el transcript del job '{job_id}': {e}");
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

/// Lee `transcript.jsonl` línea por línea (streaming, `BufReader::lines()`, nunca
/// `fs::read_to_string` completo — mismo estilo que `embeddings.rs::run_embedding_phase`) y arma
/// el `Vec<TranscriptEntry>` de la respuesta.
fn leer_transcript_entries(transcript_path: &str) -> anyhow::Result<Vec<TranscriptEntry>> {
    let file = std::fs::File::open(transcript_path)?;
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
