//! Endpoints REST de solo lectura sobre jobs (`GET /api/jobs`, `GET /api/jobs/{job_id}`,
//! `GET /api/jobs/{job_id}/transcript`, `GET /api/jobs/{job_id}/metrics`) — wrappers finos sobre
//! `audio_pipeline::job`, pensados para que un cliente web no tenga que hablar el protocolo MCP
//! completo solo para leer estado. Montados bajo el mismo middleware de bearer token que `/mcp`
//! (ver `router.rs`), porque exponen el mismo contenido transcrito sensible.

use std::io::{BufRead, BufReader};
use std::process::Stdio;

use axum::body::Body;
use axum::extract::Query;
use axum::http::header::CONTENT_TYPE;
use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::io::ReaderStream;

use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::job::{list_jobs, load_job};
use crate::audio_pipeline::models::{
    ChunkMetrics, JobMetadata, JobStatus, SummaryStatus, TranscriptEntry,
};

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
    /// Ver `SummaryStatus` — estado del resumen por audio (map-reduce, tarea independiente
    /// disparada después de `Completed`, ver `handlers::audio_handler::lanzar_generacion_resumen`).
    pub summary_status: SummaryStatus,
    /// Contenido de `summary.txt`, leído solo cuando `summary_status == Ready` — `None` en
    /// cualquier otro caso (nunca un endpoint separado para esto, alcanza con este campo).
    pub summary: Option<String>,
    /// Ver `JobMetadata::original_filename` — nombre de archivo original del multipart, `None`
    /// si el cliente no lo mandó. Nunca es una ruta de disco.
    pub original_filename: Option<String>,
    /// Ver `JobMetadata::duration_seconds` — `None` hasta que `ffprobe` termine de medirlo (o si
    /// falló al medirlo).
    pub duration_seconds: Option<f32>,
    /// Ver `JobMetadata::title` — título elegido por el usuario en el campo `title` del
    /// multipart, `None` si no lo mandó (o si mandó un valor en blanco). El fallback
    /// `title ?? original_filename ?? job_id` para mostrar en el dashboard es responsabilidad del
    /// cliente, no se resuelve acá.
    pub title: Option<String>,
}

impl From<&JobMetadata> for JobSummary {
    fn from(metadata: &JobMetadata) -> Self {
        let summary = if metadata.summary_status == SummaryStatus::Ready {
            match std::fs::read_to_string(metadata.summary_path()) {
                Ok(texto) => Some(texto),
                Err(e) => {
                    tracing::error!(
                        "summary_status es Ready pero no se pudo leer summary.txt del job '{}': {e}",
                        metadata.job_id
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            job_id: metadata.job_id.clone(),
            status: metadata.status,
            transcript_ready: metadata.transcript_ready,
            created_at: metadata.created_at.clone(),
            processing_started_at: metadata.processing_started_at.clone(),
            transcript_ready_at: metadata.transcript_ready_at.clone(),
            completed_at: metadata.completed_at.clone(),
            summary_status: metadata.summary_status,
            summary,
            original_filename: metadata.original_filename.clone(),
            duration_seconds: metadata.duration_seconds,
            title: metadata.title.clone(),
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

    tracing::info!(target: "lifecycle", job_id = %job_id, "Enviando el estado del job");

    Json(JobStatusResponse {
        summary: JobSummary::from(&metadata),
        last_chunk: checkpoint.last_chunk,
        processed_seconds: checkpoint.processed_seconds,
    })
    .into_response()
}

/// Envuelve `TranscriptEntry` con un campo calculado (nunca persistido) `low_confidence` — el
/// frontend nunca necesita conocer el umbral (`RagConfig::low_confidence_thold`), solo lee un
/// booleano ya resuelto por el servidor. Cambiar el umbral a futuro no requiere reprocesar nada:
/// se recalcula en cada lectura.
#[derive(Debug, Serialize)]
struct TranscriptEntryView {
    #[serde(flatten)]
    entry: TranscriptEntry,
    low_confidence: bool,
}

impl From<TranscriptEntry> for TranscriptEntryView {
    fn from(entry: TranscriptEntry) -> Self {
        let low_confidence = entry.avg_logprob < crate::config::get().rag.low_confidence_thold;
        Self {
            entry,
            low_confidence,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TranscriptResponse {
    pub status: JobStatus,
    pub transcript_ready: bool,
    entries: Vec<TranscriptEntryView>,
}

/// Arma la respuesta de transcript completo a partir de metadata ya cargada — compartida entre
/// `obtener_transcript_handler` (REST) y la tool MCP `get_transcript` (`mcp::mod`), para no
/// duplicar el chequeo de "transcript.jsonl no existe todavía" → `entries: []`. No valida
/// `job_id` ni resuelve `metadata` — eso es responsabilidad de cada caller (cada uno tiene su
/// propio criterio de error: `404` HTTP vs `McpError::invalid_params`).
pub(crate) fn construir_transcript_response(
    metadata: &JobMetadata,
) -> anyhow::Result<TranscriptResponse> {
    let entries: Vec<TranscriptEntryView> =
        if std::path::Path::new(&metadata.transcript_path).exists() {
            leer_jsonl_entries::<TranscriptEntry>(&metadata.transcript_path)?
                .into_iter()
                .map(TranscriptEntryView::from)
                .collect()
        } else {
            Vec::new()
        };

    Ok(TranscriptResponse {
        status: metadata.status,
        transcript_ready: metadata.transcript_ready,
        entries,
    })
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

    match construir_transcript_response(&metadata) {
        Ok(response) => Json(response).into_response(),
        Err(e) => {
            tracing::error!("No se pudo leer el transcript del job '{job_id}': {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response()
        }
    }
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

#[derive(Debug, Deserialize)]
pub struct AudioSegmentParams {
    pub start: f32,
    pub end: f32,
}

/// `GET /api/jobs/{job_id}/audio-segment?start=..&end=..` — transcodea al vuelo con `ffmpeg` el
/// rango `[start, end)` (segundos) del audio original a mp3, para que el cliente web pueda
/// escuchar un segmento marcado con `avg_logprob` bajo (ver CLAUDE.local.md: reproducción de
/// segmentos). Reusa el binario `ffmpeg` ya requerido por el pipeline (mismo orden `-ss` antes de
/// `-i` que `audio_pipeline::decoder` para seek rápido de input) — no persiste nada nuevo en
/// disco, sirve la calidad real del archivo subido (no el PCM 16kHz que Whisper procesó).
///
/// `tokio::process::Command` (async, no el `std::process::Command` síncrono que usa el decoder
/// dentro de `spawn_blocking`) porque este handler no hace trabajo CPU-bound propio, solo proxea
/// la salida de `ffmpeg`. `.kill_on_drop(true)` es el equivalente nativo de tokio al `Drop for
/// Source` de `decoder.rs`: si el cliente corta la conexión y el stream se dropea antes de
/// terminar, `ffmpeg` se mata solo, sin necesitar un guard manual.
///
/// No pasa por `heavy_compute_semaphore`: transcodificar unos segundos/minutos de audio es una
/// carga de CPU corta, no comparable a Whisper/Ollama — mismo criterio que el fallback de
/// `ffmpeg` en el decoder, que tampoco pasa por el semáforo.
pub async fn obtener_segmento_audio_handler(
    Path(job_id): Path<String>,
    Query(params): Query<AudioSegmentParams>,
) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("No se pudo leer el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    let max_segment_seconds = crate::config::get().rag.max_segment_seconds;
    let duracion = params.end - params.start;
    if params.start < 0.0 || duracion <= 0.0 || duracion > max_segment_seconds {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "Rango inválido: start debe ser >= 0, end > start, y la duración no puede \
                 superar {max_segment_seconds} segundos"
            ),
        )
            .into_response();
    }

    let mut child = match Command::new(&crate::config::get().paths.ffmpeg_bin)
        .args([
            "-v",
            "error",
            "-ss",
            &params.start.to_string(),
            "-to",
            &params.end.to_string(),
            "-i",
            &metadata.audio_path,
            "-f",
            "mp3",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            tracing::error!("No se pudo ejecutar ffmpeg para el job '{job_id}': {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response();
        }
    };

    let stdout = child.stdout.take().expect("stdout configurado como piped");
    let stream = ReaderStream::new(stdout);

    // El proceso sigue corriendo en background hasta terminar (o hasta que `kill_on_drop` lo
    // mate si el stream se dropea antes, ej. el cliente corta la conexión) — no bloqueamos la
    // respuesta esperando su `wait()`. Limitación conocida: si `ffmpeg` falla a mitad de la
    // transcodificación, el cliente ya recibió el header `200` y el audio llega truncado/corrupto
    // en vez de un error HTTP — inherente a cualquier respuesta streaming, no resoluble sin
    // bufferear todo el clip antes de responder (lo que iría contra streaming).
    tokio::spawn(async move {
        match child.wait().await {
            Ok(status) if !status.success() => {
                let mut stderr_buf = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_string(&mut stderr_buf).await;
                }
                tracing::warn!(
                    "ffmpeg terminó con error extrayendo un segmento del job '{job_id}': {stderr_buf}"
                );
            }
            Err(e) => tracing::warn!("No se pudo esperar el proceso ffmpeg (job '{job_id}'): {e}"),
            _ => {}
        }
    });

    ([(CONTENT_TYPE, "audio/mpeg")], Body::from_stream(stream)).into_response()
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
