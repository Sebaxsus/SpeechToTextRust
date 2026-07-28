use crate::audio_pipeline::embeddings::run_embedding_phase;
use crate::audio_pipeline::job::{create_job, load_job};
use crate::audio_pipeline::models::JobMetadata;
use crate::audio_pipeline::pipeline::run_pipeline;
use crate::state::SharedState;
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use tokio::io::AsyncWriteExt;

/// Detecta el contenedor real a partir de los primeros bytes del archivo (magic bytes), en vez
/// de confiar en la extensión declarada por el cliente. Devuelve la extensión interna ("mp3" /
/// "mp4") que se usa después para el nombre de archivo en disco, o `None` si no es un formato
/// soportado.
///
/// mp4 y m4a son el mismo contenedor ISO-BMFF (caja `ftyp`), así que ambos caen en "mp4".
fn sniff_audio_extension(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return Some("mp3");
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Some("mp3");
    }
    if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        return Some("mp4");
    }
    None
}

pub async fn recibir_y_procesar_audio(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    println!("Recibiendo petición en la app: {}", state.nombre_app);

    let Ok(Some(mut campo)) = multipart.next_field().await else {
        return (
            StatusCode::BAD_REQUEST,
            "Error al leer el archivo multipart",
        )
            .into_response();
    };

    let nombre_archivo = campo.file_name().unwrap_or("audio_desconocido").to_string();

    // Sniff del primer chunk ANTES de crear cualquier archivo o directorio en disco.
    let primer_chunk = match campo.chunk().await {
        Ok(Some(chunk)) => chunk,
        _ => {
            return (StatusCode::BAD_REQUEST, "Archivo vacío o inválido").into_response();
        }
    };

    let Some(extension) = sniff_audio_extension(&primer_chunk) else {
        return (
            StatusCode::BAD_REQUEST,
            "Formato no soportado: solo se aceptan mp3 y mp4/m4a",
        )
            .into_response();
    };

    let metadata = match create_job(extension) {
        Ok(metadata) => metadata,
        Err(e) => {
            eprintln!("Error creando el job: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response();
        }
    };

    let mut archivo_local = match tokio::fs::File::create(&metadata.audio_path).await {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error al crear el archivo: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response();
        }
    };

    // Streaming directo al SSD para evitar Thrashing de la RAM.
    if let Err(e) = archivo_local.write_all(&primer_chunk).await {
        eprintln!("Error escribiendo en disco: {}", e);
        let _ = tokio::fs::remove_dir_all(format!("./jobs/{}", metadata.job_id)).await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error al guardar el archivo",
        )
            .into_response();
    }

    while let Ok(Some(chunk)) = campo.chunk().await {
        if let Err(e) = archivo_local.write_all(&chunk).await {
            eprintln!("Error escribiendo en disco: {}", e);
            let _ = tokio::fs::remove_dir_all(format!("./jobs/{}", metadata.job_id)).await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error al guardar el archivo",
            )
                .into_response();
        }
    }

    let job_id = metadata.job_id.clone();

    println!(
        "Archivo {} guardado exitosamente en el SSD (job {}).",
        nombre_archivo, job_id
    );

    lanzar_procesamiento_job(state, metadata);

    (
        StatusCode::ACCEPTED,
        axum::Json(serde_json::json!({
            "job_id": job_id,
            "status": "processing",
        })),
    )
        .into_response()
}

/// Corre Fase 2/3 (Whisper) y, si termina bien, encadena Fase 4 (embeddings) — la misma cadena
/// tanto para un job recién creado (`recibir_y_procesar_audio`) como para uno reanudado
/// (`reanudar_job`). No hace falta distinguir "nuevo" de "reanudado" acá: `run_pipeline` ya es
/// resume-safe por sí mismo (lee `checkpoint.json` y retoma donde cortó — ver `pipeline.rs`), y
/// `run_embedding_phase` es idempotente (point ID determinístico), así que reintentar Fase 4
/// sobre un transcript ya embebido tampoco duplica vectores.
fn lanzar_procesamiento_job(state: SharedState, metadata: JobMetadata) {
    tokio::spawn(async move {
        let permit = state
            .transcription_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let _permit = permit;

        let audio_id = metadata.job_id.clone();
        let transcript_path = metadata.transcript_path.clone();

        // Fase 2/3 (Whisper, CPU-bound) corre entera dentro de spawn_blocking y termina de
        // liberar WhisperRunner antes de que Fase 4 (embeddings, I/O-bound) arranque — nunca
        // se solapan Whisper y Ollama (ver CLAUDE.local.md: Concurrencia).
        match tokio::task::spawn_blocking(move || run_pipeline(metadata)).await {
            Ok(Err(e)) => eprintln!("Error en el pipeline: {}", e),
            Err(e) => eprintln!("Error en el pipeline (join error): {}", e),
            Ok(Ok(())) => {
                if let Err(e) =
                    run_embedding_phase(&state.ollama, &state.qdrant, &audio_id, &transcript_path)
                        .await
                {
                    eprintln!("Error en la fase de embeddings: {}", e);
                }
            }
        }
    });
}

/// `POST /api/jobs/{job_id}/resume` — reanuda un job existente cuyo procesamiento se cortó a
/// mitad de camino (proceso matado, crash, etc. — ver docs/TODO.md, Fase 3). `run_pipeline` ya
/// era resume-safe a nivel de función; lo que faltaba era este entry point HTTP.
///
/// Deliberadamente NO es una tool de MCP (ver CLAUDE.local.md: "MCP de solo lectura") — el
/// servidor MCP nunca dispara procesamiento, ni siquiera para "solo continuar" algo ya pedido,
/// porque esa es la línea que hace aceptable exponerlo en LAN.
pub async fn reanudar_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            eprintln!("No se pudo reanudar el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    lanzar_procesamiento_job(state, metadata);

    (
        StatusCode::ACCEPTED,
        axum::Json(serde_json::json!({
            "job_id": job_id,
            "status": "processing",
        })),
    )
        .into_response()
}
