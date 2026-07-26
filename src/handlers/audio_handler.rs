use crate::audio_pipeline::job::create_job;
use crate::audio_pipeline::pipeline::run_pipeline;
use crate::state::SharedState;
use axum::{
    extract::{Multipart, State},
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

    tokio::spawn({
        let state = state.clone();

        async move {
            let permit = state
                .transcription_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            let _permit = permit;

            match tokio::task::spawn_blocking(move || run_pipeline(metadata)).await {
                Ok(Err(e)) => eprintln!("Error en el pipeline: {}", e),
                Err(e) => eprintln!("Error en el pipeline (join error): {}", e),
                Ok(Ok(())) => {}
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        axum::Json(serde_json::json!({
            "job_id": job_id,
            "status": "processing",
        })),
    )
        .into_response()
}
