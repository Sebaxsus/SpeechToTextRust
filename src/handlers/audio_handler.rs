use crate::audio_pipeline::job::create_job;
use crate::audio_pipeline::pipeline::run_pipeline;
use crate::state::SharedState;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

pub async fn recibir_y_procesar_audio(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    println!("Recibiendo petición en la app: {}", state.nombre_app);

    // Verificamos que la carpeta uploads exista, si no, la creamos
    let _ = tokio::fs::create_dir_all("./uploads").await;

    if let Ok(Some(mut campo)) = multipart.next_field().await {
        let nombre_archivo = campo.file_name().unwrap_or("audio_desconocido").to_string();
        let ruta_destino = format!("./uploads/{}", nombre_archivo);

        let mut archivo_local = match tokio::fs::File::create(&ruta_destino).await {
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

        // Streaming directo al SSD para evitar Thrashing de la RAM
        while let Ok(Some(chunk)) = campo.chunk().await {
            if let Err(e) = archivo_local.write_all(&chunk).await {
                eprintln!("Error escribiendo en disco: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Error al guardar el archivo",
                )
                    .into_response();
            }
        }

        println!(
            "Archivo {} guardado exitosamente en el SSD.",
            nombre_archivo
        );

        tokio::spawn({
            let state = state.clone();
            let audio_path = PathBuf::from(&ruta_destino);

            async move {
                let permit = state
                    .transcription_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();

                let _permit = permit;

                let job = match create_job(audio_path) {
                    Ok(job) => job,
                    Err(e) => {
                        eprintln!("Error creando el job: {}", e);
                        return;
                    }
                };

                let _ = tokio::task::spawn_blocking(move || run_pipeline(job)).await;
            }
        });

        (
            StatusCode::ACCEPTED,
            format!(
                "Archivo {} recibido. Transcripción en progreso...",
                nombre_archivo
            ),
        )
            .into_response()
    } else {
        (
            StatusCode::BAD_REQUEST,
            "Error al leer el archivo multipart",
        )
            .into_response()
    }
}
