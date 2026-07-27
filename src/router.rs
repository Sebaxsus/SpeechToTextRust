use crate::handlers::audio_handler::recibir_y_procesar_audio;
use crate::state::SharedState;
use axum::{Router, routing::post};

pub fn crear_router(estado: SharedState) -> Router {
    Router::new()
        .route("/api/upload-audio", post(recibir_y_procesar_audio))
        // Aquí agregaremos .route("/api/contexto", post(...)) más adelante
        .with_state(estado)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tower::ServiceExt;

    fn estado_de_prueba() -> SharedState {
        Arc::new(AppState {
            nombre_app: "test".to_string(),
            transcription_semaphore: Arc::new(Semaphore::new(1)),
            ollama: ollama_rs::Ollama::default(),
            qdrant: qdrant_client::Qdrant::from_url("http://localhost:6334")
                .build()
                .expect("cliente de Qdrant de prueba"),
        })
    }

    #[tokio::test]
    async fn ruta_desconocida_devuelve_404() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/no-existe")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upload_audio_con_content_type_invalido_es_rechazado() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header("content-type", "text/plain")
                    .body(Body::from("no soy multipart"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // El extractor Multipart de axum rechaza la petición antes de que el handler corra
        // si el content-type no es multipart/form-data con boundary.
        assert!(response.status().is_client_error());
    }

    fn construir_cuerpo_multipart(
        boundary: &str,
        filename: &str,
        content_type: &str,
        contenido: &[u8],
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(contenido);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn contar_entradas(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir).map(|rd| rd.count()).unwrap_or(0)
    }

    /// No depende de audio real ni del modelo — los magic bytes son basura, así que la
    /// validación de Fase 1 debe rechazar antes de siquiera crear un directorio de job.
    #[tokio::test]
    async fn upload_de_archivo_invalido_es_rechazado_sin_crear_job() {
        let jobs_dir = std::path::Path::new("./jobs");
        let _ = std::fs::create_dir_all(jobs_dir);
        let jobs_antes = contar_entradas(jobs_dir);

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYINVALIDO";
        let body = construir_cuerpo_multipart(
            boundary,
            "campo_basura.bin",
            "application/octet-stream",
            b"esto no es audio, son bytes cualquiera",
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(contar_entradas(jobs_dir), jobs_antes);
    }

    /// Test de punta a punta con un archivo real hardcodeado: sube el mp3 más chico de
    /// `sample_Media/` a través del endpoint real y espera a que el pipeline completo
    /// (decode + resample + chunk + whisper + persistencia JSONL) escriba una transcripción.
    ///
    /// Ignorado por defecto porque depende de dos cosas que no están versionadas en el repo:
    /// `sample_Media/` (audio real del usuario) y `models/ggml-small-q5_1.bin` (el modelo
    /// GGML). Correr con `cargo test -- --ignored --nocapture` después de colocar el modelo.
    #[tokio::test]
    #[ignore = "requiere sample_Media/ real y models/ggml-small-q5_1.bin"]
    async fn pipeline_hardcodeado_transcribe_audio_real() {
        const AUDIO_PATH: &str = "sample_Media/Tuesday at 07_17_42 pm.mp3";

        let audio_bytes = std::fs::read(AUDIO_PATH)
            .unwrap_or_else(|e| panic!("no se pudo leer {AUDIO_PATH}: {e}"));

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYREAL";
        let body = construir_cuerpo_multipart(boundary, "audio.mp3", "audio/mpeg", &audio_bytes);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/upload-audio")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let body_bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let job_id = json["job_id"]
            .as_str()
            .expect("respuesta sin job_id")
            .to_string();

        let transcript_path = format!("./jobs/{job_id}/transcript.jsonl");

        let mut intentos = 0;
        loop {
            if let Ok(contenido) = std::fs::read_to_string(&transcript_path)
                && !contenido.trim().is_empty()
            {
                println!("Transcript generado ({transcript_path}):\n{contenido}");
                assert!(contenido.lines().count() >= 1);
                break;
            }
            intentos += 1;
            assert!(
                intentos < 150,
                "timeout esperando la transcripción en {transcript_path}"
            );
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }
}
