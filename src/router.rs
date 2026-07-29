use crate::handlers::audio_handler::{reanudar_job, recibir_y_procesar_audio};
use crate::mcp;
use crate::state::SharedState;
use axum::{
    Router,
    extract::{Request, State},
    http::{StatusCode, header::AUTHORIZATION},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
};
use std::sync::Arc;

/// Token opcional para autenticar el endpoint `/mcp` (ver CLAUDE.local.md: "MCP de solo
/// lectura" — "Bearer token en el endpoint MCP antes de bindear a la IP de LAN"). Leído una sola
/// vez al armar el router, no en cada request.
#[derive(Clone)]
struct McpAuthState {
    token: Option<Arc<str>>,
}

async fn exigir_bearer_token(
    State(auth): State<McpAuthState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(esperado) = &auth.token else {
        // Sin MCP_BEARER_TOKEN configurado: sin autenticación. Aceptable solo mientras el
        // servidor no se expone más allá de localhost — ver advertencia impresa en main.rs.
        return next.run(req).await;
    };

    let autorizado = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|token| token == esperado.as_ref());

    if autorizado {
        next.run(req).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

/// Router del endpoint MCP (`/mcp`, Streamable HTTP) separado del router principal porque tiene
/// su propio estado de autenticación (`McpAuthState`), distinto de `SharedState` — se mergea
/// después vía `Router::merge`, no `nest`, así que ambos quedan al mismo nivel (`/api/...` y
/// `/mcp` como rutas hermanas, no una anidada dentro de la otra a nivel de path).
fn crear_router_mcp(estado: SharedState) -> Router {
    let token = std::env::var("MCP_BEARER_TOKEN").ok().map(Arc::from);
    let auth_state = McpAuthState { token };

    Router::new()
        .nest_service("/mcp", mcp::build_service(estado))
        .route_layer(middleware::from_fn_with_state(
            auth_state.clone(),
            exigir_bearer_token,
        ))
        .with_state(auth_state)
}

pub fn crear_router(estado: SharedState) -> Router {
    Router::new()
        .route("/api/upload-audio", post(recibir_y_procesar_audio))
        .route("/api/jobs/{job_id}/resume", post(reanudar_job))
        .with_state(estado.clone())
        .merge(crear_router_mcp(estado))
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
            heavy_compute_semaphore: Arc::new(Semaphore::new(1)),
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

    #[tokio::test]
    async fn resume_de_job_inexistente_devuelve_404() {
        let app = crear_router(estado_de_prueba());
        let job_id = uuid::Uuid::new_v4();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/jobs/{job_id}/resume"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// `job_id` llega crudo desde el path de la URL — un valor que no es un UUID (ej. un intento
    /// de path traversal como `../../etc`) tiene que rechazarse igual que uno bien formado pero
    /// inexistente, nunca usarse tal cual para construir una ruta de disco (ver
    /// `audio_pipeline::job::load_job`).
    #[tokio::test]
    async fn resume_con_job_id_no_uuid_devuelve_404_sin_tocar_disco_fuera_de_jobs() {
        let app = crear_router(estado_de_prueba());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/jobs/no-es-un-uuid/resume")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
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

    /// Ruta del audio real usado por el test de punta a punta — configurable vía la variable de
    /// entorno `TEST_AUDIO_PATH` (no vía argumento de `cargo test`, que el harness de test
    /// integrado de Rust no expone al binario) para poder probar con cualquier archivo de
    /// `sample_Media/` sin tocar código. Por defecto usa la muestra corta (~2 min) para que la
    /// corrida sea rápida durante el desarrollo de las fases siguientes.
    fn ruta_audio_de_prueba() -> String {
        std::env::var("TEST_AUDIO_PATH")
            .unwrap_or_else(|_| "sample_Media/Muestra2_02min.m4a".to_string())
    }

    /// Content-type + nombre de archivo acordes a la extensión real, solo por prolijidad del
    /// multipart de prueba — el handler nunca confía en esto, sniffea magic bytes (ver
    /// `audio_handler.rs`).
    fn content_type_y_nombre(path: &str) -> (&'static str, String) {
        let filename = std::path::Path::new(path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("audio")
            .to_string();
        let content_type = if path.to_lowercase().ends_with(".mp3") {
            "audio/mpeg"
        } else {
            "audio/mp4"
        };
        (content_type, filename)
    }

    /// Test de punta a punta con un archivo real: sube el audio de `ruta_audio_de_prueba()` a
    /// través del endpoint real, espera a que el pipeline completo (decode + resample + chunk +
    /// whisper + persistencia JSONL, Fase 2/3) escriba una transcripción, y además confirma que
    /// la Fase 4 (embeddings) encadenada después del pipeline (ver `audio_handler.rs`) terminó
    /// insertando puntos en Qdrant para este job — es el primer test que ejercita las fases entre
    /// sí en vez de aisladas, base para construir el retrieval de Fase 5 encima.
    ///
    /// Ignorado por defecto porque depende de piezas que no están versionadas ni siempre
    /// levantadas: `sample_Media/` (audio real del usuario), `models/ggml-small-q5_1.bin` (el
    /// modelo GGML), y Ollama (`bge-m3`) + Qdrant reales corriendo en localhost. Correr con
    /// `cargo test pipeline_hardcodeado -- --ignored --nocapture`, o con otra muestra:
    /// `TEST_AUDIO_PATH="sample_Media/otro.m4a" cargo test pipeline_hardcodeado -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "requiere sample_Media/ real, models/ggml-small-q5_1.bin, y Ollama+Qdrant reales"]
    async fn pipeline_hardcodeado_transcribe_audio_real() {
        let audio_path = ruta_audio_de_prueba();

        let audio_bytes = std::fs::read(&audio_path)
            .unwrap_or_else(|e| panic!("no se pudo leer {audio_path}: {e}"));

        let app = crear_router(estado_de_prueba());
        let boundary = "TESTBOUNDARYREAL";
        let (content_type, filename) = content_type_y_nombre(&audio_path);
        let body = construir_cuerpo_multipart(boundary, &filename, content_type, &audio_bytes);

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

        // Fase 4 corre encadenada justo después de Fase 2/3 (ver audio_handler.rs) — confirmar
        // que los embeddings de este job realmente llegaron a Qdrant es lo que hace de este test
        // una prueba de los módulos entre sí, no solo de Whisper en aislamiento.
        let qdrant = qdrant_client::Qdrant::from_url("http://localhost:6334")
            .build()
            .expect("cliente de Qdrant de prueba");
        let filter =
            qdrant_client::qdrant::Filter::must([qdrant_client::qdrant::Condition::matches(
                "audio_id",
                job_id.clone(),
            )]);

        let mut intentos = 0;
        loop {
            let count = qdrant
                .count(
                    qdrant_client::qdrant::CountPointsBuilder::new("transcripts")
                        .filter(filter.clone())
                        .exact(true),
                )
                .await
                .expect("no se pudo contar los puntos del job en Qdrant")
                .result
                .expect("respuesta de count sin resultado")
                .count;

            if count > 0 {
                println!("Embeddings del job {job_id} encontrados en Qdrant: {count} puntos");
                break;
            }

            intentos += 1;
            assert!(
                intentos < 60,
                "timeout esperando los embeddings del job {job_id} en Qdrant"
            );
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }

        // Limpieza: no dejar puntos de este job de prueba en la colección real.
        let _ = qdrant
            .delete_points(
                qdrant_client::qdrant::DeletePointsBuilder::new("transcripts").points(filter),
            )
            .await;
    }
}
