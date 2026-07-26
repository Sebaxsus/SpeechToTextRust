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
}
