use axum::{routing::post, Router};
use crate::state::SharedState;
use crate::handlers::audio_handler::recibir_y_procesar_audio;

pub fn crear_router(estado: SharedState) -> Router {
    Router::new()
        .route("/api/upload-audio", post(recibir_y_procesar_audio))
        // Aquí agregaremos .route("/api/contexto", post(...)) más adelante
        .with_state(estado)
}