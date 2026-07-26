pub mod audio_pipeline;
mod handlers;
mod router;
mod state;

use state::AppState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // 1. Inicializamos nuestro estado compartido
    let estado = Arc::new(AppState {
        nombre_app: "Rust RAG Local-First".to_string(),
        transcription_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
    });

    // 2. Creamos el enrutador de Axum
    let app = router::crear_router(estado);

    // 3. Levantamos el servidor en el puerto 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Servidor Axum corriendo en http://localhost:3000");

    axum::serve(listener, app).await.unwrap();
}
