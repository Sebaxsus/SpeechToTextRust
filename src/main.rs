pub mod audio_pipeline;
mod handlers;
mod router;
mod state;

use state::AppState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Clientes livianos: no cargan ningún modelo, solo abren la conexión — construirlos acá no
    // viola la regla de "nunca dejar Whisper/Ollama residentes" (ver CLAUDE.local.md).
    let ollama = ollama_rs::Ollama::default(); // http://127.0.0.1:11434
    let qdrant = qdrant_client::Qdrant::from_url("http://localhost:6334")
        .build()
        .expect("no se pudo construir el cliente de Qdrant (¿URL inválida?)");

    // 1. Inicializamos nuestro estado compartido
    let estado = Arc::new(AppState {
        nombre_app: "Rust RAG Local-First".to_string(),
        transcription_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        ollama,
        qdrant,
    });

    // 2. Creamos el enrutador de Axum
    let app = router::crear_router(estado);

    // 3. Levantamos el servidor en el puerto 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Servidor Axum corriendo en http://localhost:3000");

    axum::serve(listener, app).await.unwrap();
}
