pub mod audio_pipeline;
mod handlers;
mod mcp;
mod rag;
mod router;
mod state;

use state::AppState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Logger centralizado (singleton): `tracing_subscriber::fmt().init()` instala un único
    // `Subscriber` global — cualquier módulo del proyecto llama a `tracing::info!/warn!/error!/
    // debug!` sin pasar ningún logger de un lado a otro. Nivel por defecto `INFO` (igual de
    // visible que los `println!`/`eprintln!` que reemplaza: arranque, warnings de seguridad,
    // errores). `cargo run -- --log` sube a `DEBUG`, que suma las métricas por chunk del pipeline
    // (`audio_pipeline::pipeline::run_pipeline`) — ver CLAUDE.local.md: logger de métricas.
    let modo_debug = std::env::args().any(|arg| arg == "--log");
    let nivel = if modo_debug {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(nivel).init();

    // Clientes livianos: no cargan ningún modelo, solo abren la conexión — construirlos acá no
    // viola la regla de "nunca dejar Whisper/Ollama residentes" (ver CLAUDE.local.md).
    let ollama = ollama_rs::Ollama::default(); // http://127.0.0.1:11434
    let qdrant = qdrant_client::Qdrant::from_url("http://localhost:6334")
        .build()
        .expect("no se pudo construir el cliente de Qdrant (¿URL inválida?)");

    // 1. Inicializamos nuestro estado compartido
    let estado = Arc::new(AppState {
        nombre_app: "Rust RAG Local-First".to_string(),
        heavy_compute_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        ollama,
        qdrant,
    });

    // 2. Creamos el enrutador de Axum
    let app = router::crear_router(estado);

    // 3. Levantamos el servidor en el puerto 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::info!("Servidor Axum corriendo en http://localhost:3000");
    tracing::info!(
        "Endpoint MCP (Streamable HTTP, Fase 6) en http://localhost:3000/mcp — probar con la \
         extensión de HTTP requests de VSCode."
    );
    if std::env::var("MCP_BEARER_TOKEN").is_err() {
        tracing::warn!(
            "MCP_BEARER_TOKEN no configurado: /mcp queda sin autenticación (ok solo para \
             pruebas locales, ver CLAUDE.local.md: Fase 6)."
        );
    }

    axum::serve(listener, app).await.unwrap();
}
