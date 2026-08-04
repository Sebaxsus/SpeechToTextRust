pub mod audio_pipeline;
mod handlers;
mod mcp;
mod rag;
mod router;
mod state;

use state::AppState;
use std::sync::Arc;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::{LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() {
    // Logger centralizado: dos `Layer`s sobre el mismo `Subscriber` global (registry), cada uno
    // con su propio filtro — consola curada, archivo completo. Cualquier módulo llama a
    // `tracing::info!/warn!/error!/debug!` sin pasar ningún logger de un lado a otro.
    //
    // Consola: solo eventos marcados `target: "lifecycle"` (las transiciones de estado de un job
    // — "transcribiendo", "resumiendo", "generando resumen", "enviando estado" — ver
    // `handlers::audio_handler`/`handlers::jobs_handler`) más cualquier WARN/ERROR de cualquier
    // módulo (una degradación tolerada o un error real siempre es "información importante",
    // aunque no esté taggeado). `cargo run -- --log` vuelve al comportamiento viejo (todo, DEBUG
    // incluido) para debugging interactivo en vivo.
    //
    // Archivo (`logs/server.YYYY-MM-DD.log`, rotación diaria): SIEMPRE a nivel DEBUG,
    // independientemente de `--log` — es "el log real": arranca en las mismas transiciones de
    // `lifecycle` y sigue con el detalle completo (métricas por chunk de whisper en
    // `audio_pipeline::pipeline::run_pipeline`, y lo que devuelven Ollama/Qdrant en
    // `audio_pipeline::embeddings`/`rag::summary`/`rag::retrieval`/`rag::generation`).
    let modo_debug = std::env::args().any(|arg| arg == "--log");

    // `tracing_appender::rolling` no crea el directorio destino solo — mismo patrón que
    // `audio_pipeline::job::create_job` para `./jobs/`.
    if let Err(e) = std::fs::create_dir_all("logs") {
        eprintln!("No se pudo crear el directorio ./logs: {e}");
    }
    let file_appender = tracing_appender::rolling::daily("logs", "server.log");
    // `_log_guard` tiene que vivir hasta el final de `main` (por eso el `_` en vez de `_guard`
    // descartable): al dropearse, `non_blocking` deja de flushear su thread de escritura — si se
    // dropeara acá, el archivo se quedaría sin las últimas líneas de cada corrida.
    let (file_writer, _log_guard) = tracing_appender::non_blocking(file_appender);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(LevelFilter::DEBUG);

    let console_filter = if modo_debug {
        Targets::new().with_default(LevelFilter::DEBUG)
    } else {
        Targets::new()
            .with_target("lifecycle", LevelFilter::INFO)
            .with_default(LevelFilter::WARN)
    };
    let console_layer = tracing_subscriber::fmt::layer().with_filter(console_filter);

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    // Clientes livianos: no cargan ningún modelo, solo abren la conexión — construirlos acá no
    // viola la regla de "nunca dejar Whisper/Ollama residentes" (ver CLAUDE.local.md).
    let ollama = ollama_rs::Ollama::default(); // http://127.0.0.1:11434
    let qdrant = qdrant_client::Qdrant::from_url("http://localhost:6334")
        .build()
        .expect("no se pudo construir el cliente de Qdrant (¿URL inválida?)");

    // Token compartido con `mcp::build_service` (vía `AppState`) — cancelarlo en el shutdown le
    // avisa a `LocalSessionManager` que cierre las sesiones MCP abiertas en vez de dejarlas
    // colgadas hasta que el proceso muera. Se clona ANTES de mover `estado` a `crear_router`,
    // porque `esperar_ctrl_c` necesita su propia copia para cancelarlo más abajo.
    let mcp_cancellation_token = tokio_util::sync::CancellationToken::new();

    // 1. Inicializamos nuestro estado compartido
    let estado = Arc::new(AppState {
        nombre_app: "Rust RAG Local-First".to_string(),
        heavy_compute_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        ollama,
        qdrant,
        mcp_cancellation_token: mcp_cancellation_token.clone(),
    });

    // 2. Creamos el enrutador de Axum
    let app = router::crear_router(estado);

    // 3. Levantamos el servidor en el puerto 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::info!(target: "lifecycle", "Servidor Axum corriendo en http://localhost:3000");
    tracing::info!(
        target: "lifecycle",
        "Endpoint MCP (Streamable HTTP, Fase 6) en http://localhost:3000/mcp — probar con la \
         extensión de HTTP requests de VSCode."
    );
    if std::env::var("MCP_BEARER_TOKEN").is_err() {
        tracing::warn!(
            "MCP_BEARER_TOKEN no configurado: /mcp queda sin autenticación (ok solo para \
             pruebas locales, ver CLAUDE.local.md: Fase 6)."
        );
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(esperar_ctrl_c(mcp_cancellation_token))
        .await
        .unwrap();
}

/// Espera CTRL+C (funciona igual en Windows y Unix, `tokio::signal::ctrl_c()` es cross-platform),
/// cancela las sesiones MCP abiertas, y deja que `axum::serve` drene las requests HTTP en curso
/// antes de salir. No cubre el trabajo CPU-bound de un chunk de Whisper a mitad de
/// `spawn_blocking` — ese progreso parcial se pierde igual que con un kill directo, pero por
/// diseño no es un problema: `checkpoint.json` solo avanza tras un chunk completo, así que el
/// próximo `POST /api/jobs/{job_id}/resume` retoma exactamente ahí (ver CLAUDE.local.md:
/// Persistencia — checkpoint y resume).
async fn esperar_ctrl_c(mcp_cancellation_token: tokio_util::sync::CancellationToken) {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!(target: "lifecycle", "CTRL+C recibido, cerrando el servidor (drenando requests en curso)...");
    mcp_cancellation_token.cancel();
}
