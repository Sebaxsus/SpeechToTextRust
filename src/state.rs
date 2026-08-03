use ollama_rs::Ollama;
use qdrant_client::Qdrant;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

pub struct AppState {
    pub nombre_app: String,
    // 1 permit: como máximo una carga pesada de CPU/modelo a la vez en todo el proceso —
    // Whisper (audio_pipeline::pipeline::run_pipeline) y RAG server-side (rag::generation::
    // rag_answer, rag::reranker vía las tools de MCP) compiten por el mismo único permiso, nunca
    // corren dos a la vez sea cual sea la combinación (ver CLAUDE.local.md: Concurrencia — "NO
    // paralelismo agresivo").
    pub heavy_compute_semaphore: Arc<Semaphore>,
    // Clientes livianos (no cargan modelos, no son lo mismo que dejar Whisper/Ollama
    // residentes) — construidos una vez en main.rs y compartidos vía Arc<AppState>.
    pub ollama: Ollama,
    // Qdrant nunca se expone en la LAN — solo bindeado a localhost (ver CLAUDE.local.md).
    pub qdrant: Qdrant,
    // Cancela las sesiones MCP abiertas (`StreamableHttpServerConfig::cancellation_token`, ver
    // `mcp::build_service`) al recibir CTRL+C — sin esto, `LocalSessionManager` no se entera del
    // shutdown y las sesiones quedan "abiertas" del lado del servidor hasta que el proceso muere
    // (ver docs/TODO.md: "Sin CancellationToken propio ni graceful shutdown").
    pub mcp_cancellation_token: CancellationToken,
}

pub type SharedState = Arc<AppState>;
