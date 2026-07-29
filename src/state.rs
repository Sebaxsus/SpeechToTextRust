use ollama_rs::Ollama;
use qdrant_client::Qdrant;
use std::sync::Arc;
use tokio::sync::Semaphore;

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
}

pub type SharedState = Arc<AppState>;
