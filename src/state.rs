use ollama_rs::Ollama;
use qdrant_client::Qdrant;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct AppState {
    pub nombre_app: String,
    // 1 permit: solo una transcripción pesada simultánea (ver CLAUDE.local.md: Concurrencia).
    pub transcription_semaphore: Arc<Semaphore>,
    // Clientes livianos (no cargan modelos, no son lo mismo que dejar Whisper/Ollama
    // residentes) — construidos una vez en main.rs y compartidos vía Arc<AppState>.
    pub ollama: Ollama,
    // Qdrant nunca se expone en la LAN — solo bindeado a localhost (ver CLAUDE.local.md).
    pub qdrant: Qdrant,
}

pub type SharedState = Arc<AppState>;
