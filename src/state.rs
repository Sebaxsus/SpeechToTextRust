use std::sync::Arc;
use tokio::sync::Semaphore;

// Por ahora está vacío, pero aquí vivirán:
// pub qdrant: QdrantClient
// pub ollama: Ollama
pub struct AppState {
    pub nombre_app: String,
    // 1 permit: solo una transcripción pesada simultánea (ver CLAUDE.local.md: Concurrencia).
    pub transcription_semaphore: Arc<Semaphore>,
}

pub type SharedState = Arc<AppState>;
