use std::sync::Arc;

// Por ahora está vacío, pero aquí vivirán:
// pub qdrant: QdrantClient
// pub ollama: Ollama
pub struct AppState {
    pub nombre_app: String,
}

pub type SharedState = Arc<AppState>;