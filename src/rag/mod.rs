pub mod generation;
mod reranker;
pub mod retrieval;
pub mod summary;

pub use generation::rag_answer;
pub use retrieval::{ScopeArg, hit_to_json, search};
pub use summary::generate_summary;

// El umbral de "baja confianza" (antes `LOW_CONFIDENCE_THOLD` acá) y el top-k de retrieval crudo
// (antes `SEARCH_TOP_K`) viven ahora en `config::RagConfig` (`low_confidence_thold`/
// `search_top_k`, ver `docs/configuracion.md`) — mismo rol de "un solo lugar para este número",
// ahora configurable vía `.env` en vez de const fija.
