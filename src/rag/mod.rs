pub mod generation;
mod reranker;
pub mod retrieval;
pub mod summary;

pub use generation::rag_answer;
pub use retrieval::{SEARCH_TOP_K, ScopeArg, hit_to_json, search};
pub use summary::generate_summary;

/// Umbral de "baja confianza" sobre `avg_logprob` — compartido entre el aviso `[BAJA CONFIANZA]`
/// que `rag::generation` agrega al contexto del RAG y el campo `low_confidence` calculado por
/// `handlers::jobs_handler::obtener_transcript_handler`. Un solo lugar para este número evita que
/// el frontend tenga que duplicarlo, y cambiarlo no requiere reprocesar nada (se recalcula en
/// cada lectura). No tiene relación con `logprob_thold` de whisper.cpp (`FullParams`, ver
/// `audio_pipeline::whisper_runner`) — ese gatea si Whisper reintenta/descarta un chunk durante la
/// transcripción; este solo decide cuándo señalar un chunk ya transcrito para revisión manual.
/// Bajado de -0.8 a -0.6 (2026-08-01, decisión explícita del usuario) para flaguear más segmentos
/// dado que el dataset real tiene mucho cross-talk/ruido de fondo.
pub(crate) const LOW_CONFIDENCE_THOLD: f32 = -0.6;
