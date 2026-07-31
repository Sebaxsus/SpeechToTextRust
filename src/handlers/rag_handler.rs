//! Wrappers REST de solo lectura sobre `rag::retrieval`/`rag::generation` (`POST /api/search`,
//! `POST /api/rag/answer`) — mismo shape de body que las tools MCP `search_transcript`/
//! `rag_answer`, cero lógica nueva de retrieval/generación. Montados bajo el mismo middleware de
//! bearer token que `/mcp` (ver `router.rs`), porque exponen el mismo contenido transcrito
//! sensible.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::Deserialize;

use crate::rag::{SEARCH_TOP_K, ScopeArg, hit_to_json, rag_answer, search};
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(flatten)]
    pub scope: ScopeArg,
}

#[derive(Debug, Deserialize)]
pub struct RagAnswerRequest {
    pub question: String,
    #[serde(flatten)]
    pub scope: ScopeArg,
}

/// `POST /api/search` — wrapper REST de la tool MCP `search_transcript`: retrieval puro, sin
/// generación. Adquiere `heavy_compute_semaphore` igual que la tool MCP porque `scope:
/// all_corpus` dispara el reranker cross-encoder (CPU-bound).
pub async fn buscar_handler(
    State(state): State<SharedState>,
    Json(SearchRequest { query, scope }): Json<SearchRequest>,
) -> impl IntoResponse {
    let _permit = state
        .heavy_compute_semaphore
        .clone()
        .acquire_owned()
        .await
        .unwrap();

    match search(
        &state.ollama,
        &state.qdrant,
        &query,
        &scope.into(),
        SEARCH_TOP_K,
    )
    .await
    {
        Ok(hits) => {
            let payload = serde_json::json!({
                "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>(),
            });
            Json(payload).into_response()
        }
        Err(e) => {
            eprintln!("Error en /api/search: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response()
        }
    }
}

/// `POST /api/rag/answer` — wrapper REST de la tool MCP `rag_answer`: retrieval + generación
/// server-side vía Ollama. Mismo semáforo que `buscar_handler` y que la tool MCP equivalente.
pub async fn rag_answer_handler(
    State(state): State<SharedState>,
    Json(RagAnswerRequest { question, scope }): Json<RagAnswerRequest>,
) -> impl IntoResponse {
    let _permit = state
        .heavy_compute_semaphore
        .clone()
        .acquire_owned()
        .await
        .unwrap();

    match rag_answer(&state.ollama, &state.qdrant, &question, &scope.into()).await {
        Ok(respuesta) => Json(serde_json::json!({ "answer": respuesta })).into_response(),
        Err(e) => {
            eprintln!("Error en /api/rag/answer: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response()
        }
    }
}
