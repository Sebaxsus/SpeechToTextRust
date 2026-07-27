use std::collections::HashSet;

use ollama_rs::Ollama;
use ollama_rs::generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest};
use qdrant_client::qdrant::{Condition, Filter, GetPointsBuilder, PointId, QueryPointsBuilder};
use qdrant_client::{Payload, Qdrant};
use serde::Deserialize;

use super::reranker;
use crate::audio_pipeline::embeddings::{COLLECTION_NAME, EMBEDDING_MODEL, deterministic_point_id};

/// Cuando `scope = AllCorpus`, el bi-encoder trae este pool de candidatos (en vez de solo
/// `top_k`) para que el cross-encoder tenga margen real de reordenar antes de recortar a
/// `top_k` — rerankear un puñado de 6 hits ya casi-óptimos del coseno no vale la pena (ver
/// CLAUDE.local.md: por eso el reranker está gateado a este scope, no a `Audio`).
const RERANK_CANDIDATE_POOL: u64 = 30;

/// Alcance de una búsqueda semántica — nunca un `audio_id: Option<String>` (ver
/// CLAUDE.local.md: "Scope forzado, sin default implícito a todo el corpus"). El caller tiene
/// que construir `AllCorpus` explícitamente para salir del audio seleccionado; no hay ningún
/// camino donde "buscar en todo el corpus" ocurra por omisión.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchScope {
    Audio(String),
    AllCorpus,
}

impl SearchScope {
    fn filter(&self) -> Option<Filter> {
        match self {
            SearchScope::Audio(audio_id) => Some(Filter::must([Condition::matches(
                "audio_id",
                audio_id.clone(),
            )])),
            SearchScope::AllCorpus => None,
        }
    }
}

/// Un chunk tal como quedó persistido en el payload de Qdrant (ver CLAUDE.local.md: schema de
/// Fase 4) — deserializado directo del `Payload` devuelto por la búsqueda, sin passar por
/// `TranscriptEntry` (que no tiene `audio_id`/`speaker`).
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkPayload {
    pub audio_id: String,
    pub chunk_id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub speaker: String,
    pub avg_logprob: f32,
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub chunk: ChunkPayload,
    /// Similitud coseno del bi-encoder contra la query (`SearchScope::Audio`), o la relevancia
    /// del cross-encoder si `scope = AllCorpus` y el reranker corrió (ver `reranker.rs` — el
    /// score se **reemplaza**, no se promedia, cuando hay reranking). `None` en los chunks
    /// vecinos agregados por context assembly, que nunca pasaron por ninguna de las dos
    /// búsquedas.
    pub score: Option<f32>,
}

async fn embed_query(ollama: &Ollama, query: &str) -> anyhow::Result<Vec<f32>> {
    let request = GenerateEmbeddingsRequest::new(
        EMBEDDING_MODEL.to_string(),
        EmbeddingsInput::Single(query.to_string()),
    );
    let response = ollama.generate_embeddings(request).await?;
    response
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Ollama no devolvió ningún embedding para la query"))
}

/// Retrieval de Fase 5: embebe la query con el mismo modelo/colección que Fase 4, busca el
/// top-k dentro del `scope` pedido, y expande cada hit con su chunk vecino (`chunk_id ± 1`) para
/// mitigar oraciones cortadas en el borde de 30s (ver CLAUDE.local.md: "Context assembly").
///
/// Los vecinos se recuperan por ID determinístico (`deterministic_point_id`, mismo esquema que
/// Fase 4) vía `get_points` — una lookup exacta O(1) por ID, no una búsqueda vectorial extra. Si
/// un vecino no existe (borde del audio, o todavía no se embebió), `get_points` simplemente no lo
/// devuelve; no es un error.
///
/// Reranking (cross-encoder, `candle-transformers`, ver `reranker.rs`) corre acá cuando
/// `scope = AllCorpus`, sobre los hits crudos del bi-encoder y **antes** de la expansión de
/// vecinos — los vecinos son contexto agregado, no candidatos a rerankear. Nunca corre para
/// `SearchScope::Audio` (ver CLAUDE.local.md: con un solo audio el ranking por coseno ya es casi
/// exhaustivo, no justifica el costo de cargar el cross-encoder).
pub async fn search(
    ollama: &Ollama,
    qdrant: &Qdrant,
    query: &str,
    scope: &SearchScope,
    top_k: u64,
) -> anyhow::Result<Vec<ChunkHit>> {
    let query_vector = embed_query(ollama, query).await?;

    // Con AllCorpus traemos más candidatos de los que finalmente se usan, para darle al
    // reranker un pool real sobre el que reordenar (ver `RERANK_CANDIDATE_POOL`).
    let fetch_k = match scope {
        SearchScope::AllCorpus => top_k.max(RERANK_CANDIDATE_POOL),
        SearchScope::Audio(_) => top_k,
    };

    let mut request = QueryPointsBuilder::new(COLLECTION_NAME)
        .query(query_vector)
        .limit(fetch_k)
        .with_payload(true);
    if let Some(filter) = scope.filter() {
        request = request.filter(filter);
    }

    let response = qdrant.query(request).await?;

    let mut hits = Vec::with_capacity(response.result.len());
    for scored in response.result {
        let payload: Payload = scored.payload.into();
        let chunk: ChunkPayload = payload.deserialize()?;
        hits.push(ChunkHit {
            chunk,
            score: Some(scored.score),
        });
    }

    let mut hits = if *scope == SearchScope::AllCorpus {
        let mut reranked = reranker::rerank(query, hits).await;
        reranked.truncate(top_k as usize);
        reranked
    } else {
        hits
    };

    // `seen` se arma recién acá, sobre los hits finales post-rerank/truncate — no sobre el pool
    // de `fetch_k` candidatos crudos. Si se armara antes, un chunk que el reranker descartó (pero
    // que seguía en el pool ampliado) podría bloquear silenciosamente su propia inclusión como
    // vecino de un hit que sí sobrevivió, aunque ya no esté en `hits`.
    let mut seen: HashSet<(String, usize)> = hits
        .iter()
        .map(|hit| (hit.chunk.audio_id.clone(), hit.chunk.chunk_id))
        .collect();

    let mut neighbor_ids: Vec<PointId> = Vec::new();
    for hit in &hits {
        let audio_id = &hit.chunk.audio_id;
        let chunk_id = hit.chunk.chunk_id;
        if chunk_id > 0 && seen.insert((audio_id.clone(), chunk_id - 1)) {
            neighbor_ids.push(deterministic_point_id(audio_id, chunk_id - 1).into());
        }
        if seen.insert((audio_id.clone(), chunk_id + 1)) {
            neighbor_ids.push(deterministic_point_id(audio_id, chunk_id + 1).into());
        }
    }

    if !neighbor_ids.is_empty() {
        let neighbors = qdrant
            .get_points(GetPointsBuilder::new(COLLECTION_NAME, neighbor_ids).with_payload(true))
            .await?;

        for retrieved in neighbors.result {
            let payload: Payload = retrieved.payload.into();
            if let Ok(chunk) = payload.deserialize::<ChunkPayload>() {
                hits.push(ChunkHit { chunk, score: None });
            }
        }
    }

    hits.sort_by(|a, b| {
        a.chunk
            .audio_id
            .cmp(&b.chunk.audio_id)
            .then(a.chunk.chunk_id.cmp(&b.chunk.chunk_id))
    });

    Ok(hits)
}
