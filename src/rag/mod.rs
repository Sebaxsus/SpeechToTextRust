// Fase 5 (retrieval + generación RAG) todavía no tiene un caller fuera de los tests de
// integración `#[ignore]`: el consumidor real es el servidor MCP de Fase 6 (`rmcp`, no
// implementado aún — ver docs/TODO.md), no un endpoint Axum ad hoc que se adelante a ese
// diseño. Hasta que Fase 6 exista, este módulo es "código muerto" para `cargo build`/`clippy`
// fuera de `#[cfg(test)]` solo porque su único uso real todavía no está escrito, no porque esté
// de más. Quitar este `allow` cuando algo llame a `rag_answer`/`search` en el binario.
#![allow(dead_code, unused_imports)]

pub mod generation;
mod reranker;
pub mod retrieval;

pub use generation::rag_answer;
pub use retrieval::{ChunkHit, ChunkPayload, SearchScope, search};
