//! Servidor MCP de solo lectura (Fase 6, ver CLAUDE.local.md: "MCP de solo lectura") montado
//! sobre el mismo Axum vía transporte Streamable HTTP (`rmcp`, feature `transport-streamable-http-server`).
//! Ninguna tool de acá escribe ni borra en Qdrant, ni dispara transcripciones nuevas — todas
//! delegan en `rag::retrieval`/`rag::generation` (Fase 5) o leen `job.json` de disco.

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, ContentBlock, Implementation, ProtocolVersion, ServerCapabilities,
        ServerInfo,
    },
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Deserialize;

use crate::audio_pipeline::models::JobMetadata;
use crate::rag::{ChunkHit, SearchScope, rag_answer as ejecutar_rag_answer, search};
use crate::state::SharedState;

/// Top-k de `search_transcript`: más generoso que el `TOP_K = 6` de `rag::generation` porque acá
/// el caller ve los chunks crudos (sin generación) y puede querer explorar más contexto.
const SEARCH_TOP_K: u64 = 8;

/// Scope de búsqueda expuesto a clientes MCP — refleja 1:1 `rag::retrieval::SearchScope` (ver
/// CLAUDE.local.md: "Scope forzado explícitamente, sin default implícito a todo el corpus").
/// Al serializar con `#[serde(tag = "scope")]` no existe ningún valor por omisión: el caller
/// tiene que mandar `"scope": "audio"` + `"audio_id": "..."` o `"scope": "all_corpus"` de forma
/// explícita — no hay un `audio_id: Option<String>` que pueda quedar en `None` por descuido.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum ScopeArg {
    Audio { audio_id: String },
    AllCorpus,
}

impl From<ScopeArg> for SearchScope {
    fn from(scope: ScopeArg) -> Self {
        match scope {
            ScopeArg::Audio { audio_id } => SearchScope::Audio(audio_id),
            ScopeArg::AllCorpus => SearchScope::AllCorpus,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchTranscriptRequest {
    pub query: String,
    #[serde(flatten)]
    pub scope: ScopeArg,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RagAnswerRequest {
    pub question: String,
    #[serde(flatten)]
    pub scope: ScopeArg,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetAudioMetadataRequest {
    pub audio_id: String,
}

fn hit_to_json(hit: &ChunkHit) -> serde_json::Value {
    serde_json::json!({
        "audio_id": hit.chunk.audio_id,
        "chunk_id": hit.chunk.chunk_id,
        "start": hit.chunk.start,
        "end": hit.chunk.end,
        "text": hit.chunk.text,
        "speaker": hit.chunk.speaker,
        "avg_logprob": hit.chunk.avg_logprob,
        "score": hit.score,
    })
}

fn error_interno(contexto: &str, err: anyhow::Error) -> McpError {
    eprintln!("Error MCP ({contexto}): {err:?}");
    McpError::internal_error(contexto.to_string(), None)
}

/// Enumera `./jobs/*` y delega en `audio_pipeline::job::load_job` (mismo loader validado que usa
/// el endpoint de resume, ver `handlers::audio_handler::reanudar_job`) para leer cada `job.json`
/// — un directorio sin `job.json` legible se saltea con un log en vez de abortar el listado
/// completo. Los nombres de directorio ya son UUIDs generados por `create_job`, así que la
/// validación de UUID de `load_job` nunca falla acá; se reusa igual para no duplicar el parseo.
async fn listar_jobs() -> anyhow::Result<Vec<JobMetadata>> {
    let mut entradas = tokio::fs::read_dir("./jobs").await?;
    let mut jobs = Vec::new();
    while let Some(entrada) = entradas.next_entry().await? {
        if !entrada.file_type().await?.is_dir() {
            continue;
        }
        let job_id = entrada.file_name().to_string_lossy().into_owned();
        match crate::audio_pipeline::job::load_job(&job_id) {
            Ok(metadata) => jobs.push(metadata),
            Err(e) => eprintln!("No se pudo leer el job '{job_id}': {e}"),
        }
    }
    Ok(jobs)
}

/// `audio_id` viene directo de un argumento de tool MCP (no confiable) — `load_job` valida que
/// sea un UUID bien formado antes de usarlo en una ruta de disco, así que acá no hace falta
/// repetir esa validación (a diferencia de la versión anterior de esta función, que construía la
/// ruta a mano sin validar nada).
async fn leer_job(audio_id: &str) -> anyhow::Result<JobMetadata> {
    crate::audio_pipeline::job::load_job(audio_id)
}

#[derive(Clone)]
pub struct RagServer {
    state: SharedState,
    // `#[tool_handler]` genera `call_tool`/`list_tools` llamando a `Self::tool_router()` fresco
    // en cada request (no lee este campo) — se guarda igual, por paridad con el patrón de
    // referencia de `rmcp` (examples/servers/src/common/counter.rs), donde otro código (tests,
    // futuras extensiones) puede necesitar la instancia ya construida.
    #[allow(dead_code)]
    tool_router: ToolRouter<RagServer>,
}

#[tool_router]
impl RagServer {
    fn new(state: SharedState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Búsqueda semántica de solo lectura sobre las transcripciones ya \
            embebidas (Fase 4/5): NO genera una respuesta, solo devuelve los chunks más \
            relevantes con su score y su texto. El scope es obligatorio: \
            {\"scope\":\"audio\",\"audio_id\":\"...\"} para un audio puntual, o \
            {\"scope\":\"all_corpus\"} para buscar en todas las reuniones (activa el reranker \
            de cross-encoder sobre el pool de candidatos)."
    )]
    async fn search_transcript(
        &self,
        Parameters(SearchTranscriptRequest { query, scope }): Parameters<SearchTranscriptRequest>,
    ) -> Result<CallToolResult, McpError> {
        let hits = search(
            &self.state.ollama,
            &self.state.qdrant,
            &query,
            &scope.into(),
            SEARCH_TOP_K,
        )
        .await
        .map_err(|e| error_interno("search_transcript", e))?;

        let payload = serde_json::json!({
            "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>(),
        });
        Ok(CallToolResult::success(vec![ContentBlock::text(
            payload.to_string(),
        )]))
    }

    #[tool(
        description = "Retrieval + generación server-side vía Ollama (Fase 5) sobre las \
            transcripciones — el motor de RAG por defecto del proyecto (Gemini Nano, si se usa, \
            corre client-side en el browser y no pasa por acá). Mismo scope obligatorio que \
            search_transcript."
    )]
    async fn rag_answer(
        &self,
        Parameters(RagAnswerRequest { question, scope }): Parameters<RagAnswerRequest>,
    ) -> Result<CallToolResult, McpError> {
        let respuesta = ejecutar_rag_answer(
            &self.state.ollama,
            &self.state.qdrant,
            &question,
            &scope.into(),
        )
        .await
        .map_err(|e| error_interno("rag_answer", e))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(respuesta)]))
    }

    #[tool(
        description = "Lista todos los audios/jobs procesados, leyendo el job.json de cada uno."
    )]
    async fn list_audios(&self) -> Result<CallToolResult, McpError> {
        let jobs = listar_jobs()
            .await
            .map_err(|e| error_interno("list_audios", e))?;
        let payload = serde_json::to_string(&jobs).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![ContentBlock::text(payload)]))
    }

    #[tool(
        description = "Devuelve el job.json completo (status, rutas, created_at) de un audio_id puntual."
    )]
    async fn get_audio_metadata(
        &self,
        Parameters(GetAudioMetadataRequest { audio_id }): Parameters<GetAudioMetadataRequest>,
    ) -> Result<CallToolResult, McpError> {
        let metadata = leer_job(&audio_id).await.map_err(|_| {
            McpError::invalid_params(format!("no existe el job '{audio_id}'"), None)
        })?;
        let payload = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
        Ok(CallToolResult::success(vec![ContentBlock::text(payload)]))
    }
}

#[tool_handler]
impl ServerHandler for RagServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(
                "Servidor MCP de solo lectura sobre transcripciones RAG local-first. Tools: \
                 search_transcript (retrieval), rag_answer (retrieval + generación vía Ollama), \
                 list_audios, get_audio_metadata. Nunca escribe ni borra en Qdrant, nunca \
                 dispara transcripciones nuevas."
                    .to_string(),
            )
    }
}

/// Construye el `Service` de Streamable HTTP a montar en el router de Axum (ver `router.rs`,
/// `nest_service("/mcp", ...)`). `state` se clona por sesión nueva: los clientes de Ollama/Qdrant
/// son livianos (no cargan modelos), así que esto no viola la regla de "nunca dejar Whisper/Ollama
/// residentes" (ver CLAUDE.local.md).
pub fn build_service(state: SharedState) -> StreamableHttpService<RagServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(RagServer::new(state.clone())),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    )
}
