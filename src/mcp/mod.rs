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
use crate::rag::{ScopeArg, hit_to_json, rag_answer as ejecutar_rag_answer, search};
use crate::state::SharedState;

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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetTranscriptRequest {
    pub audio_id: String,
}

fn error_interno(contexto: &str, err: anyhow::Error) -> McpError {
    tracing::error!("Error MCP ({contexto}): {err:?}");
    McpError::internal_error(contexto.to_string(), None)
}

/// Delega en `audio_pipeline::job::list_jobs` (compartido con `GET /api/jobs`, ver
/// `handlers::jobs_handler`) para no duplicar el `read_dir` + `load_job`.
async fn listar_jobs() -> anyhow::Result<Vec<JobMetadata>> {
    crate::audio_pipeline::job::list_jobs()
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
        // Mismo permiso que Whisper (ver CLAUDE.local.md: "NO paralelismo agresivo") — con
        // `scope: all_corpus` esto dispara el reranker cross-encoder (CPU-bound, `candle`), así
        // que nunca debe correr a la vez que una transcripción. Si Whisper está transcribiendo,
        // esta llamada espera acá (potencialmente minutos en un audio largo) en vez de competir
        // por los mismos threads de CPU.
        let _permit = self
            .state
            .heavy_compute_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let hits = search(
            &self.state.ollama,
            &self.state.qdrant,
            &query,
            &scope.into(),
            self.state.config.rag.search_top_k,
            &self.state.config.rag,
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
        // Mismo permiso que Whisper y search_transcript — acá además se suma la generación con
        // el modelo de 7-8B en Ollama, la carga más pesada de las dos rutas de RAG.
        let _permit = self
            .state
            .heavy_compute_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let respuesta = ejecutar_rag_answer(
            &self.state.ollama,
            &self.state.qdrant,
            &question,
            &scope.into(),
            &self.state.config.rag,
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

    #[tool(
        description = "Solo lectura: devuelve el transcript COMPLETO de un audio_id (todas las \
            entries de transcript.jsonl — chunk/start/end/text/avg_logprob y el flag \
            low_confidence ya calculado), NO un resumen. Pensada para que un cliente MCP externo \
            (otro asistente de IA) lea el texto completo de la reunión y genere su propio \
            resumen o responda preguntas libres sobre él, sin pasar por el Ollama local de este \
            servidor. Si el pipeline todavía no escribió transcript.jsonl, devuelve entries: [] \
            en vez de error. Nunca dispara ni modifica ninguna transcripción."
    )]
    async fn get_transcript(
        &self,
        Parameters(GetTranscriptRequest { audio_id }): Parameters<GetTranscriptRequest>,
    ) -> Result<CallToolResult, McpError> {
        let metadata = leer_job(&audio_id).await.map_err(|_| {
            McpError::invalid_params(format!("no existe el job '{audio_id}'"), None)
        })?;
        let response = crate::handlers::jobs_handler::construir_transcript_response(&metadata)
            .map_err(|e| error_interno("get_transcript", e))?;
        let payload = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
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
                 list_audios, get_audio_metadata, get_transcript (transcript completo, para que \
                 un modelo de IA externo genere su propio análisis/resumen). Nunca escribe ni \
                 borra en Qdrant, nunca dispara transcripciones nuevas."
                    .to_string(),
            )
    }
}

/// Construye el `Service` de Streamable HTTP a montar en el router de Axum (ver `router.rs`,
/// `nest_service("/mcp", ...)`). `state` se clona por sesión nueva: los clientes de Ollama/Qdrant
/// son livianos (no cargan modelos), así que esto no viola la regla de "nunca dejar Whisper/Ollama
/// residentes" (ver CLAUDE.local.md).
pub fn build_service(state: SharedState) -> StreamableHttpService<RagServer, LocalSessionManager> {
    let config = StreamableHttpServerConfig::default()
        .with_cancellation_token(state.mcp_cancellation_token.clone());
    StreamableHttpService::new(
        move || Ok(RagServer::new(state.clone())),
        Arc::new(LocalSessionManager::default()),
        config,
    )
}
