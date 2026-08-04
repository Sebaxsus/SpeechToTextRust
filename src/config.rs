//! Configuración del proceso, cargada una sola vez desde variables de entorno (`.env` opcional
//! vía `dotenvy`, ver `main.rs`). Todos los defaults son un calco exacto de los valores que
//! estaban hardcodeados antes de este módulo — sin ningún `.env` presente, el comportamiento es
//! idéntico al de siempre (ver `docs/configuracion.md` para el mapeo completo variable → default
//! → archivo:línea original).
//!
//! Dos formas de acceso, a propósito (ver `docs/configuracion.md`):
//! - `config::get()` (este módulo): para funciones libres que hoy no reciben ningún parámetro de
//!   estado (ej. `audio_pipeline::job`), evita agregarles un parámetro nuevo solo para esto.
//! - `&Config`/sub-structs pasados como parámetro explícito: para funciones que ya son
//!   parametrizadas (`WhisperRunner::new`, `rag::retrieval::search`, etc.) — se les agrega un
//!   parámetro más, en vez de depender del accessor global, para que sigan siendo testeables sin
//!   estado oculto.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::OnceLock;

/// Hop de Whisper: 10ms @ 16kHz. `CHUNK_SECONDS`/`OVERLAP_SECONDS` tienen que producir un número
/// de samples múltiplo de esto (ver CLAUDE.local.md: "Context Aware Chunking") o Whisper recibe
/// una ventana que no calza con su log-mel spectrogram.
const WHISPER_HOP_SAMPLES: usize = 160;
const SAMPLE_RATE_HZ: f32 = 16_000.0;

#[derive(Debug, Clone)]
pub struct PathsConfig {
    pub jobs_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub whisper_model_path: PathBuf,
    pub ffmpeg_bin: String,
    pub ffprobe_bin: String,
}

#[derive(Debug, Clone)]
pub struct ServicesConfig {
    pub ollama_host: String,
    pub ollama_port: u16,
    pub qdrant_url: String,
    pub bind_addr: String,
    /// Origen permitido para CORS (`Access-Control-Allow-Origin`) — necesario para que el
    /// cliente web (Astro, corriendo en un origen/puerto distinto) pueda llamar la API directo
    /// desde `fetch()` del browser. Default `http://localhost:4321`, el puerto default del modo
    /// dev de Astro. Ver `router::crear_router`.
    pub client_origin: String,
}

/// Calco 1:1 de los `set_*` de `FullParams` en `audio_pipeline::whisper_runner::build_params` —
/// ver ahí el porqué de cada valor (afinados junto con el fix del bug de alucinación
/// "(Portuguesa)", 2026-07-31). Cambiar estos valores puede degradar la precisión de la
/// transcripción — la prioridad #1 del proyecto — no son "solo performance".
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub entropy_thold: f32,
    pub suppress_blank: bool,
    pub suppress_nst: bool,
    pub temperature: f32,
    pub temperature_inc: f32,
    pub logprob_thold: f32,
    pub no_speech_thold: f32,
    pub single_segment: bool,
    pub token_timestamps: bool,
    pub split_on_word: bool,
    pub n_threads: i32,
    pub initial_prompt: String,
}

/// `chunk_seconds`/`overlap_seconds` en segundos (no samples, para que quien edite `.env` no
/// tenga que hacer la cuenta) — `Config::load` los valida contra el hop de Whisper (ver
/// `validar_chunking`). `SAMPLE_RATE_OUT` (16kHz) NO está acá: es un requisito duro del modelo,
/// nunca configurable, sigue fijo en `audio_pipeline::decoder`.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub chunk_seconds: f32,
    pub overlap_seconds: f32,
    pub resampler_chunk_size: usize,
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub embedding_model: String,
    pub generation_model: String,
    pub reranker_model_id: String,
    pub reranker_revision: String,
    pub top_k: u64,
    pub search_top_k: u64,
    pub rerank_candidate_pool: u64,
    pub low_confidence_thold: f32,
    pub batch_max_chunks: usize,
    pub batch_max_chars: usize,
    pub batch_num_ctx: u64,
    pub batch_num_predict: i32,
    pub consolidation_num_predict: i32,
    pub rag_answer_num_ctx: u64,
    pub max_segment_seconds: f32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub paths: PathsConfig,
    pub services: ServicesConfig,
    pub whisper: WhisperConfig,
    pub chunking: ChunkingConfig,
    pub rag: RagConfig,
    /// Igual semántica que antes de este módulo: `None` = `/mcp` y los endpoints REST protegidos
    /// quedan sin autenticación (ver `router::exigir_bearer_token`), no es un error de parseo.
    pub mcp_bearer_token: Option<String>,
}

/// `Ok(default)` si la variable no está seteada; `Err` con mensaje claro si está seteada pero no
/// parsea al tipo esperado — a propósito distinto del `.ok()` que se usa para `MCP_BEARER_TOKEN`
/// (ese es opcional por diseño, esto es "el usuario escribió algo inválido a propósito o por
/// error, hay que decírselo y no arrancar con un valor que no pidió").
fn env_or_default<T>(key: &str, default: T) -> anyhow::Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(key) {
        Ok(val) => val
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("variable de entorno {key}='{val}' inválida: {e}")),
        Err(_) => Ok(default),
    }
}

/// Ver "Context Aware Chunking" en `CLAUDE.local.md`: el chunk/overlap tienen que ser múltiplos
/// exactos del hop de 10ms de Whisper, o el log-mel spectrogram queda mal alineado.
fn validar_chunking(chunking: &ChunkingConfig) -> anyhow::Result<()> {
    if chunking.overlap_seconds >= chunking.chunk_seconds {
        anyhow::bail!(
            "OVERLAP_SECONDS ({}) debe ser menor que CHUNK_SECONDS ({})",
            chunking.overlap_seconds,
            chunking.chunk_seconds
        );
    }

    let chunk_samples = (chunking.chunk_seconds * SAMPLE_RATE_HZ).round() as usize;
    let overlap_samples = (chunking.overlap_seconds * SAMPLE_RATE_HZ).round() as usize;

    if chunk_samples == 0 || !chunk_samples.is_multiple_of(WHISPER_HOP_SAMPLES) {
        anyhow::bail!(
            "CHUNK_SECONDS={} no da un número de samples múltiplo del hop de 10ms de Whisper \
             ({WHISPER_HOP_SAMPLES} samples/hop @ 16kHz) — ver 'Context Aware Chunking' en \
             CLAUDE.local.md",
            chunking.chunk_seconds
        );
    }
    if !overlap_samples.is_multiple_of(WHISPER_HOP_SAMPLES) {
        anyhow::bail!(
            "OVERLAP_SECONDS={} no da un número de samples múltiplo del hop de 10ms de Whisper \
             ({WHISPER_HOP_SAMPLES} samples/hop @ 16kHz) — ver 'Context Aware Chunking' en \
             CLAUDE.local.md",
            chunking.overlap_seconds
        );
    }

    Ok(())
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let paths = PathsConfig {
            jobs_dir: env_or_default("JOBS_DIR", PathBuf::from("./jobs"))?,
            logs_dir: env_or_default("LOGS_DIR", PathBuf::from("./logs"))?,
            whisper_model_path: env_or_default(
                "WHISPER_MODEL_PATH",
                PathBuf::from("models/ggml-small-q5_1.bin"),
            )?,
            ffmpeg_bin: env_or_default("FFMPEG_BIN", "ffmpeg".to_string())?,
            ffprobe_bin: env_or_default("FFPROBE_BIN", "ffprobe".to_string())?,
        };

        let services = ServicesConfig {
            // `Ollama::new(host, port)` espera una URL completa en `host` (`into_url()` interno),
            // no un hostname suelto — ver ollama-rs 0.3.4, `impl Default for Ollama`.
            ollama_host: env_or_default("OLLAMA_HOST", "http://127.0.0.1".to_string())?,
            ollama_port: env_or_default("OLLAMA_PORT", 11434u16)?,
            qdrant_url: env_or_default("QDRANT_URL", "http://localhost:6334".to_string())?,
            bind_addr: env_or_default("SERVER_BIND_ADDR", "0.0.0.0:3000".to_string())?,
            client_origin: env_or_default("CLIENT_ORIGIN", "http://localhost:4321".to_string())?,
        };

        let whisper = WhisperConfig {
            entropy_thold: env_or_default("WHISPER_ENTROPY_THOLD", 2.3f32)?,
            suppress_blank: env_or_default("WHISPER_SUPPRESS_BLANK", true)?,
            suppress_nst: env_or_default("WHISPER_SUPPRESS_NST", true)?,
            temperature: env_or_default("WHISPER_TEMPERATURE", 0.0f32)?,
            temperature_inc: env_or_default("WHISPER_TEMPERATURE_INC", 0.1f32)?,
            logprob_thold: env_or_default("WHISPER_LOGPROB_THOLD", -0.8f32)?,
            no_speech_thold: env_or_default("WHISPER_NO_SPEECH_THOLD", 0.5f32)?,
            single_segment: env_or_default("WHISPER_SINGLE_SEGMENT", false)?,
            token_timestamps: env_or_default("WHISPER_TOKEN_TIMESTAMPS", false)?,
            split_on_word: env_or_default("WHISPER_SPLIT_ON_WORD", true)?,
            n_threads: env_or_default("WHISPER_N_THREADS", 8i32)?,
            initial_prompt: env_or_default(
                "WHISPER_INITIAL_PROMPT",
                "Transcripción de una reunión de trabajo en español, conversación natural."
                    .to_string(),
            )?,
        };

        let chunking = ChunkingConfig {
            chunk_seconds: env_or_default("CHUNK_SECONDS", 30.0f32)?,
            overlap_seconds: env_or_default("OVERLAP_SECONDS", 2.0f32)?,
            resampler_chunk_size: env_or_default("RESAMPLER_CHUNK_SIZE", 1024usize)?,
        };
        validar_chunking(&chunking)?;

        let rag = RagConfig {
            embedding_model: env_or_default("RAG_EMBEDDING_MODEL", "bge-m3".to_string())?,
            generation_model: env_or_default(
                "RAG_GENERATION_MODEL",
                "qcwind/qwen2.5-7b-instruct-Q4_K_M:latest".to_string(),
            )?,
            reranker_model_id: env_or_default(
                "RAG_RERANKER_MODEL_ID",
                "BAAI/bge-reranker-v2-m3".to_string(),
            )?,
            reranker_revision: env_or_default("RAG_RERANKER_REVISION", "main".to_string())?,
            top_k: env_or_default("RAG_TOP_K", 6u64)?,
            search_top_k: env_or_default("RAG_SEARCH_TOP_K", 8u64)?,
            rerank_candidate_pool: env_or_default("RAG_RERANK_CANDIDATE_POOL", 30u64)?,
            low_confidence_thold: env_or_default("RAG_LOW_CONFIDENCE_THOLD", -0.6f32)?,
            batch_max_chunks: env_or_default("RAG_SUMMARY_BATCH_MAX_CHUNKS", 50usize)?,
            batch_max_chars: env_or_default("RAG_SUMMARY_BATCH_MAX_CHARS", 12_000usize)?,
            batch_num_ctx: env_or_default("RAG_SUMMARY_BATCH_NUM_CTX", 8192u64)?,
            batch_num_predict: env_or_default("RAG_SUMMARY_BATCH_NUM_PREDICT", 300i32)?,
            consolidation_num_predict: env_or_default(
                "RAG_SUMMARY_CONSOLIDATION_NUM_PREDICT",
                500i32,
            )?,
            rag_answer_num_ctx: env_or_default("RAG_ANSWER_NUM_CTX", 4096u64)?,
            max_segment_seconds: env_or_default("MAX_SEGMENT_SECONDS", 120.0f32)?,
        };

        let mcp_bearer_token = std::env::var("MCP_BEARER_TOKEN").ok();

        Ok(Self {
            paths,
            services,
            whisper,
            chunking,
            rag,
            mcp_bearer_token,
        })
    }
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// Accessor global de solo lectura, para código que hoy no recibe ningún parámetro de estado (ver
/// doc del módulo). Falla rápido (imprime y sale del proceso) si la config es inválida — nunca
/// arranca con un valor a medias o silenciosamente distinto al que el usuario pidió en `.env`.
/// `eprintln!` (no `tracing::error!`): `main.rs` llama a `config::get()` ANTES de inicializar el
/// logger (el logger necesita `paths.logs_dir` de acá para saber dónde escribir), así que en este
/// punto no hay ningún `Subscriber` de `tracing` instalado todavía — un `tracing::error!` acá
/// sería un no-op silencioso.
pub fn get() -> &'static Config {
    CONFIG.get_or_init(|| {
        Config::load().unwrap_or_else(|e| {
            eprintln!("Config inválida: {e}");
            std::process::exit(1);
        })
    })
}
