# Configuración vía `.env`

Toda la configuración del proceso (rutas, URLs de servicios, modelos, tuning de Whisper/RAG,
chunking) se puede personalizar sin tocar el código Rust, copiando `.env.example` a `.env` en la
raíz del repo y descomentando/editando lo que haga falta. Implementado en `src/config.rs`.

## Cómo funciona

- `.env` se carga una sola vez, en la primera línea de `main()` (vía `dotenvy::dotenv().ok()`) —
  si el archivo no existe, no pasa nada (es opcional).
- `Config::load()` lee cada variable con un default idéntico al valor que estaba hardcodeado
  antes de este módulo — **sin ningún `.env`, el comportamiento es exactamente el mismo que en un
  checkout limpio**.
- Si una variable está seteada pero no puede parsearse al tipo esperado (ej.
  `WHISPER_N_THREADS=banana`), el servidor no arranca: imprime el error y sale (`exit(1)`) en vez
  de arrancar con un valor a medias o distinto al que pediste. Esto es intencional — ver
  "Validación" más abajo.
- `CHUNK_SECONDS`/`OVERLAP_SECONDS` se validan contra el hop de 10ms de Whisper (ver "Context
  Aware Chunking" en `CLAUDE.local.md`) — un valor que no calce falla al arrancar, con un mensaje
  que explica por qué.
- **Los tests (`cargo test`) no cargan `.env`** — nunca llaman a `dotenvy::dotenv()` (solo lo hace
  `main()`), así que siempre ven los mismos defaults, sea cual sea tu `.env` local. Los literales
  hardcodeados dentro de bloques `#[cfg(test)]` (ej. la URL de Qdrant en varios tests) tampoco se
  tocaron — siguen siendo herméticos a propósito.
- Dos formas de acceso en el código (ver el doc de `src/config.rs`): `config::get()` (accessor
  global, para funciones que no reciben ningún parámetro de estado) y sub-structs pasados como
  parámetro explícito (`&WhisperConfig`, `&RagConfig` — para funciones que ya son parametrizadas,
  así siguen siendo testeables sin depender de estado oculto).

## Rutas

| Variable | Default | Dónde se lee | Qué controla |
|---|---|---|---|
| `JOBS_DIR` | `./jobs` | `audio_pipeline/job.rs` (`create_job`, `load_job`, `update_job_metadata`, `list_jobs`); `handlers/audio_handler.rs` (limpieza en error) | Directorio raíz de todos los jobs (`./jobs/{job_id}/...`). |
| `LOGS_DIR` | `./logs` | `main.rs` (init del logger) | Directorio de los logs rotados diariamente. |
| `WHISPER_MODEL_PATH` | `models/ggml-small-q5_1.bin` | `audio_pipeline/pipeline.rs` (`WhisperRunner::new`) | Ruta al modelo GGML de Whisper. |
| `FFMPEG_BIN` | `ffmpeg` | `audio_pipeline/decoder.rs` (fallback de decode); `handlers/jobs_handler.rs` (segmento de audio) | Nombre/ruta del binario `ffmpeg`. |
| `FFPROBE_BIN` | `ffprobe` | `audio_pipeline/decoder.rs` (fallback de decode) | Nombre/ruta del binario `ffprobe`. |

## Servicios y seguridad

| Variable | Default | Dónde se lee | Qué controla |
|---|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1` | `main.rs` (`Ollama::new`) | Host de Ollama (URL completa, no un hostname suelto). |
| `OLLAMA_PORT` | `11434` | `main.rs` | Puerto de Ollama. |
| `QDRANT_URL` | `http://localhost:6334` | `main.rs` (`Qdrant::from_url`) | URL gRPC de Qdrant. |
| `SERVER_BIND_ADDR` | `0.0.0.0:3000` | `main.rs` (bind del listener + logs de arranque) | Dirección/puerto donde escucha el servidor Axum. |
| `MCP_BEARER_TOKEN` | *(sin valor)* | `router.rs` (`crear_router_protegido`) | Token bearer de `/mcp` y los endpoints REST protegidos. Sin esto, esas rutas quedan sin autenticación. |
| `MCP_ALLOWED_HOSTS` | *(vacío)* | `mcp/mod.rs` (`build_service`) | Hosts (`host` o `host:puerto`, separados por coma) que se suman a los defaults de `allowed_hosts` de `rmcp` (`localhost`/`127.0.0.1`/`::1`, protección anti DNS-rebinding). Sin agregar acá la IP:puerto de LAN real del servidor, un cliente que conecte por esa IP recibe `403` en `/mcp`. |

## Whisper (ver advertencia en `.env.example` antes de tocar esto)

Todas se leen una sola vez en `WhisperRunner::new` (cacheadas en el struct, nunca releídas por
chunk) y afectan directamente la precisión de la transcripción — ver CLAUDE.local.md para el
porqué de cada default (afinados junto con el fix del bug de alucinación "(Portuguesa)").

| Variable | Default | Qué controla |
|---|---|---|
| `WHISPER_ENTROPY_THOLD` | `2.3` | `entropy_thold` de `FullParams`. |
| `WHISPER_SUPPRESS_BLANK` | `true` | `suppress_blank`. |
| `WHISPER_SUPPRESS_NST` | `true` | `suppress_nst` — parte del fix del bug "(Portuguesa)". |
| `WHISPER_TEMPERATURE` | `0.0` | `temperature`. |
| `WHISPER_TEMPERATURE_INC` | `0.1` | `temperature_inc`. |
| `WHISPER_LOGPROB_THOLD` | `-0.8` | `logprob_thold`. |
| `WHISPER_NO_SPEECH_THOLD` | `0.5` | `no_speech_thold`. |
| `WHISPER_SINGLE_SEGMENT` | `false` | `single_segment`. |
| `WHISPER_TOKEN_TIMESTAMPS` | `false` | `token_timestamps`. |
| `WHISPER_SPLIT_ON_WORD` | `true` | `split_on_word`. |
| `WHISPER_N_THREADS` | `8` | `n_threads`. |
| `WHISPER_INITIAL_PROMPT` | *(frase genérica en español, ver `.env.example`)* | `initial_prompt` — sesgo de idioma/registro. |
| `WHISPER_GREEDY_BEST_OF` | `5` | `best_of` de `SamplingStrategy::Greedy`. Verificado en `whisper.cpp` (vendorizado en `whisper-rs-sys`) que con `WHISPER_TEMPERATURE=0.0` no afecta el primer intento de ningún chunk — solo se usa en los reintentos a temperatura más alta que whisper.cpp dispara cuando ese primer intento falla `logprob_thold`+`no_speech_thold` (ver `WHISPER_TUNING_LOG.md`, 2026-08-06). |

Todas se leen en `audio_pipeline/whisper_runner.rs` (`WhisperRunner::build_params`).

## Chunking

| Variable | Default | Dónde se lee | Qué controla |
|---|---|---|---|
| `CHUNK_SECONDS` | `30` | `audio_pipeline/decoder.rs` (`StreamingDecoder::new`) | Duración de cada chunk de audio. Validado contra el hop de Whisper en `config.rs`. |
| `OVERLAP_SECONDS` | `2` | igual | Overlap entre chunks consecutivos. Igual validación. |
| `RESAMPLER_CHUNK_SIZE` | `1024` | igual | Tamaño de bloque de entrada del resampler `rubato`. |

`SAMPLE_RATE_OUT` (16kHz) sigue fijo en el código, nunca configurable — es un requisito duro de
Whisper, no un tuning.

## RAG (retrieval, generación, reranker, resumen)

| Variable | Default | Dónde se lee | Qué controla |
|---|---|---|---|
| `RAG_EMBEDDING_MODEL` | `bge-m3` | `audio_pipeline/embeddings.rs`, `rag/retrieval.rs` | Modelo de embeddings de Ollama. |
| `RAG_GENERATION_MODEL` | `qcwind/qwen2.5-7b-instruct-Q4_K_M:latest` | `rag/generation.rs`, `rag/summary.rs` | Modelo de generación de Ollama (RAG y resúmenes). |
| `RAG_RERANKER_MODEL_ID` | `BAAI/bge-reranker-v2-m3` | `rag/reranker.rs` | Repo de Hugging Face del cross-encoder. |
| `RAG_RERANKER_REVISION` | `main` | `rag/reranker.rs` | Revisión/branch del repo del reranker. |
| `RAG_TOP_K` | `6` | `rag/generation.rs` (`rag_answer`) | Top-k de contexto para la generación. |
| `RAG_SEARCH_TOP_K` | `8` | `handlers/rag_handler.rs`, `mcp/mod.rs` (`search_transcript`) | Top-k de un retrieval crudo (sin generación). |
| `RAG_RERANK_CANDIDATE_POOL` | `30` | `rag/retrieval.rs` (`search`) | Pool de candidatos que trae el bi-encoder antes de rerankear (solo `scope: all_corpus`). |
| `RAG_LOW_CONFIDENCE_THOLD` | `-0.6` | `rag/generation.rs`, `handlers/jobs_handler.rs` | Umbral de `avg_logprob` para marcar un chunk `low_confidence`/`[BAJA CONFIANZA]`. |
| `RAG_ANSWER_NUM_CTX` | `4096` | `rag/generation.rs` (`rag_answer`) | `num_ctx` de la llamada a Ollama en `rag_answer`. |
| `RAG_SUMMARY_BATCH_MAX_CHUNKS` | `50` | `rag/summary.rs` (`build_batches`) | Cuántos chunks entran en un lote del resumen map-reduce. |
| `RAG_SUMMARY_BATCH_MAX_CHARS` | `12000` | igual | Límite de caracteres por lote (red de seguridad). |
| `RAG_SUMMARY_BATCH_NUM_CTX` | `8192` | `rag/summary.rs` (`summarize_batch`/`consolidate_summaries`) | `num_ctx` de las llamadas de resumen. |
| `RAG_SUMMARY_BATCH_NUM_PREDICT` | `300` | `rag/summary.rs` (`summarize_batch`) | `num_predict` del resumen de cada lote. |
| `RAG_SUMMARY_CONSOLIDATION_NUM_PREDICT` | `500` | `rag/summary.rs` (`consolidate_summaries`) | `num_predict` de la consolidación final. |
| `MAX_SEGMENT_SECONDS` | `120` | `handlers/jobs_handler.rs` (`obtener_segmento_audio_handler`) | Duración máxima de un clip pedido a `GET /api/jobs/{job_id}/audio-segment`. |

## Validación

`Config::load()` falla rápido (imprime el error y sale del proceso) en dos casos:

1. Una variable está seteada pero no parsea al tipo esperado (ej. un booleano que no sea
   `true`/`false`, un número que no sea número).
2. `CHUNK_SECONDS`/`OVERLAP_SECONDS` no producen un número de samples múltiplo del hop de 10ms de
   Whisper, o `OVERLAP_SECONDS >= CHUNK_SECONDS`.

En ambos casos el mensaje de error dice qué variable y por qué — nunca arranca en un estado a
medias.
