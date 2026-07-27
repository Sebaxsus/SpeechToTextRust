# TODO

Pendientes detectados en el estado actual del código, priorizados de forma aproximada (bloqueantes primero). Ver `docs/Arquitechture.md` para el contexto de cada fase y `CLAUDE.local.md` para los detalles de parámetros/decisiones ya fijadas.

## Completado — Fase 1 (Upload)

- [x] Validar formato de archivo (mp3/mp4, incluyendo `.m4a` como mp4) por magic bytes, antes de escribir a disco.
- [x] Implementar `audio_pipeline::job::create_job`: Job ID, carpeta propia `./jobs/{job_id}/`, rutas fijas, `job.json` con metadata inicial.
- [x] Eliminar el riesgo de path traversal: el nombre de archivo del cliente ya no se usa como parte de ninguna ruta de disco (en vez de "sanearlo", se dejó de usar).

## Completado — Fase 2 (Whisper/Symphonia)

- [x] Implementar `StreamingDecoder`: demux con Symphonia, solo track de audio (nunca video), resample a PCM mono 16kHz con `rubato` (sinc, anti-aliasing).
- [x] Chunking de 30s con overlap de 2s y crossfade (Hann) en los bordes internos, respetando múltiplos del hop size de Whisper.
- [x] Implementar `WhisperRunner`: carga el modelo GGML, aplica el `FullParams` de `CLAUDE.local.md`, mantiene un único `WhisperState` por job para continuidad de contexto entre chunks.
- [x] Confirmado: la llamada real a Whisper corre dentro de `spawn_blocking` (heredado desde `audio_handler.rs`).
- [x] Fallback a `ffmpeg`/`ffprobe` para códecs que Symphonia no soporta (AMR-NB, el códec real de las grabaciones de `sample_Media/` — ver `CLAUDE.local.md`).

## Importante — Precisión / tdrz

El modelo cambió de `ggml-base-tdrz-q5_1.bin` (no se consiguió) a `ggml-small-q5_1.bin`, que no tiene variante tdrz — todo este bloque queda en espera hasta que se retome con un modelo tdrz.

- [ ] Si se consigue un modelo tdrz en el futuro: decidir si se activa `set_tdrz_enable(true)`. Si se activa:
  - [ ] Mantener un contador de `speaker_id` / timestamp del último turno confirmado en el estado del *job* (no en el scope de cada llamada a `full()`).
  - [ ] Implementar dedupe de turnos en la ventana de overlap entre chunks (umbral mínimo ~0.5-1s entre turnos, tratar el overlap como una sola zona de decisión).
  - [ ] Probar calidad empíricamente en español — tdrz solo está validado oficialmente sobre `small.en-tdrz` (inglés).
- [ ] Evaluar `set_suppress_nst(true)` para reducir alucinaciones en silencios, complementando `no_speech_thold`.
- [ ] Evaluar VAD real (`set_vad_model_path` / `enable_vad`) para saltar tramos de silencio en audios de 5h y bajar CPU, sin perder tramos con voz baja.

## Fase 3 — Persistencia

- [x] Implementar `JsonlWriter::append`: una línea JSON + flush por llamada.
- [x] Implementar `CheckpointManager` (`new`/`load`/`save`): persistencia de `(last_chunk, processed_seconds)`.
- [x] Corregir que `chunk_index` en `StreamingDecoder` se reinicia en 0 tras un `seek_seconds` real — `seek_seconds` ahora recibe `resume_from_chunk` (`checkpoint.last_chunk + 1`) y lo aplica dentro del guard de `seconds <= 0.0`, así que un job nuevo (sin checkpoint) no se ve afectado.
- [x] Test de resume a nivel decoder (sin Whisper, sin `sample_Media/`): genera un WAV PCM16 sintético en `std::env::temp_dir()` y verifica que `seek_seconds(seconds, resume_from_chunk)` retoma la numeración de chunk correcta y que `seconds <= 0.0` sigue siendo no-op (`audio_pipeline::decoder::tests`).
- [ ] Probar el flujo de recuperación end-to-end con audio real: matar el proceso a mitad de un audio de varias horas y confirmar que retoma desde el checkpoint sin re-transcribir desde el principio (pendiente porque requiere `sample_Media/` + el modelo GGML, igual que el test `#[ignore]` de `router.rs`).

## Fase 4 — Embeddings + Qdrant (implementado y probado contra servicios reales)

Diseño cerrado en `CLAUDE.local.md` (secciones Ollama/Qdrant) y `docs/Arquitechture.md` (Fase 4) — ver ahí el razonamiento completo. Resumen accionable:

- [x] Agregar `avg_logprob: f32` a `TranscriptEntry` (`models.rs`) y calcularlo en `WhisperRunner::transcribe_chunk` (ahora devuelve `(String, f32)`: texto + promedio de `ln(token_probability())` sobre todos los tokens del chunk; `0.0` si el chunk no tiene tokens) — precondición para poblar el payload de Qdrant.
- [x] Módulo `audio_pipeline/embeddings.rs`:
  - [x] `ensure_collection`: crea la colección única `transcripts` si no existe (`Distance::Cosine`, vectors+payload `on_disk: true`, sin quantization, índice de payload `Keyword` sobre `audio_id`).
  - [x] `run_embedding_phase`: lee `transcript.jsonl` línea por línea (streaming, `Peekable` sobre `BufReader::lines()`), genera un embedding por línea con Ollama (`bge-m3`), hace upsert en Qdrant con point ID determinístico (`Uuid::new_v5(Uuid::NAMESPACE_OID, "audio_id:chunk_id")`) — idempotente ante reintentos, no requiere checkpoint propio.
  - [x] Salta chunks con `text` vacío (silencio/no-speech) — no vale la pena embeberlos.
  - [x] `keep_alive`: default (Ollama gestiona su propio timeout corto) en cada llamada intermedia, `KeepAlive::UnloadOnCompletion` explícito en la última línea del job (detectado con `Peekable::peek()`).
- [x] `run_embedding_phase` corre como tarea async normal (no `spawn_blocking`) en `audio_handler.rs`, **después** de que el `spawn_blocking` de `run_pipeline` (Fase 2/3) resuelve `Ok(Ok(()))` — nunca se solapa con Whisper.
- [x] `AppState` gana clientes `ollama: Ollama` / `qdrant: Qdrant` (construidos una vez en `main.rs` con `Ollama::default()` y `Qdrant::from_url("http://localhost:6334").build()` — clientes livianos, no cargan modelos).
- [x] Qdrant bindeado solo a localhost (`http://localhost:6334`) — nunca expuesto directo en la LAN (ver Fase 6).
- [x] Probado end-to-end contra servicios reales: `bge-m3:latest` en Ollama y un Qdrant real (Docker, `qdrant/qdrant`, bindeado a `127.0.0.1` únicamente — nunca a `0.0.0.0`, ver CLAUDE.local.md). Test de integración `audio_pipeline::embeddings::tests::run_embedding_phase_contra_servicios_reales` (`#[ignore]` porque depende de esos servicios vivos): confirma que el chunk de texto vacío se salta, que se insertan exactamente los puntos esperados (`count` filtrado por `audio_id`), y limpia sus propios puntos de prueba al terminar. Correr manualmente con `cargo test embeddings -- --ignored --nocapture` una vez que Ollama y Qdrant estén arriba.
- [ ] Levantar Qdrant no está automatizado todavía (hoy es un `docker run` manual) — documentar en el README o en un script de setup, junto con el pull de `bge-m3` en Ollama.
- [ ] **Gap conocido, documentado deliberadamente sin resolver todavía**: si `run_embedding_phase` falla a mitad de un audio largo (Ollama caído, red, etc.) sin que el proceso completo crashee, hoy no hay una forma automática de reintentar solo esa fase para ese job — toca volver a invocar la función manualmente (es segura de reintentar, ver idempotencia arriba). Una cola de reintentos o un escaneo de jobs con transcript completo pero sin embeddings es trabajo futuro, no bloqueante para la primera versión.

## Fase 5 — RAG: retrieval, reranking y generación (implementado y probado end-to-end)

Ver `CLAUDE.local.md` (sección "RAG — scope, retrieval y reranking" y "Generación — Ollama por defecto, Gemini Nano opcional") y `docs/Arquitechture.md` (Fase 5). Módulo nuevo: `src/rag/` (`retrieval.rs`, `reranker.rs`, `generation.rs`) — separado de `audio_pipeline/` porque opera sobre datos ya embebidos en Qdrant, no sobre audio.

- [x] `SearchScope` (`Audio(audio_id)` / `AllCorpus`, sin default implícito) definido en `rag::retrieval` y usado por `search`/`rag_answer` — nunca un `audio_id: Option<String>`.
- [x] Retrieval (`rag::retrieval::search`): embebe la query con `bge-m3` (mismo modelo/colección que Fase 4), busca vía la Query API de Qdrant (`qdrant.query`, no el `search_points` legacy que la crate marca para remover), filtrado por `scope`. Con `AllCorpus` trae un pool de 30 candidatos (no solo `top_k`) para dejarle margen real al reranker antes de recortar a `top_k` (default 6, rango 5-8). Expande cada hit final con su chunk vecino (`chunk_id ± 1`) recuperado por **ID determinístico** (`get_points`, mismo esquema `Uuid::new_v5` que Fase 4) — lookup exacta O(1), no una segunda búsqueda vectorial. Dedupe vía un `HashSet<(audio_id, chunk_id)>` armado **después** del rerank/truncado (no antes — si se arma antes, un candidato descartado por el reranker podía bloquear silenciosamente su propia inclusión como vecino de un hit que sí sobrevivió).
- [x] **Reranker cross-encoder implementado** (`rag::reranker`, gateado a `scope = AllCorpus` únicamente): `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa-large + cabeza de clasificación de 1 salida) corriendo in-process con `candle-transformers` 0.11.0 (`XLMRobertaForSequenceClassification`), basado en el ejemplo oficial `candle-examples/examples/xlm-roberta` (tarea `Reranker`). Pesos en **f32** (no f16 como el ejemplo de referencia — CPU-only y accuracy es la prioridad #1, no vale la pena arriesgar precisión para ahorrar la mitad de la RAM transitoria). Carga transiente por llamada (mmap vía `VarBuilder::from_mmaped_safetensors`, nunca reside en `AppState`), corre dentro de `spawn_blocking` (CPU-bound, igual que Whisper). **Degrada sin romper la query**: si falla (sin red la primera vez, cache corrupta, etc.) loguea el error y devuelve el orden del bi-encoder en vez de propagar el fallo.
  - Dependencias nuevas: `candle-core`/`candle-nn`/`candle-transformers` 0.11.0, `hf-hub` 0.5.0, `tokenizers` 0.22.0 **con `default-features = false` y sin `esaxx_fast`** — ese feature compila una dependencia C++ con runtime estático (`/MT`) que en Windows/MSVC choca en link time (`LNK2038`) con el build CMake de `whisper-rs-sys` (runtime dinámico `/MD`). Solo afecta *entrenar* un tokenizer Unigram desde cero, no cargar uno ya entrenado — sin pérdida funcional.
  - **Nueva excepción de red deliberada** (documentada igual que `ffmpeg`/`ffprobe` y el primer `ollama pull`): la primera vez que corre `SearchScope::AllCorpus`, `hf-hub` descarga `BAAI/bge-reranker-v2-m3` (~2.2GB en f32) de Hugging Face Hub y lo cachea en `~/.cache/huggingface/hub`. Corridas siguientes son 100% offline.
  - [x] Probado: `rag::reranker::tests::rerank_reordena_hits_por_relevancia_real` (`#[ignore]`, no depende de Qdrant/Ollama — es una prueba aislada del cross-encoder sobre `ChunkHit` en memoria) confirma que reordena un pasaje realmente relevante al primer lugar aunque el score de bi-encoder que traía fuera deliberadamente el más bajo. `cargo test rerank_reordena -- --ignored --nocapture`.
- [x] Generación (`rag::generation::rag_answer`): default Ollama server-side, `qcwind/qwen2.5-7b-instruct-Q4_K_M:latest`, para ambos scopes. Arma el contexto con `[audio | chunk | start-end]` + un aviso `[BAJA CONFIANZA]` cuando `avg_logprob < -0.8` (mismo umbral que `logprob_thold` de Whisper), para que el modelo module su respuesta en tramos de cross-talk/ruido en vez de citarlos como transcripción limpia. `keep_alive: UnloadOnCompletion` — el modelo de 7-8B no queda residente entre preguntas sueltas.
- [x] Probado end-to-end contra servicios reales: `rag::generation::tests::rag_answer_contra_servicios_reales` (`#[ignore]`) siembra una transcripción sintética vía `run_embedding_phase` (reutiliza Fase 4), corre `rag_answer` sobre ese `audio_id`, verifica que la respuesta menciona el dato recuperado del contexto, y limpia sus puntos de prueba. Corrida manual: `cargo test rag_answer -- --ignored --nocapture`.
- [ ] Sin interfaz todavía: `rag_answer`/`search` no están conectados a ningún endpoint Axum — el consumidor real es el servidor MCP de Fase 6 (`rmcp`, no implementado). Por eso todo `src/rag/` tiene un `#![allow(dead_code, unused_imports)]` documentado en `mod.rs`, a quitar cuando Fase 6 los llame.
- [ ] Generación opcional: Gemini Nano client-side (Chrome Built-in AI Prompt API), toggle explícito de usuario + detección de capability + fallback a Ollama — fuera del alcance del binario de Rust, se invoca desde el browser (ver CLAUDE.local.md).

## Fase 6 — MCP de solo lectura + interfaz web (no iniciado)

Ver `CLAUDE.local.md` (sección "MCP de solo lectura") y `docs/Arquitechture.md` (Fase 6).

- [ ] Integrar `rmcp` (confirmar versión exacta) montado sobre el Axum existente vía transporte Streamable HTTP.
- [ ] Tools de solo lectura: `search_transcript(query, scope)`, `rag_answer(question, scope)`, `list_audios()`, `get_audio_metadata(audio_id)`.
- [ ] Bearer token en el endpoint MCP antes de bindear a la IP de LAN (no solo localhost).
- [ ] (Opcional) WebSocket + TLS (`axum-server` + `rustls`) para conversación multi-turno con streaming, separado del transporte MCP.

## Housekeeping

- [ ] `src/handlers/QuadrantConf.json` parece un fixture/ejemplo de payload de Qdrant suelto en `handlers/` — mover a un directorio de fixtures o a `docs/` si es referencia, o a tests si es un caso de prueba. Su schema (`source`, sin `avg_logprob`) ya quedó desactualizado frente al payload real definido para Fase 4 — actualizar o reemplazar al implementar `embeddings.rs`.
- [ ] Tests unitarios para `decoder`/`checkpoint`/`jsonl_writer` en aislamiento (hoy la única cobertura de Fase 1/2 es el test de integración `#[ignore]` en `router.rs` que depende de audio real + el modelo).
- [ ] Manejo de errores / reintentos cuando `run_pipeline` falla a mitad de un audio largo (hoy `audio_handler.rs` loguea el error por `eprintln!` pero no reintenta ni marca el job como `Failed` en `job.json`).
- [ ] Documentar en el README (o en un script) cómo instalar `ffmpeg`/`ffprobe` para quien clone el repo, ya que ahora son una dependencia real del pipeline.
