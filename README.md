# SpeechToTextRust

App local-first en Rust para transcribir audios largos (grabaciones de reuniones por teléfono, ~5h en promedio) con `whisper-rs`, indexarlos semánticamente en Qdrant con embeddings de Ollama, y responder preguntas sobre esas transcripciones vía RAG (retrieval + reranking opcional + generación).

Todo corre en local, sin GPU obligatoria, pensado para una laptop doméstica (AMD Ryzen 7000, 16 GB RAM, sin GPU dedicada). La prioridad de diseño es **precisión de transcripción/RAG primero, estabilidad segundo, velocidad al final**: nada de mantener modelos residentes en memoria permanentemente, nada de paralelismo agresivo, todo por fases con liberación explícita de memoria entre etapas.

## Estado actual

| Fase | Qué hace | Estado |
|---|---|---|
| 1 — Upload | Recibe multipart, valida por magic bytes (mp3/mp4 únicamente), streaming a disco, crea Job ID | Implementado |
| 2 — Whisper | Decode (Symphonia + fallback `ffmpeg`), resample 16kHz (`rubato`), chunking 30s/overlap 2s, transcripción `whisper-rs` | Implementado |
| 3 — Persistencia | JSONL incremental + checkpoint de `(last_chunk, processed_seconds)` para resume tras crash | Implementado (falta probar el crash real con audio largo) |
| 4 — Embeddings + Qdrant | Embeddings `bge-m3` por chunk, upsert idempotente en Qdrant (point ID determinístico) | Implementado y probado contra servicios reales |
| 5 — RAG | Retrieval (`SearchScope::Audio`/`AllCorpus`) + reranker cross-encoder (solo `AllCorpus`) + generación vía Ollama | Implementado y probado contra servicios reales; sin interfaz HTTP/MCP todavía |
| 6 — MCP + interfaz web | Servidor MCP de solo lectura (`rmcp`) sobre el mismo Axum | No iniciado |

Ver [`docs/TODO.md`](docs/TODO.md) para el detalle línea por línea de qué falta en cada fase.

**Nota importante sobre formatos reales**: las grabaciones de teléfono de este proyecto (tanto `.mp3` como `.m4a`) resultaron ser, en la práctica, **AMR-NB dentro de un contenedor MP4/3GP** — un códec de voz de banda angosta (8kHz) que Symphonia (la librería principal, pura Rust) no sabe decodificar. El pipeline detecta esto automáticamente y cae a invocar `ffmpeg`/`ffprobe` como subproceso para esos archivos (streaming, sin bufferear el audio completo en memoria). Ver "Instalación" más abajo.

## Flujo

```
multipart/form-data
        │
        ▼
POST /api/upload-audio     (streaming a SSD, valida mp3/mp4 por magic bytes)
        │
        ▼
Fase 1 — Upload            crea Job ID + metadata, encola
        │
        ▼
Fase 2 — Whisper            Symphonia (o ffmpeg como fallback), resample a
   (spawn_blocking)         16kHz mono (rubato), chunks de 30s + overlap 2s,
                            whisper-rs CPU-only, libera memoria al terminar
        │
        ▼
Fase 3 — Persistencia       JSONL incremental + checkpoint (resume tras crash)
        │
        ▼
Fase 4 — Embeddings         Ollama (bge-m3): carga → embed por chunk → descarga
        │                   modelo. Upsert idempotente en Qdrant.
        ▼
Fase 5 — RAG (sin HTTP aún) search(query, scope) → [reranker si AllCorpus] →
                            rag_answer(question, scope) vía Ollama
```

Cada fase libera sus recursos antes de que empiece la siguiente — nunca hay dos modelos pesados (Whisper + LLM/reranker de Ollama/candle) cargados en memoria al mismo tiempo.

## Stack

| Crate | Versión | Uso |
|---|---|---|
| axum | 0.8.9 (multipart) | HTTP server |
| tokio | 1.52.3 (full) | runtime async |
| whisper-rs | 0.16.0 | transcripción (GGML, CPU) |
| symphonia | 0.6.0 (all) | decode de audio (puro Rust) |
| rubato | 0.15 | resampler sinc anti-aliasing |
| qdrant-client | 1.18 | cliente de Qdrant (vector DB) |
| ollama-rs | 0.3.4 | cliente de Ollama (embeddings + generación) |
| candle-core / candle-nn / candle-transformers | 0.11.0 | reranker cross-encoder in-process (`XLMRobertaForSequenceClassification`) |
| hf-hub | 0.5.0 | descarga/cachea los pesos del reranker desde Hugging Face Hub |
| tokenizers | 0.22.0 (sin `esaxx_fast`) | tokenización del reranker |
| serde / serde_json | 1.0.228 / 1.0.150 | (de)serialización |
| uuid | 1.23.1 (v4, v5) | Job IDs / point IDs determinísticos |

Rust edition 2024. Ver `Cargo.toml`/`Cargo.lock` para el detalle completo.

**Nota de build en Windows/MSVC**: `tokenizers` desactiva su feature default `esaxx_fast` (dependencia C++ compilada con runtime estático `/MT`) porque choca en link time (`LNK2038`) con el build CMake de `whisper-rs-sys` (runtime dinámico `/MD`). Ese feature solo acelera *entrenar* un tokenizer Unigram/SentencePiece desde cero — este proyecto solo carga tokenizers ya entrenados (`tokenizer.json`), así que no hay pérdida funcional.

**Dependencias externas (no gestionadas por Cargo)**:
- `ffmpeg` y `ffprobe` en el `PATH` — fallback de decode cuando Symphonia no reconoce el códec (AMR-NB, ver "Estado actual").
- Un servidor **Ollama** corriendo en `localhost:11434` con los modelos `bge-m3` y `qcwind/qwen2.5-7b-instruct-Q4_K_M:latest` ya descargados.
- Un servidor **Qdrant** corriendo en `localhost:6334` (gRPC), bindeado solo a loopback.
- Los pesos del **reranker** (`BAAI/bge-reranker-v2-m3`, ~2.2GB en f32) se descargan solos la primera vez que se usa `SearchScope::AllCorpus`, vía `hf-hub` — necesita red esa única vez, después queda cacheado en `~/.cache/huggingface/hub` (o `$HF_HOME`).

## Estructura

```
src/
├── main.rs                    # bootstrap Tokio + Axum, construye AppState (Ollama/Qdrant)
├── state.rs                   # AppState compartido (semáforo de transcripción, clientes)
├── router.rs                  # rutas + test de integración end-to-end
├── handlers/
│   └── audio_handler.rs       # POST /api/upload-audio: streaming a disco + spawn del job
├── audio_pipeline/
│   ├── job.rs                 # creación de Job ID / metadata
│   ├── decoder.rs             # decode streaming (Symphonia + fallback ffmpeg) + resample 16kHz
│   ├── whisper_runner.rs      # wrapper de whisper-rs / FullParams
│   ├── checkpoint.rs          # persistencia de progreso (crash recovery)
│   ├── jsonl_writer.rs        # escritura incremental de la transcripción
│   ├── models.rs              # tipos compartidos (JobMetadata, AudioChunk, TranscriptEntry...)
│   ├── pipeline.rs            # orquesta Fase 2/3
│   └── embeddings.rs          # Fase 4: embeddings (bge-m3) + upsert idempotente en Qdrant
└── rag/                       # Fase 5: RAG, opera sobre datos ya embebidos (no sobre audio)
    ├── retrieval.rs           # SearchScope, búsqueda vectorial + expansión de contexto
    ├── reranker.rs            # cross-encoder in-process (candle), gateado a AllCorpus
    └── generation.rs          # rag_answer: retrieval + generación vía Ollama
```

## Instalación

### 1. Rust

Toolchain estable, edition 2024 (`rustup default stable`, `rustup update`).

### 2. ffmpeg / ffprobe

Deben estar en el `PATH`. En Windows, la forma más simple es con `winget`:

```powershell
winget install ffmpeg
```

Verificar: `ffmpeg -version` y `ffprobe -version` deben correr desde cualquier terminal nueva.

### 3. Modelo de Whisper (GGML)

Descargar `ggml-small-q5_1.bin` (cuantización oficial de whisper.cpp) y colocarlo en `./models/ggml-small-q5_1.bin` desde la raíz del proyecto.

### 4. Ollama

Instalar Ollama ([ollama.com](https://ollama.com)) y descargar los dos modelos que usa el pipeline:

```bash
ollama pull bge-m3
ollama pull qcwind/qwen2.5-7b-instruct-Q4_K_M:latest
```

Verificar que ambos aparecen con `ollama list`. Ollama debe estar corriendo (`localhost:11434`) antes de levantar el servidor o correr los tests de integración.

### 5. Qdrant

Se corre como contenedor Docker, **bindeado solo a loopback** (nunca expuesto en la LAN):

```bash
docker run -d --name qdrant_local \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verificar: `curl http://127.0.0.1:6333/collections` debe responder `200`.

Para revisar/administrar la instancia ya corriendo (dashboard web, ver los vectores guardados,
entender qué significa cada campo) ver [`docs/qdrant.md`](docs/qdrant.md).

### 6. Reranker (candle + Hugging Face Hub)

No requiere instalación manual: la primera vez que corre código que usa `SearchScope::AllCorpus` (incluyendo el test de integración del reranker), `hf-hub` descarga automáticamente `tokenizer.json`, `config.json` y `model.safetensors` de `BAAI/bge-reranker-v2-m3` (~2.2GB) y los cachea en `~/.cache/huggingface/hub`. Necesita red esa única vez; corridas siguientes son 100% offline.

### 7. Configuración (`.env`, opcional)

Todas las rutas, URLs de servicios, modelos y parámetros de tuning (Whisper, chunking, RAG) se pueden personalizar sin tocar el código:

```bash
cp .env.example .env
```

Editá solo lo que necesites — sin `.env`, o con una variable comentada, el comportamiento es idéntico al de un checkout limpio. Ver [`docs/configuracion.md`](docs/configuracion.md) para el detalle completo (qué controla cada variable y en qué archivo se lee).

## Correr en local

```bash
cargo run
```

Levanta el servidor en `http://localhost:3000`. Endpoint principal:

```
POST /api/upload-audio   (multipart/form-data, campo de archivo mp3/mp4)
```

La Fase 5 (RAG) todavía no tiene endpoint HTTP propio — se prueba directamente vía los tests de integración descritos abajo, hasta que la Fase 6 (MCP) los conecte.

### Logs

Por consola se ven solo los eventos importantes (arranque, transiciones de estado de un job — "transcribiendo", "resumiendo", "generando embeddings/resumen", "enviando el estado" — y cualquier warning/error). El detalle completo (métricas por chunk de Whisper, lo que devuelven Ollama y Qdrant en cada llamada) siempre queda en `logs/server.YYYY-MM-DD.log` (rotación diaria), sin importar cómo se corra el servidor:

```bash
cargo run             # consola curada + logs/server.<fecha>.log completo
cargo run -- --log    # consola también muestra todo el detalle (DEBUG), útil para debugging en vivo
```

Ver `docs/Arquitechture.md` (sección "Observabilidad y operación") para el diseño del logger.

### Cerrar el servidor

`CTRL+C` hace un shutdown prolijo: cierra las sesiones MCP abiertas y deja terminar las requests HTTP en curso antes de salir. Si había una transcripción de Whisper en curso, el chunk que estaba procesándose en ese momento se pierde (no hay forma de cancelar CPU-bound a mitad de camino), pero eso ya está cubierto por diseño — `checkpoint.json` solo avanza tras un chunk completo, así que `POST /api/jobs/{job_id}/resume` retoma exactamente ahí.

## Tests

`cargo test` corre todos los tests **excepto** los marcados `#[ignore]`, que dependen de servicios externos vivos (Ollama, Qdrant, red) o de archivos que no están versionados (`sample_Media/`, el modelo GGML). Cada uno se puede correr individualmente con `cargo test <nombre> -- --ignored --nocapture`.

### Tests normales (`cargo test`)

| Test | Qué verifica |
|---|---|
| `audio_pipeline::decoder::tests::primer_chunk_de_un_decoder_nuevo_arranca_en_index_0` | Un `StreamingDecoder` recién creado (sin checkpoint) empieza a numerar chunks desde `0`. |
| `audio_pipeline::decoder::tests::seek_seconds_retoma_la_numeracion_de_chunk_desde_resume_from_chunk` | Tras un `seek_seconds(seconds, resume_from_chunk)` con `seconds > 0` (resume real tras un crash), el decoder retoma la numeración de chunk en `resume_from_chunk` en vez de reiniciar en `0` — es el bug de Fase 3 que se corrigió. |
| `audio_pipeline::decoder::tests::seek_a_cero_en_decoder_nuevo_es_no_op_e_ignora_resume_from_chunk` | Con `seconds <= 0.0` (job nuevo, sin checkpoint previo), `seek_seconds` es un no-op y no aplica `resume_from_chunk` — un job nuevo nunca arranca en un chunk distinto de `0`. |
| `router::tests::ruta_desconocida_devuelve_404` | Una ruta no registrada en el router devuelve `404`. |
| `router::tests::upload_audio_con_content_type_invalido_es_rechazado` | El extractor `Multipart` de axum rechaza una petición sin `content-type: multipart/form-data; boundary=...` antes de que el handler llegue a correr. |
| `router::tests::upload_de_archivo_invalido_es_rechazado_sin_crear_job` | Bytes que no son ni mp3 ni mp4 (magic bytes basura) se rechazan con `400` **sin** crear ningún directorio de job en `./jobs/` — valida que Fase 1 rechaza antes de tocar disco. |

Los tests de `decoder` generan un WAV PCM16 sintético (tono senoidal) en `std::env::temp_dir()` para no depender de `sample_Media/` ni de Whisper.

### Tests ignorados (`--ignored`, requieren servicios/archivos reales)

| Test | Requiere | Qué verifica | Comando |
|---|---|---|---|
| `router::tests::pipeline_hardcodeado_transcribe_audio_real` | `sample_Media/`, `models/ggml-small-q5_1.bin`, Ollama + Qdrant vivos | Sube un audio real por `POST /api/upload-audio`, espera a que Fase 2/3 (Whisper) escriba `transcript.jsonl`, y además confirma que la Fase 4 (embeddings) encadenada automáticamente insertó puntos en Qdrant para ese job — primer test que ejercita Whisper + embeddings juntos, no aislados. Ruta configurable vía `TEST_AUDIO_PATH` (default: `sample_Media/Muestra2_02min.m4a`). Limpia sus propios puntos de Qdrant al terminar. | `cargo test pipeline_hardcodeado -- --ignored --nocapture` (o con otra muestra: `TEST_AUDIO_PATH="sample_Media/otro.m4a" cargo test pipeline_hardcodeado -- --ignored --nocapture`) |
| `audio_pipeline::embeddings::tests::run_embedding_phase_contra_servicios_reales` | Ollama (`bge-m3`) + Qdrant vivos | Siembra un `transcript.jsonl` sintético (un chunk vacío + dos con texto), corre `run_embedding_phase` contra servicios reales, confirma que el chunk vacío se saltea y que se insertan exactamente los puntos esperados (`count` filtrado por `audio_id`), y limpia esos puntos al terminar. | `cargo test embeddings -- --ignored --nocapture` |
| `rag::generation::tests::rag_answer_contra_servicios_reales` | Ollama (`bge-m3` + `qcwind/qwen2.5-7b-instruct-Q4_K_M`) + Qdrant vivos | Siembra una transcripción sintética vía `run_embedding_phase` (reutiliza Fase 4), pregunta sobre un dato puntual con `SearchScope::Audio`, y confirma que la respuesta generada por Ollama efectivamente lo recupera y lo cita — primer test que ejercita retrieval + generación juntos. Limpia sus puntos de Qdrant al terminar. | `cargo test rag_answer -- --ignored --nocapture` |
| `rag::reranker::tests::rerank_reordena_hits_por_relevancia_real` | Red la primera vez (descarga `BAAI/bge-reranker-v2-m3`, ~2.2GB), después offline | No depende de Qdrant/Ollama — es una prueba aislada del cross-encoder: le da tres pasajes con un score de bi-encoder deliberadamente "al revés" del orden esperado, y confirma que el reranker reordena el pasaje realmente relevante a la query al primer lugar. | `cargo test rerank_reordena -- --ignored --nocapture` |

### Verificación completa antes de un commit

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test
```

(Los tests `--ignored` se corren aparte, manualmente, porque dependen de servicios externos vivos y en algunos casos de una descarga inicial.)

## Limitaciones conocidas

- **No hay forma confiable de detectar un job "atascado"**: si el proceso completo del servidor muere (no un error capturado dentro del pipeline, sino el binario cayendo), `job.json.status` queda en `Processing` para siempre — nada distingue eso de un audio de 5h que legítimamente sigue transcribiendo. La señal más precisa disponible es el `mtime` de `checkpoint.json` (se reescribe tras cada chunk de ~30s, así que "hace cuánto no se mueve" mide actividad real), pero **no existe un umbral universal**: cuánto tarda un chunk depende de la máquina, la carga concurrente y el hardware real donde corre el proceso, así que cualquier número fijo sería una verdad relativa a una laptop específica, no una garantía general. Ver `docs/TODO.md` (sección "Nuevos endpoints REST para el cliente web") para el detalle de diseño — deliberadamente sin implementar hasta tener una medición real de cuánto tarda un chunk en condiciones normales.

## Documentación

- [`docs/Arquitechture.md`](docs/Arquitechture.md) — arquitectura y decisiones de diseño por fase (incluye logger y graceful shutdown, sección "Observabilidad y operación").
- [`docs/TODO.md`](docs/TODO.md) — trabajo pendiente, priorizado.
- [`docs/terminology.md`](docs/terminology.md) — glosario de términos (ASR, Nyquist, chunk, tdrz, etc.).
- [`docs/qdrant.md`](docs/qdrant.md) — cómo revisar/administrar Qdrant a mano (dashboard, comandos `curl`, qué significa cada campo).
- [`docs/configuracion.md`](docs/configuracion.md) — todas las variables de `.env`: default, dónde se leen en el código, qué controlan.

Las decisiones técnicas detalladas (params exactos de Whisper, reglas de dedupe de tdrz, límites de RAM) viven en `CLAUDE.local.md`, que es un archivo personal no versionado.
