# Arquitectura

## Objetivo del sistema

Transcribir audios de reuniones grabados con teléfono (~5h promedio, condiciones de audio no controladas: cross-talk, ruido de fondo) y, en fases posteriores, indexarlos semánticamente para RAG. Todo local, sin GPU obligatoria, en una laptop con 16 GB de RAM como límite duro.

La prioridad al tomar cualquier decisión de diseño es, en orden: **precisión de transcripción > estabilidad > velocidad**.

## Principio rector: fases desacopladas, memoria liberada entre ellas

El pipeline nunca mantiene dos cargas pesadas (Whisper, Ollama) residentes en memoria a la vez. Cada fase:

1. Carga solo lo que necesita.
2. Procesa por streaming (nunca el archivo/corpus completo en RAM).
3. Persiste su resultado de forma incremental (JSONL, checkpoints).
4. Libera sus recursos antes de que arranque la fase siguiente.

## Fase 1 — Upload (`handlers/audio_handler.rs`, `audio_pipeline/job.rs`) — implementado

- Recibe `multipart/form-data` en `POST /api/upload-audio`.
- **Validación por magic bytes, no por extensión declarada**: se lee el primer chunk del stream *antes* de crear ningún archivo o directorio, y se detecta el contenedor real (`ID3`/frame sync MPEG → mp3; caja `ftyp` ISO-BMFF → mp4, cubre también `.m4a`). Cualquier otro contenido se rechaza con `400` sin tocar disco.
- `audio_pipeline::job::create_job(extension)` genera un `job_id` (UUID v4) y una carpeta propia `./jobs/{job_id}/` con `audio_path`/`transcript_path`/`checkpoint_path` fijos y un `job.json` con la metadata inicial (`status: Pending`).
- El nombre de archivo que manda el cliente **nunca se usa como parte de una ruta de disco** — solo se usa para logging. Esto elimina path traversal por construcción en vez de "sanearlo".
- Tras guardar el archivo, adquiere un permiso de `transcription_semaphore` (1 solo permit — ver `state.rs`) y lanza el pipeline en background con `tokio::spawn` + `tokio::task::spawn_blocking` para la parte CPU-bound.
- La respuesta `202` incluye el `job_id` (JSON) para que el cliente pueda consultar el resultado más adelante.

## Fase 2 — Whisper (`audio_pipeline/decoder.rs`, `whisper_runner.rs`, `pipeline.rs`) — implementado

- **Decode + resample** (`StreamingDecoder`): intenta demuxear/decodificar con Symphonia primero (puro Rust). Se queda únicamente con el track de audio del contenedor — nunca instancia un decoder de video para un mp4. Si Symphonia no reconoce el códec (ver nota AMR-NB más abajo), cae a un fallback que invoca `ffmpeg` como subproceso y lee PCM crudo de su stdout de forma incremental (bloques de 64KB, nunca el archivo completo en memoria). En ambos casos, el audio decodificado (downmixed a mono) pasa por el mismo resampler `rubato` (`SincFixedIn`, interpolación Cubic, ventana BlackmanHarris2) hacia PCM mono 16kHz — sinc con filtro anti-aliasing, nunca decimación/interpolación lineal.
- **Nota real sobre el sample rate de entrada**: el supuesto original de "el input viene a 44.1/48kHz" no se cumple para el dataset real de este proyecto — ver "Códec real de las grabaciones" abajo. El resampler calcula el ratio dinámicamente a partir del sample rate reportado por el decoder/`ffprobe`, así que funciona igual de bien haciendo upsampling que downsampling.
- **Chunking**: ventanas de 30s (480,000 samples @ 16kHz, múltiplo exacto del hop de Whisper), con overlap de 2s y crossfade (Hann) aplicado únicamente en los bordes *internos* entre chunks consecutivos — nunca en el primer inicio ni el último final del audio completo.
- **Inferencia** (`WhisperRunner`): whisper-rs 0.16.0, modelo GGML cuantizado `ggml-small-q5_1.bin` (ver nota de cambio de modelo abajo), `FullParams` con `SamplingStrategy::Greedy` (config exacta fijada en `CLAUDE.local.md`, sin `tdrz`). Mantiene un único `WhisperState` vivo durante todo el job (no uno por chunk), para que `set_no_context(false)` aporte continuidad de contexto real entre los chunks de una reunión larga. Siempre corre dentro de `tokio::task::spawn_blocking` (heredado desde `audio_handler.rs`, que ya envuelve todo `run_pipeline`) — nunca `tokio::spawn` directo.
- **Cambio de modelo**: el modelo originalmente previsto, `ggml-base-tdrz-q5_1.bin` (*tinydiarize*, turn-detection de speaker), no se consiguió. Se usa `ggml-small-q5_1.bin` — mejor accuracy de transcripción (prioridad #1 del proyecto) a costa de más RAM/CPU por chunk, pero sin variante tdrz. La detección de turnos de speaker queda fuera de alcance por ahora (no es una regresión: tdrz tampoco estaba activado antes). Ver `CLAUDE.local.md` para el detalle de costo/beneficio de tdrz, guardado como referencia por si se retoma con otro modelo.
- **Liberación de memoria**: el runner de Whisper se dropea al terminar la fase — no queda residente esperando la siguiente subida.

### Códec real de las grabaciones — AMR-NB y fallback a ffmpeg

Se verificó con `ffprobe` que las grabaciones reales de `sample_Media/` (tanto `.mp3` como `.m4a`) son en realidad **AMR-NB a 8kHz mono dentro de un contenedor MP4/3GP**, sin importar la extensión declarada. Symphonia (puro Rust) no implementa un decoder de AMR-NB. Por eso `StreamingDecoder` prueba Symphonia primero (sigue siendo el camino principal para cualquier mp3/mp4/AAC bien formado) y cae a `ffmpeg`/`ffprobe` como subproceso únicamente cuando Symphonia no encuentra un track de audio decodificable.

Esto introduce la única dependencia externa al binario de Rust del proyecto: `ffmpeg` y `ffprobe` deben estar en el `PATH`. Sigue siendo 100% local/offline (no hay llamadas de red), solo deja de ser puro-Rust para el paso de decode en los archivos que lo requieren. AMR-NB además es un códec de voz de banda angosta (~4kHz de ancho útil) — es un techo de calidad inherente a la grabación de origen, no algo que el resampling o el tuning de Whisper puedan compensar.

## Fase 3 — Persistencia incremental (`checkpoint.rs`, `jsonl_writer.rs`) — parcialmente implementado

- `JsonlWriter::append` escribe una línea JSON por chunk (`{"chunk":1,"start":0.0,"end":30.0,"text":"..."}`) y flushea inmediatamente — nunca acumula la transcripción completa en memoria.
- `CheckpointManager` persiste `(last_chunk, processed_seconds)` en disco (`load` devuelve el default si el archivo no existe todavía, es decir la primera corrida de un job).
- `StreamingDecoder::seek_seconds(seconds, resume_from_chunk)` retoma la numeración de chunk del job original: `resume_from_chunk` (`checkpoint.last_chunk + 1`) se aplica dentro del guard de `seconds <= 0.0`, así que un job nuevo (sin checkpoint, `processed_seconds == 0.0`) no se ve afectado y sigue arrancando en el chunk 0 por default del constructor.
- **Pendiente**: el flujo de recuperación ante crash real todavía no se probó end-to-end (matar el proceso a mitad de un audio largo y confirmar que retoma sin re-transcribir desde el principio). Ver `docs/TODO.md`.

## Fase 4 — Embeddings + Qdrant (implementado y probado contra servicios reales)

- **Embeddings**: `audio_pipeline::embeddings::run_embedding_phase` lee `transcript.jsonl` línea por línea (streaming, `Peekable` sobre `BufReader::lines()`) y le pide a Ollama un embedding por `TranscriptEntry` (el mismo chunk de 30s del JSONL, sin re-chunking semántico separado) con el modelo `bge-m3` (dense, 1024 dims) — elegido por soporte multilingüe real en español, verificado en benchmarks frente a alternativas como `nomic-embed-text`. Chunks con `text` vacío (silencio/no-speech) se saltan. `keep_alive` queda en default (Ollama gestiona su propio timeout corto) en todas las líneas salvo la última del job, donde se fuerza `KeepAlive::UnloadOnCompletion` — el modelo nunca queda residente entre jobs.
- **Secuenciación**: `run_embedding_phase` corre en `audio_handler.rs` como tarea async normal (nunca `spawn_blocking`), disparada solo cuando el `spawn_blocking` de `run_pipeline` (Fase 2/3, síncrono/CPU-bound) resuelve `Ok(Ok(()))` — Whisper y Ollama nunca están cargados al mismo tiempo.
- **Qdrant — topología**: una sola colección (`transcripts`) para todo el corpus (todos los audios/jobs), no una colección por audio. A esta escala (~600 vectores por audio de 5h) separar por audio multiplicaría el overhead fijo de HNSW/segmentos/WAL por colección sin reducir el volumen total de datos. `on_disk: true` en vectores y payload resuelve el crecimiento indefinido del corpus. `ensure_collection` crea la colección (idempotente, vía `collection_exists`) y el índice de payload `Keyword` sobre `audio_id` en el primer job.
- **Config de colección**: distancia Cosine, precisión f32 completa (sin quantization — no se justifica en RAM a este volumen y arriesga accuracy), índice de payload sobre `audio_id` (filtrado casi gratis dentro de la colección única), point ID determinístico `Uuid::new_v5(Uuid::NAMESPACE_OID, "audio_id:chunk_id")` (upsert idempotente si Fase 4 se reintenta tras un crash, en vez de duplicar puntos).
- **Exposición de red**: `AppState.qdrant` se construye en `main.rs` apuntando a `http://localhost:6334` — nunca se expone directamente en la LAN, incluso cuando el servidor MCP (Fase 6) sí se expone. Un solo punto de entrada guardado.
- **Payload**: `audio_id` (= `job_id`, un job es un audio), `chunk_id`, `start`, `end`, `text`, `speaker` (fijo en `"unknown"` — tdrz no está activo, ver Fase 2), y `avg_logprob` (confianza promedio del chunk según Whisper — permite a la Fase 5 matizar respuestas basadas en tramos de baja confianza, dado el ruido/cross-talk real del dataset).
- **Verificado end-to-end**: `run_embedding_phase` corrió contra `bge-m3:latest` real en Ollama y un Qdrant real (contenedor Docker `qdrant/qdrant`, bindeado solo a `127.0.0.1:6333`/`127.0.0.1:6334` — nunca `0.0.0.0`), confirmando el skip de chunks vacíos, el conteo exacto de puntos insertados y la limpieza sin dejar residuos. Ver `docs/TODO.md` para el comando de test manual.

## Fase 5 — RAG: retrieval, reranking y generación (no implementado aún)

- **Scope forzado, sin default implícito a todo el corpus**: las tools de retrieval reciben un `SearchScope` (`Audio(audio_id)` por defecto, o `AllCorpus` solo si el caller lo pide explícitamente) en vez de un `audio_id` opcional — evita que "buscar en todo el corpus" ocurra por omisión.
- **Context assembly**: top-k (5-8) expandido con el chunk vecino (`chunk_id ± 1`) de cada hit, para mitigar oraciones cortadas en el borde de 30s.
- **Reranker (cross-encoder)**, gateado únicamente a `scope = AllCorpus`: con un solo audio el ranking por coseno del bi-encoder ya es casi exhaustivo (beneficio marginal); con el corpus completo (miles de chunks de distintas reuniones) un cross-encoder (`bge-reranker-v2-m3`, candidato) recupera precisión real evaluando query+pasaje con atención cruzada. Pendiente verificar si corre vía Ollama nativamente o requiere `candle-transformers` in-process.
- **Generación — default: Ollama server-side** (`qwen2.5:7b-instruct`, propuesto), para ambos scopes. Decidido así porque la prioridad #1 del proyecto es accuracy: se descartó deliberadamente hacer default a un modelo on-device más liviano.
- **Generación — opcional: Gemini Nano client-side** vía Chrome Built-in AI (Prompt API), activable como toggle explícito del usuario, nunca automático. En este camino el servidor solo hace retrieval (sin generación) y manda los chunks recuperados al browser, que genera la respuesta 100% on-device con streaming nativo. Requiere detección de capability y fallback a Ollama si el navegador no la soporta.

## Fase 6 — MCP de solo lectura + interfaz web (no implementado aún)

- `rmcp` (SDK oficial de Rust para MCP) montado sobre el mismo servidor Axum vía transporte Streamable HTTP — sin microservicio aparte.
- Tools, todas de solo lectura (nunca escriben/borran en Qdrant ni disparan transcripciones): `search_transcript(query, scope)` (retrieval only, alimenta el camino de Gemini Nano client-side), `rag_answer(question, scope)` (retrieval + generación vía Ollama — default y fallback cuando Gemini Nano no aplica), `list_audios()` / `get_audio_metadata(audio_id)`.
- **Exposición en LAN**: el servidor se bindea también a la IP de red local para que otro dispositivo lo consulte (motivo: performance, no correr el pipeline pesado en el equipo más débil). Requiere bearer token en el endpoint MCP — de solo lectura no implica sin autenticación, dado que expone contenido transcrito de reuniones.
- **Fase opcional — WebSocket + TLS** para la conversación interfaz↔RAG (además del MCP): `axum-server` + `rustls` (certificado autofirmado válido para LAN), streaming de tokens y sesión persistente multi-turno. No es parte del MVP (que funciona con request/response simple vía MCP); es una mejora de UX posterior.

## Concurrencia

- `AppState.transcription_semaphore` limita a **una** transcripción pesada simultánea (`tokio::sync::Semaphore::new(1)`), para evitar CPU thrashing, invalidación de cache y thermal throttling en la laptop objetivo. Ver `router.rs` / `state.rs`.
- Nada de paralelismo agresivo entre chunks ni entre jobs.

## Por qué no hay arquitectura distribuida

Deliberadamente no hay microservicios, colas externas (Kafka/RabbitMQ) ni Redis. Es una app local-first de un solo proceso: el objetivo es procesar audios largos de forma estable en hardware doméstico, no escalar horizontalmente.
