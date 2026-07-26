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

## Fase 1 — Upload (`handlers/audio_handler.rs`)

- Recibe `multipart/form-data` en `POST /api/upload-audio`.
- Escribe el archivo a disco incrementalmente (`campo.chunk().await` + `write_all`), nunca junta el archivo completo en un `Vec<u8>`.
- **Pendiente**: validar que el archivo sea mp3 o mp4 *antes* de escribir a disco (o al menos antes de encolar la Fase 2) — hoy el handler acepta cualquier `file_name` sin chequear extensión ni magic bytes.
- Tras guardar el archivo, adquiere un permiso de `transcription_semaphore` (1 solo permit — ver `state.rs`) y lanza el pipeline en background con `tokio::spawn` + `tokio::task::spawn_blocking` para la parte CPU-bound.
- Genera un Job ID y metadata (`audio_pipeline::job::create_job`) — hoy es un `todo!()`.

## Fase 2 — Whisper (`audio_pipeline/decoder.rs`, `whisper_runner.rs`, `pipeline.rs`)

- **Decode + resample** (`StreamingDecoder`, hoy sin implementar): Symphonia decodifica el contenedor (mp3/mp4) pero **no resamplea** — el pipeline debe resamplear explícitamente a PCM mono 16kHz con un resampler basado en sinc con filtro anti-aliasing (`rubato`), nunca decimación/interpolación lineal, para no aliasar frecuencias por encima de la nueva Nyquist (8kHz) sobre el espectro de voz.
- **Chunking**: ventanas de 30s, tamaño múltiplo del hop size de Whisper (10ms hop / 25ms window), con overlap de 2-3s y crossfade (Hann/Tukey) en los bordes para evitar cortar palabras o duplicarlas.
- **Inferencia** (`WhisperRunner`, hoy sin implementar): whisper-rs 0.16.0, modelo GGML cuantizado `ggml-base-tdrz-q5_1.bin`, `FullParams` con `SamplingStrategy::Greedy`. Siempre dentro de `tokio::task::spawn_blocking` — nunca `tokio::spawn` directo, porque es trabajo CPU-bound y bloquearía el runtime async.
- **Turn-detection (tdrz)**: el modelo es *tinydiarize*, pero hoy la config no llama `set_tdrz_enable(true)`, así que el campo `speaker` del payload nunca se puebla. Si se activa: tinydiarize da *turn segmentation* (cambios de speaker dentro de una llamada a `full()`), no clustering de identidad — hace falta un contador de turno propio que viva en el estado del job, no en el scope de cada chunk. Los turnos detectados en la ventana de overlap entre chunks deben dedupearse (una sola zona de decisión, no dos independientes) para no contar un mismo cambio de speaker dos veces en la costura.
- **Liberación de memoria**: el runner de Whisper se dropea al terminar la fase — no queda residente esperando la siguiente subida.

## Fase 3 — Persistencia incremental (`checkpoint.rs`, `jsonl_writer.rs`)

- `JsonlWriter` escribe una línea JSON por chunk (`{"chunk":1,"text":"..."}`) — nunca acumula la transcripción completa en memoria antes de escribir.
- `CheckpointManager` persiste `(last_chunk, processed_seconds)` para poder retomar un audio de 5h si el proceso crashea a la mitad, sin re-transcribir desde el principio.

## Fase 4 — Embeddings + Qdrant (no implementado aún)

- Ollama genera embeddings sobre el texto transcrito. El modelo de Ollama se levanta, se usa, y se libera — no queda cargado permanentemente entre jobs.
- Qdrant almacena los vectores con un payload fijo: `audio_id`, `chunk_id`, `start`, `end`, `text`, `speaker`.

## Concurrencia

- `AppState.transcription_semaphore` limita a **una** transcripción pesada simultánea (`tokio::sync::Semaphore::new(1)`), para evitar CPU thrashing, invalidación de cache y thermal throttling en la laptop objetivo. Ver `router.rs` / `state.rs`.
- Nada de paralelismo agresivo entre chunks ni entre jobs.

## Por qué no hay arquitectura distribuida

Deliberadamente no hay microservicios, colas externas (Kafka/RabbitMQ) ni Redis. Es una app local-first de un solo proceso: el objetivo es procesar audios largos de forma estable en hardware doméstico, no escalar horizontalmente.
