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
- **Pendiente**: el flujo de recuperación ante crash real todavía no se probó end-to-end (matar el proceso a mitad de un audio largo y confirmar que retoma sin re-transcribir desde el principio). Además, tras un `seek_seconds` real el contador `chunk_index` del decoder se reinicia en 0 — los timestamps `start_sec`/`end_sec` quedan correctos porque se derivan del segundo de reanudación, pero la numeración de chunk no continúa la del job original. Ver `docs/TODO.md`.

## Fase 4 — Embeddings + Qdrant (no implementado aún)

- Ollama genera embeddings sobre el texto transcrito. El modelo de Ollama se levanta, se usa, y se libera — no queda cargado permanentemente entre jobs.
- Qdrant almacena los vectores con un payload fijo: `audio_id`, `chunk_id`, `start`, `end`, `text`, `speaker`.

## Concurrencia

- `AppState.transcription_semaphore` limita a **una** transcripción pesada simultánea (`tokio::sync::Semaphore::new(1)`), para evitar CPU thrashing, invalidación de cache y thermal throttling en la laptop objetivo. Ver `router.rs` / `state.rs`.
- Nada de paralelismo agresivo entre chunks ni entre jobs.

## Por qué no hay arquitectura distribuida

Deliberadamente no hay microservicios, colas externas (Kafka/RabbitMQ) ni Redis. Es una app local-first de un solo proceso: el objetivo es procesar audios largos de forma estable en hardware doméstico, no escalar horizontalmente.
