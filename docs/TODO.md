# TODO

Pendientes detectados en el estado actual del código, priorizados de forma aproximada (bloqueantes primero). Ver `docs/Arquitechture.md` para el contexto de cada fase y `CLAUDE.local.md` para los detalles de parámetros/decisiones ya fijadas.

## Bloqueante — Fase 1 (Upload)

- [ ] Validar formato de archivo (mp3/mp4 únicamente) en `handlers/audio_handler.rs` **antes** de escribir a disco, o al menos antes de encolar la Fase 2. Hoy no hay ningún chequeo de extensión ni magic bytes — se acepta y se escribe cualquier archivo.
- [ ] Implementar `audio_pipeline::job::create_job` (hoy `todo!()`): generar Job ID, rutas de `audio_path` / `transcript_path` / `checkpoint_path`, persistir metadata inicial.
- [ ] Sanear `nombre_archivo` en `audio_handler.rs` antes de usarlo como parte de una ruta de disco (`format!("./uploads/{}", nombre_archivo)`) — hoy viene directo del `file_name()` del multipart sin validar path traversal (`../`) ni caracteres inválidos.

## Bloqueante — Fase 2 (Whisper)

- [ ] Implementar `StreamingDecoder` (`audio_pipeline/decoder.rs`, hoy `todo!()` en los 3 métodos): demux con Symphonia, extraer solo el track de audio (nunca decodificar video en los mp4), y **resamplear a PCM mono 16kHz con `rubato`** (filtro anti-aliasing, sinc-based) — obligatorio, no opcional, ver `CLAUDE.local.md`.
- [ ] Chunking de 30s con overlap de 2-3s y crossfade (Hann/Tukey) en los bordes, respetando que el tamaño de chunk sea múltiplo del hop size de Whisper.
- [ ] Implementar `WhisperRunner` (`audio_pipeline/whisper_runner.rs`, hoy `todo!()`): cargar el modelo GGML, aplicar el `FullParams` ya definido en `CLAUDE.local.md`, ejecutar siempre dentro de `tokio::task::spawn_blocking`.
- [ ] Confirmar que la llamada real a Whisper ocurre en `spawn_blocking` y no en el `tokio::spawn` que ya envuelve el job en `audio_handler.rs` (hoy ese `spawn_blocking` interno existe en `pipeline.rs` vía `run_pipeline`, pero falta la implementación real que lo justifique).

## Importante — Precisión / tdrz

- [ ] Decidir si se activa `set_tdrz_enable(true)` (hoy no se llama). Si se activa:
  - [ ] Mantener un contador de `speaker_id` / timestamp del último turno confirmado en el estado del *job* (no en el scope de cada llamada a `full()`).
  - [ ] Implementar dedupe de turnos en la ventana de overlap entre chunks (umbral mínimo ~0.5-1s entre turnos, tratar el overlap como una sola zona de decisión).
  - [ ] Probar calidad empíricamente en español — la combinación `base` + `set_language("es")` + tdrz no está validada por upstream (tdrz solo está probado oficialmente sobre `small.en-tdrz`).
- [ ] Evaluar `set_suppress_nst(true)` para reducir alucinaciones en silencios, complementando `no_speech_thold`.
- [ ] Evaluar VAD real (`set_vad_model_path` / `enable_vad`) para saltar tramos de silencio en audios de 5h y bajar CPU, sin perder tramos con voz baja.

## Fase 3 — Persistencia

- [ ] Implementar `JsonlWriter::append` (hoy `todo!()`): escribir una línea JSON + flush por llamada, sin acumular el transcript completo en memoria.
- [ ] Implementar `CheckpointManager` (`new`/`load`/`save`, hoy `todo!()` los 3): persistencia de `(last_chunk, processed_seconds)` para recuperación ante crash en audios largos.
- [ ] Probar el flujo de recuperación end-to-end: matar el proceso a mitad de un audio de varias horas y confirmar que retoma desde el checkpoint sin re-transcribir desde el principio.

## Fase 4 — Embeddings + Qdrant (no iniciado)

- [ ] Integración con Ollama: levantar modelo → generar embeddings del texto transcrito → liberar el modelo. Nunca dejarlo residente entre jobs.
- [ ] Integración con Qdrant: insertar vectores con el payload fijo (`audio_id`, `chunk_id`, `start`, `end`, `text`, `speaker`).
- [ ] Definir cómo se segmenta el texto de cada chunk de transcripción en unidades de embedding (¿un embedding por chunk de 30s, o re-agrupando por oración/turno?).

## Housekeeping

- [ ] `src/handlers/QuadrantConf.json` parece un fixture/ejemplo de payload de Qdrant suelto en `handlers/` — mover a un directorio de fixtures o a `docs/` si es referencia, o a tests si es un caso de prueba.
- [ ] Tests unitarios para el pipeline (`decoder`, `checkpoint`, `jsonl_writer`) a medida que se implementen — hoy solo hay tests de routing en `router.rs`.
- [ ] Definir manejo de errores / reintentos cuando `run_pipeline` falla a mitad de un audio largo (hoy `audio_handler.rs` solo hace `eprintln!` y descarta el error silenciosamente en el `tokio::spawn`).
