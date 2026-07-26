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
- [ ] Probar el flujo de recuperación end-to-end: matar el proceso a mitad de un audio de varias horas y confirmar que retoma desde el checkpoint sin re-transcribir desde el principio.
- [ ] Corregir que `chunk_index` en `StreamingDecoder` se reinicia en 0 tras un `seek_seconds` real (los timestamps sí quedan correctos, pero la numeración de chunk no continúa la del job original) — relevante recién cuando se implemente el punto anterior.

## Fase 4 — Embeddings + Qdrant (no iniciado)

- [ ] Integración con Ollama: levantar modelo → generar embeddings del texto transcrito → liberar el modelo. Nunca dejarlo residente entre jobs.
- [ ] Integración con Qdrant: insertar vectores con el payload fijo (`audio_id`, `chunk_id`, `start`, `end`, `text`, `speaker`).
- [ ] Definir cómo se segmenta el texto de cada chunk de transcripción en unidades de embedding (¿un embedding por chunk de 30s, o re-agrupando por oración/turno?).

## Housekeeping

- [ ] `src/handlers/QuadrantConf.json` parece un fixture/ejemplo de payload de Qdrant suelto en `handlers/` — mover a un directorio de fixtures o a `docs/` si es referencia, o a tests si es un caso de prueba.
- [ ] Tests unitarios para `decoder`/`checkpoint`/`jsonl_writer` en aislamiento (hoy la única cobertura de Fase 1/2 es el test de integración `#[ignore]` en `router.rs` que depende de audio real + el modelo).
- [ ] Manejo de errores / reintentos cuando `run_pipeline` falla a mitad de un audio largo (hoy `audio_handler.rs` loguea el error por `eprintln!` pero no reintenta ni marca el job como `Failed` en `job.json`).
- [ ] Documentar en el README (o en un script) cómo instalar `ffmpeg`/`ffprobe` para quien clone el repo, ya que ahora son una dependencia real del pipeline.
