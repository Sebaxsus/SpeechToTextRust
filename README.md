# SpeechToTextRust

App local-first en Rust para transcribir audios largos (grabaciones de reuniones por teléfono, ~5h en promedio) con `whisper-rs` y, en fases posteriores, indexarlos semánticamente (RAG) en Qdrant usando embeddings generados con Ollama.

Todo corre en local, sin GPU obligatoria, pensado para una laptop doméstica (AMD Ryzen 7000, 16 GB RAM, sin GPU dedicada). La prioridad de diseño es **estabilidad y precisión de transcripción por encima de velocidad**: nada de mantener modelos residentes en memoria, nada de paralelismo agresivo, todo por fases con liberación explícita de memoria entre etapas.

## Estado actual

Fase 1 (upload) y Fase 2 (Whisper/Symphonia) funcionan de punta a punta: subida validada por magic bytes → job con directorio propio → decode + resample 16kHz + chunking con overlap/crossfade → transcripción con whisper-rs → persistencia JSONL, probado contra audio real. Fase 3 (checkpoint/crash-recovery real) y Fase 4 (embeddings + Qdrant) siguen pendientes. Ver [`docs/TODO.md`](docs/TODO.md) para el detalle de qué falta.

**Nota importante sobre formatos reales**: las grabaciones de teléfono de este proyecto (tanto `.mp3` como `.m4a`) resultaron ser, en la práctica, **AMR-NB dentro de un contenedor MP4/3GP** — un códec de voz de banda angosta (8kHz) que Symphonia (la librería principal, pura Rust) no sabe decodificar. El pipeline detecta esto automáticamente y cae a invocar `ffmpeg`/`ffprobe` como subproceso para esos archivos (streaming, sin bufferear el audio completo en memoria). **Esto requiere tener `ffmpeg` y `ffprobe` instalados y en el `PATH`** — es la única dependencia externa al binario de Rust en todo el proyecto. Ver `CLAUDE.local.md` para el detalle técnico completo.

## Flujo (diseño objetivo)

```
multipart/form-data
        │
        ▼
POST /api/upload-audio   (streaming a SSD, valida mp3/mp4)
        │
        ▼
Fase 1 — Upload          crea Job ID + metadata, encola
        │
        ▼
Fase 2 — Whisper         Symphonia (o ffmpeg como fallback), resample a
   (spawn_blocking)      16kHz mono (rubato), chunks de 30s + overlap,
                          whisper-rs CPU-only, libera memoria al terminar
        │
        ▼
Fase 3 — Embeddings      Ollama (carga → embed → descarga modelo)
        │
        ▼
Fase 4 — Qdrant          payload semántico (audio_id, chunk_id, start,
                          end, text, speaker)
```

Cada fase libera sus recursos antes de que empiece la siguiente — nunca hay dos modelos pesados (Whisper + Ollama) cargados en memoria al mismo tiempo.

## Stack

| Crate | Versión |
|---|---|
| axum | 0.8.9 (multipart) |
| tokio | 1.52.3 (full) |
| whisper-rs | 0.16.0 |
| symphonia | 0.6.0 (all) |
| qdrant-client | 1.18 |
| ollama-rs | 0.3.4 |
| serde / serde_json | 1.0.228 / 1.0.150 |
| uuid | 1.23.1 (v4) |
| rubato | 0.15 (resampler sinc anti-aliasing) |

Rust edition 2024. Ver `Cargo.toml` para el lockfile completo.

**Dependencia externa (no-Rust)**: `ffmpeg` y `ffprobe` deben estar instalados y en el `PATH` — se usan como fallback de decode cuando Symphonia no reconoce el códec del archivo (ver nota sobre AMR-NB en "Estado actual").

## Estructura

```
src/
├── main.rs                    # bootstrap Tokio + Axum
├── state.rs                   # AppState compartido (semáforo de transcripción, etc.)
├── router.rs                  # rutas
├── handlers/
│   └── audio_handler.rs       # POST /api/upload-audio: streaming a disco + spawn del job
└── audio_pipeline/
    ├── job.rs                 # creación de Job ID / metadata
    ├── decoder.rs              # decode streaming (Symphonia) + resample 16kHz
    ├── whisper_runner.rs       # wrapper de whisper-rs / FullParams
    ├── checkpoint.rs           # persistencia de progreso (crash recovery)
    ├── jsonl_writer.rs         # escritura incremental de la transcripción
    ├── models.rs               # tipos compartidos (JobMetadata, AudioChunk, ...)
    └── pipeline.rs             # orquesta las fases anteriores
```

## Correr en local

```bash
cargo run
```

Levanta el servidor en `http://localhost:3000`. Endpoint principal:

```
POST /api/upload-audio   (multipart/form-data, campo de archivo mp3/mp4)
```

## Documentación

- [`docs/Arquitechture.md`](docs/Arquitechture.md) — arquitectura y decisiones de diseño por fase.
- [`docs/TODO.md`](docs/TODO.md) — trabajo pendiente, priorizado.
- [`docs/terminology.md`](docs/terminology.md) — glosario de términos (ASR, Nyquist, chunk, tdrz, etc.).

Las decisiones técnicas detalladas (params exactos de Whisper, reglas de dedupe de tdrz, límites de RAM) viven en `CLAUDE.local.md`, que es un archivo personal no versionado.
