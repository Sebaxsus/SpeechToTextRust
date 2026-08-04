# Interfaz web — preguntas de diseño abiertas (sin resolver todavía)

Estas son dos funcionalidades pensadas para la futura interfaz web (Fase 6, todavía no implementada) que necesitan una decisión de diseño real antes de poder escribirse como una tarea concreta y accionable en `docs/TODO.md`. Se documentan acá en vez de directamente en el TODO porque el problema central no es "falta hacer X" — es que hay 2-3 formas válidas de hacer X, con costos y beneficios genuinamente distintos en CPU/RAM/SSD, y conviene decidirlo con el usuario antes de escribir código, no a mitad de la implementación.

Ver `docs/TODO.md` (Fase 6) para el punteo corto que referencia este documento.

## 1. Resumen corto del transcript (para identificar audios en la UI) — RESUELTO 2026-08-01

**Objetivo**: que el usuario pueda mirar una lista de audios procesados y reconocer cuál es cuál sin abrir cada `transcript.jsonl` — algo como "Reunión sobre los acuerdos de uso de zonas comunes" al lado de cada `job_id`.

**Restricción dura del proyecto** (`CLAUDE.local.md`): laptop de 16GB sin GPU, "NO paralelismo agresivo", "evitar CPU thrashing", modelos que no quedan residentes permanentemente. Un audio real de este proyecto puede ser de ~5h → cientos de chunks de 30s en `transcript.jsonl`.

Tres estrategias evaluadas, cada una con un costo real distinto:

### A. Map-reduce (mini-resumen por chunk/grupo de chunks, después un resumen final)
- Más preciso: cubre todo el audio, no solo una parte.
- Costo real: cientos de llamadas a Ollama (una por chunk o por grupo) — cada una es un forward pass real del modelo de 7-8B en CPU. En un audio de 5h (~600 chunks de 30s) esto puede significar decenas de minutos de CPU adicional por job, en la misma máquina donde ya corre Whisper.
- Si este job "secundario" corre en paralelo a una transcripción nueva (el requisito explícito de "no bloquear el proceso principal"), reintroduce el problema que `CLAUDE.local.md` pide evitar explícitamente: "NO paralelismo agresivo... evitar CPU thrashing" — dos cargas CPU-bound compitiendo por los mismos 8 threads.
- Mitigación parcial ya probada en el proyecto: igual que `run_embedding_phase` (Fase 4), se puede mantener el modelo cargado (`keep_alive` default) durante todas las llamadas intermedias y forzar `UnloadOnCompletion` solo en la última — evita recargar el modelo de 7-8B desde SSD cientos de veces (que sería mucho peor: cada carga de un modelo Q4_K_M de varios GB es I/O real de disco). Pero esto no resuelve el costo de CPU/tiempo, solo el de I/O de disco por recarga.

### B. Una sola pasada sobre todo el transcript
- Una sola llamada a Ollama — costo de CPU/tiempo mínimo comparado con A.
- Riesgo real, no solo teórico: no está verificado cuál es el `num_ctx` configurado para `qcwind/qwen2.5-7b-instruct-Q4_K_M` en Ollama (el default de Ollama suele ser 2048-4096 tokens salvo que el Modelfile lo overridee). Un transcript de 600 chunks concatenado casi seguro excede eso. Sin verificar y sin ajustar `num_ctx` explícitamente en el request, Ollama trunca el contexto silenciosamente, y el resumen terminaría reflejando solo una parte del audio — el mismo problema de calidad que la opción C, pero sin que sea obvio para quien lo implemente ni para quien lo lee después.

### C. Muestra (primeros N chunks/minutos)
- Una sola llamada corta — costo mínimo, similar a B pero explícitamente acotado (no hay sorpresa de truncamiento silencioso).
- Riesgo de calidad conocido de antemano: si la reunión arranca con charla informal o gente entrando, el resumen puede no reflejar el tema real tratado después. Va directamente en contra de la prioridad #1 del proyecto (accuracy).

**Resuelto (2026-08-01) — variante de A: map-reduce por lotes**, no por chunk individual. Se verificó el `num_ctx` real: el modelo (`qwen2.5-7b-instruct`) soporta hasta 32768 tokens, pero Ollama sin `options.num_ctx` explícito corre con su default de 2048 (hallazgo que además afectaba a `rag_answer` existente, no solo a esta feature — ver `CLAUDE.local.md`: "Generación"). En vez de una sola pasada a contexto máximo (riesgo de VRAM en GPUs de 6GB, ver más abajo) o 600 llamadas sueltas, se agrupan los chunks en ~10-15 lotes (50 chunks o 12000 caracteres cada uno) con `num_ctx=8192` por lote, más una consolidación final. Prioridad frente a una transcripción nueva: el mismo `heavy_compute_semaphore` compartido (no una cola separada ni un semáforo de menor prioridad) — mismo criterio que ya rige Whisper/RAG. Implementado en `src/rag/summary.rs`, disparado por `handlers::audio_handler::lanzar_generacion_resumen`. Ver `CLAUDE.local.md`: "Resumen por audio" para el detalle completo, incluyendo el razonamiento de GPU/VRAM que motivó la estrategia de lotes en vez de un contexto único gigante.

## 2. Reproducir el audio de un segmento de baja confianza (`avg_logprob` bajo) — RESUELTO 2026-08-01

**Objetivo**: dejar que el usuario escuche el audio original de un chunk marcado con `avg_logprob` bajo (candidato a transcripción defectuosa), para verificar manualmente si Whisper se equivocó.

Ya se conoce `start`/`end` de cada chunk (persistido en el payload de Qdrant y en `transcript.jsonl`), así que la ubicación temporal no es el problema — el problema es **qué señal de audio servir y cómo**:

### A. Re-extraer y transcodear el segmento del archivo original (`audio_path`)
- Preserva la calidad "real" del archivo subido (dentro del techo de calidad del códec de origen — ver `CLAUDE.local.md`, la mayoría de las grabaciones reales del proyecto son AMR-NB).
- Los navegadores no reproducen AMR-NB nativamente — hace falta transcodear el segmento a un formato web-friendly (mp3/ogg/aac) con `ffmpeg` al vuelo (`ffmpeg -ss {start} -to {end} -i audio.mp4 -f mp3 -`, mismo patrón que el fallback de decode ya existente en `decoder.rs`). Dependencia ya presente en el proyecto, sin costo nuevo de infraestructura.

### B. Servir el PCM 16kHz mono ya resampleado que Whisper realmente "escuchó"
- Más útil para debugging real: es exactamente la señal que el modelo analizó (después de downmix + resample + crossfade), no una re-extracción distinta que podría sonar distinto.
- Pero **no se persiste hoy** — `StreamingDecoder` genera esos chunks en memoria y los descarta después de pasarlos a Whisper. Habría que, o (b1) persistir esos chunks a disco durante la Fase 2 (más I/O y más archivos por job, más espacio en SSD por job — nada gratis en un proyecto que ya es cuidadoso con el disco), o (b2) re-derivar el chunk on-demand re-decodeando desde el inicio del archivo hasta ese timestamp (lento, y duplica lógica de `StreamingDecoder` para un caso de uso puntual).

**Resuelto (2026-08-01) — opción A.** Implementado en `GET /api/jobs/{job_id}/audio-segment?start=..&end=..` (`handlers::jobs_handler::obtener_segmento_audio_handler`): `ffmpeg` async (`tokio::process::Command`, `.kill_on_drop(true)`) transcodea el rango pedido a mp3 y lo streamea directo como body de la respuesta (`tokio_util::io::ReaderStream` + `axum::body::Body::from_stream`), sin bufferear el clip completo ni persistir nada nuevo en disco. Rango máximo de 120s por request (`400` si se excede). Ver `CLAUDE.local.md`: "Reproducción de un segmento de audio" para el detalle completo, incluyendo la limitación conocida de que un fallo de `ffmpeg` a mitad de stream llega como audio truncado en vez de un error HTTP (inherente a cualquier respuesta streaming).
