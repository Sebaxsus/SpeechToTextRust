# Referencia de API

Estado actual de la superficie HTTP del servidor (`src/main.rs`, `src/router.rs`, `src/handlers/audio_handler.rs`, `src/handlers/jobs_handler.rs`, `src/handlers/rag_handler.rs`, `src/mcp/mod.rs`), documentado como base para: (1) construir el cliente web, (2) probar el servidor con distintos audios, (3) tener un punto de referencia único de qué existe hoy realmente, sin desactualizarse con lo que dice `docs/Arquitechture.md` a nivel de diseño. Para el "por qué" de cada decisión, ver `CLAUDE.local.md` y `docs/Arquitechture.md`; este documento se limita al "qué" — rutas, payloads, códigos de estado.

## Arranque del servidor

`src/main.rs`:

- Bind: `0.0.0.0:3000` — el servidor ya escucha en todas las interfaces (LAN incluida), no solo `localhost`. Ver "Exposición en LAN — gaps" más abajo antes de considerar esto seguro para un cliente remoto.
- Clientes construidos una sola vez en `main.rs` y compartidos vía `Arc<AppState>` (`src/state.rs`): `Ollama::default()` (`http://127.0.0.1:11434`) y `Qdrant::from_url("http://localhost:6334")`. Son clientes livianos (no cargan modelos) — construirlos al boot no viola la regla de "nunca dejar Whisper/Ollama residentes".
- `heavy_compute_semaphore` (`Semaphore::new(1)`): como máximo una carga pesada de CPU/modelo a la vez en todo el proceso — compartido entre `POST /api/upload-audio`/`POST /api/jobs/{job_id}/resume` (Whisper) y las tools MCP `search_transcript`/`rag_answer` **y** sus equivalentes REST `POST /api/search`/`POST /api/rag/answer` (Ollama/reranker). Antes se llamaba `transcription_semaphore` y solo gateaba el pipeline de audio; se amplió (2026-07-29) tras detectar que las tools de RAG no tenían ningún gate de concurrencia propio.
- Si `MCP_BEARER_TOKEN` no está seteado en el entorno, el proceso imprime una advertencia por consola al arrancar (no rechaza arrancar, no rechaza bindear a `0.0.0.0`) — ver "Autenticación" más abajo.
- Logging (agregado 2026-07-31, ver `CLAUDE.local.md`): `cargo run -- --log` sube el nivel del logger centralizado (`tracing`) a `DEBUG`, mostrando las métricas por chunk del pipeline además de lo que ya se ve por defecto (arranque, warnings, errores). Sin el flag, el nivel es `INFO` — misma visibilidad que antes de migrar a `tracing`.

No hay healthcheck (`GET /health` o similar) implementado todavía.

## Rutas registradas

`src/router.rs::crear_router` monta dos grupos de rutas al mismo nivel (`Router::merge`, no `nest`): uno sin autenticar (`crear_router` directo) y uno protegido por el mismo bearer token que `/mcp` (`crear_router_protegido`, ver "Autenticación" más abajo).

| Método | Ruta | Handler | Auth | Descripción corta |
|---|---|---|---|---|
| `POST` | `/api/upload-audio` | `recibir_y_procesar_audio` | No | Sube un audio nuevo, crea un job, dispara el pipeline. |
| `POST` | `/api/jobs/{job_id}/resume` | `reanudar_job` | No | Reanuda un job existente cortado a mitad de camino. |
| `GET` | `/api/jobs` | `jobs_handler::listar_jobs_handler` | Sí | Lista todos los jobs (equivalente REST de `list_audios`). |
| `GET` | `/api/jobs/{job_id}` | `jobs_handler::obtener_job_handler` | Sí | Status + progreso de un job puntual. |
| `GET` | `/api/jobs/{job_id}/transcript` | `jobs_handler::obtener_transcript_handler` | Sí | Contenido de `transcript.jsonl`. |
| `GET` | `/api/jobs/{job_id}/metrics` | `jobs_handler::obtener_metricas_handler` | Sí | Contenido de `metrics.jsonl` (tiempos + calidad por chunk). |
| `GET` | `/api/jobs/{job_id}/audio-segment` | `jobs_handler::obtener_segmento_audio_handler` | Sí | Transcodea al vuelo un rango `[start,end]` del audio original a mp3. |
| `POST` | `/api/search` | `rag_handler::buscar_handler` | Sí | Retrieval puro (equivalente REST de `search_transcript`). |
| `POST` | `/api/rag/answer` | `rag_handler::rag_answer_handler` | Sí | Retrieval + generación (equivalente REST de `rag_answer`). |
| `*` | `/mcp` | `mcp::build_service` (Streamable HTTP, `rmcp`) | Sí | Servidor MCP de solo lectura — ver sección propia. |

Cualquier otra ruta devuelve `404` (comportamiento default de Axum, verificado en `router::tests::ruta_desconocida_devuelve_404`). "Auth" se explica en la sección "Autenticación" más abajo — hoy es opcional (solo aplica si `MCP_BEARER_TOKEN` está seteado).

---

## `POST /api/upload-audio`

Sube un archivo de audio, crea un job nuevo (`job_id` UUID v4) y dispara el pipeline completo en background (Fase 2 Whisper → Fase 3 persistencia → Fase 4 embeddings). No espera a que el pipeline termine — responde apenas el archivo terminó de escribirse a disco.

### Request

- `Content-Type: multipart/form-data; boundary=...`
- Un único campo de archivo (el nombre del campo no se valida — el handler simplemente toma el primer campo del multipart con `multipart.next_field()`). El `filename` que manda el cliente se usa solo para logging, **nunca** para construir una ruta de disco.
- El **formato real se detecta por magic bytes del primer chunk recibido**, no por la extensión del nombre de archivo ni por el `Content-Type` declarado:
  - `ID3` al inicio, o frame sync MPEG (`0xFF` seguido de un byte con los 3 bits altos en `1`) → tratado como `mp3`.
  - Caja ISO-BMFF `ftyp` en el offset 4 (`bytes[4..8] == "ftyp"`) → tratado como `mp4` (cubre también `.m4a`, mismo contenedor).
  - `RIFF....WAVE` en los primeros 12 bytes → tratado como `wav` (agregado 2026-07-30; Symphonia lo decodifica nativo, sin pasar por el fallback de ffmpeg).
  - Cualquier otra cosa → `400`, sin crear directorio de job ni escribir nada a disco.
- Límite explícito de **1 GiB** (`router::UPLOAD_BODY_LIMIT_BYTES`, `DefaultBodyLimit::max(...)` aplicado solo a esta ruta, corregido 2026-07-31 — antes corría con el default de Axum, 2 MiB, que truncaba en silencio cualquier audio real de más de un par de minutos). El límite no implica bufferear en RAM: el archivo se sigue escribiendo a disco en streaming, chunk por chunk (`campo.chunk().await` + `write_all` por chunk), nunca se buferea completo en memoria — 1 GiB deja margen cómodo incluso para el caso más pesado hoy soportado (`.wav` PCM sin comprimir, ~550MB para 5h a 16kHz mono).

### Respuestas

| Código | Cuándo | Body |
|---|---|---|
| `202 Accepted` | Archivo válido, guardado, pipeline lanzado en background. | `{"job_id": "<uuid>", "status": "processing"}` |
| `400 Bad Request` | Multipart ilegible, campo vacío, magic bytes no reconocidos (ni mp3, mp4 ni wav), o error leyendo el body a mitad de la subida (conexión cortada, o el archivo supera 1 GiB). | Texto plano (`"Error al leer el archivo multipart"` / `"Archivo vacío o inválido"` / `"Formato no soportado: solo se aceptan mp3, mp4/m4a y wav"` / `"Error leyendo el archivo subido (...)"`). En el último caso se limpia (`remove_dir_all`) el directorio del job a medio subir — antes de esta corrección (2026-07-31), un error acá se trataba como fin del stream y el archivo truncado se guardaba igual con `202`. |
| `500 Internal Server Error` | Falla creando el directorio del job, creando el archivo en disco, o escribiendo un chunk. | Texto plano (`"Error interno del servidor"` / `"Error al guardar el archivo"`). Si falla escribiendo, se intenta limpiar (`remove_dir_all`) el directorio del job a medio crear. |

`status: "processing"` en la respuesta es un valor fijo del handler, **no** refleja `job.json.status` — para el status real (`Pending`/`Processing`/`Completed`/`Failed`) usar `GET /api/jobs/{job_id}` (ver más abajo). No confundir uno con otro al construir el cliente.

### Qué dispara en background (no forma parte de la respuesta HTTP)

1. Adquiere 1 permiso de `heavy_compute_semaphore` (espera si ya hay una transcripción corriendo, o si una consulta RAG vía MCP/REST está usando el permiso). `job.json.status` se marca `Processing` recién acá, no antes — mientras el job espera el permiso, su `status` sigue en `Pending` (decisión explícita: `Processing` significa "corriendo activamente", no "encolado").
2. `run_pipeline` (Whisper, Fase 2/3) dentro de `spawn_blocking` — escribe `transcript.jsonl` incrementalmente y `checkpoint.json` tras cada chunk. Si termina en `Ok(Ok(()))`, `job.json.transcript_ready` pasa a `true` (independiente de si Fase 4 después falla).
3. Si Fase 2/3 terminó bien, encadena `run_embedding_phase` (Fase 4, embeddings a Qdrant) como tarea async normal. Si termina en `Ok(())`, `job.json.status` pasa a `Completed`.
4. Un error en cualquiera de los dos pasos (o un join error del `spawn_blocking`) se loguea por `eprintln!` **y** marca `job.json.status` como `Failed` — no hay reintento automático, pero el fallo ya no queda solo en logs.

`GET /api/jobs/{job_id}` es el endpoint para consultar el progreso/resultado de un job vía HTTP; `GET /api/jobs/{job_id}/transcript` sirve el contenido de `transcript.jsonl` — ver ambos más abajo.

### Ejemplo (curl)

```bash
curl -i -X POST http://localhost:3000/api/upload-audio \
  -F "file=@sample_Media/Muestra2_02min.m4a;type=audio/mp4"
```

---

## `POST /api/jobs/{job_id}/resume`

Reanuda un job existente cuyo procesamiento se cortó a mitad de camino (proceso matado, crash). Reusa exactamente la misma cadena que `POST /api/upload-audio` (`lanzar_procesamiento_job`) a partir del `job.json` ya existente — no crea un job nuevo.

Deliberadamente **no** existe como tool MCP (ver `CLAUDE.local.md`: "MCP de solo lectura" — ninguna tool de MCP puede disparar procesamiento, ni para "solo continuar" algo ya pedido).

### Request

- Sin body.
- `job_id` como path param — se valida como UUID bien formado (`Uuid::parse_str`) **antes** de tocar el filesystem. Esto también cierra path traversal: un `job_id` como `../../etc` nunca llega a construirse en una ruta de disco.

### Respuestas

| Código | Cuándo | Body |
|---|---|---|
| `202 Accepted` | `job.json` existe y es válido; pipeline relanzado en background. | `{"job_id": "<uuid>", "status": "processing"}` |
| `404 Not Found` | `job_id` no es un UUID válido, o no existe `./jobs/{job_id}/job.json`, o es ilegible/corrupto. | Texto plano `"Job no encontrado"`. |

### Idempotencia y seguridad de reintento

- Seguro de invocar más de una vez seguidas, o sobre un job ya terminado: el semáforo serializa contra cualquier otra transcripción en curso; si el checkpoint ya está al final del audio, el decoder llega a EOF casi de inmediato (no reprocesa nada); `run_embedding_phase` es idempotente (point ID determinístico `Uuid::new_v5`), así que reintentar Fase 4 tampoco duplica vectores en Qdrant.
- `JsonlWriter` abre siempre en modo `append` (nunca trunca) — precondición para que un resume real no borre las líneas ya escritas antes del corte (bug real encontrado y corregido en Fase 3, ver `CLAUDE.local.md`).

### Ejemplo (curl)

```bash
curl -i -X POST http://localhost:3000/api/jobs/1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a/resume
```

---

## Endpoints REST de lectura (`/api/jobs*`, `/api/search`, `/api/rag/answer`)

Wrappers REST bajo `/api/*` para que un cliente web no tenga que hablar el protocolo MCP completo (handshake `initialize`/sesiones) solo para leer. Todos comparten el mismo middleware de bearer token que `/mcp` (ver "Autenticación" en la sección de `/mcp`) y reusan directamente las mismas funciones internas que las tools de MCP equivalentes — sin lógica propia duplicada.

### `GET /api/jobs`

Lista todos los jobs — equivalente REST de la tool MCP `list_audios`. Reusa `audio_pipeline::job::list_jobs` (compartida con `mcp::listar_jobs`).

- **Output**: `200` con un array de `JobSummary`:
  ```json
  [
    {
      "job_id": "1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a",
      "status": "Processing",
      "transcript_ready": false,
      "created_at": "1753700000",
      "processing_started_at": "1753700005",
      "transcript_ready_at": null,
      "completed_at": null,
      "summary_status": "NotStarted",
      "summary": null
    }
  ]
  ```
  A diferencia de `list_audios`/`get_audio_metadata` de MCP, **no** incluye `audio_path`/`transcript_path`/`checkpoint_path` (rutas de disco del servidor) — filtrado desde el día uno para este endpoint. Sin paginación. `processing_started_at`/`transcript_ready_at`/`completed_at` (agregados 2026-07-31, epoch-segundos como string, `null` si esa fase todavía no ocurrió) permiten calcular cuánto tardó cada fase — ver `CLAUDE.local.md`: logger de métricas. `summary_status`/`summary` (agregados 2026-08-01): `summary_status` es uno de `NotStarted | Generating | Ready | Failed`; `summary` trae el texto del resumen solo cuando `summary_status == "Ready"`, `null` en cualquier otro caso — ver `CLAUDE.local.md`: "Resumen por audio".

### `GET /api/jobs/{job_id}`

Status y progreso de un job puntual.

- `job_id` como path param, validado como UUID (`audio_pipeline::job::load_job`) antes de tocar el filesystem — mismo criterio que `resume`.
- **Respuestas**:

  | Código | Cuándo | Body |
  |---|---|---|
  | `200 OK` | El job existe, sea cual sea su `status`. | `JobSummary` + `last_chunk`/`processed_seconds` (ver abajo). |
  | `404 Not Found` | `job_id` no es un UUID válido, o no existe el job. | Texto plano `"Job no encontrado"`. |

- Ejemplo de body `200`:
  ```json
  {
    "job_id": "1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a",
    "status": "Processing",
    "transcript_ready": false,
    "created_at": "1753700000",
    "processing_started_at": "1753700005",
    "transcript_ready_at": null,
    "completed_at": null,
    "summary_status": "NotStarted",
    "summary": null,
    "last_chunk": 4,
    "processed_seconds": 150.0
  }
  ```
- `last_chunk`/`processed_seconds` se leen de `checkpoint.json` (`CheckpointManager::load`) — `(0, 0.0)` si el job todavía no generó ningún checkpoint (recién creado, o esperando el permiso de `heavy_compute_semaphore`), no es un error.
- **Gap conocido, sin resolver**: si el proceso completo del servidor muere a mitad de una transcripción, ningún código marca el job como `Failed` — queda en `Processing` indefinidamente, indistinguible de una transcripción larga legítima en curso. Detectar esto por `mtime` de `checkpoint.json` está documentado como pendiente en `docs/TODO.md`, no implementado.

### `GET /api/jobs/{job_id}/transcript`

Contenido de `transcript.jsonl`, leído en streaming (`BufReader::lines()`, nunca el archivo completo de una vez).

- Mismo criterio de `job_id`/`404` que el endpoint anterior.
- **Siempre `200` si el job existe** — nunca `202`/`400`/`425` para comunicar "todavía no hay transcript". El cliente decide qué hacer mirando `status`/`transcript_ready` en el body, no el código HTTP.
- Ejemplo de body (job recién creado, sin transcript todavía):
  ```json
  {"status": "Pending", "transcript_ready": false, "entries": []}
  ```
- Ejemplo de body (con contenido):
  ```json
  {
    "status": "Processing",
    "transcript_ready": false,
    "entries": [
      {"chunk": 0, "start": 0.0, "end": 30.0, "text": "...", "avg_logprob": -0.2, "low_confidence": false}
    ]
  }
  ```
  `low_confidence` (agregado 2026-08-01) es un campo calculado, no persistido en `transcript.jsonl` — `true` cuando `avg_logprob < LOW_CONFIDENCE_THOLD` (-0.6, ver `CLAUDE.local.md`). Se recalcula en cada lectura, así que cambiar el umbral no requiere reprocesar nada.

### `GET /api/jobs/{job_id}/metrics`

Contenido de `metrics.jsonl` (agregado 2026-07-31, ver `CLAUDE.local.md`: logger de métricas) — tiempos por etapa y señales de calidad de whisper-rs por chunk, separado de `transcript.jsonl` a propósito (diagnóstico del pipeline, no contenido semántico). Leído en streaming igual que el endpoint anterior.

- Mismo criterio de `job_id`/`404` que los demás endpoints de job. **Siempre `200` si el job existe**, `entries: []` si `metrics.jsonl` todavía no existe en disco (pipeline no arrancó a escribir, o corre una versión anterior a este cambio).
- Ejemplo de body:
  ```json
  {
    "entries": [
      {
        "chunk": 0, "start": 0.0, "end": 30.0,
        "decode_ms": 12, "whisper_ms": 4300, "persist_ms": 1, "total_ms": 4313,
        "text_len": 540, "avg_logprob": -0.42, "score": 0.66,
        "no_speech_prob": 0.02, "entropy": 3.1, "segment_count": 4
      }
    ]
  }
  ```
- `score` es `exp(avg_logprob)` (confianza aproximada 0-1). `entropy` reproduce la fórmula interna de whisper.cpp (Shannon entropy sobre frecuencia de `token_id()`, no un proxy) — ver `CLAUDE.local.md` para el detalle de por qué es fiel a la fórmula real pero no a la ventana exacta de 32 tokens que usa whisper.cpp internamente.

### `GET /api/jobs/{job_id}/audio-segment`

Transcodea al vuelo con `ffmpeg` el rango `[start, end)` (segundos, query params) del audio **original** (no el PCM 16kHz que Whisper procesó) a mp3, streameado directo como body de la respuesta — agregado 2026-08-01, ver `CLAUDE.local.md`: "Reproducción de un segmento de audio". Pensado para que el cliente escuche un segmento marcado con `avg_logprob`/`low_confidence` bajo y verifique a oído si Whisper se equivocó.

- Mismo criterio de `job_id`/`404` que los demás endpoints de job.
- **Query params**: `start` y `end` (segundos, `f32`). Validación: `start >= 0`, `end > start`, `end - start <= 120` segundos.
- **Respuestas**:

  | Código | Cuándo | Body |
  |---|---|---|
  | `200 OK` | Rango válido. | Stream de audio, `Content-Type: audio/mpeg`. |
  | `400 Bad Request` | `start`/`end` inválidos, o rango > 120s. | Texto plano describiendo el problema. |
  | `404 Not Found` | `job_id` no es un UUID válido, o no existe el job. | Texto plano `"Job no encontrado"`. |
  | `500 Internal Server Error` | No se pudo ejecutar `ffmpeg` (¿no está en el PATH?). | Texto plano. |

- **Limitación conocida**: si `ffmpeg` falla a mitad de la transcodificación (input corrupto, rango fuera del audio real, etc.), el cliente ya recibió el header `200` — el audio llega truncado/corrupto en vez de un `500`, inherente a cualquier respuesta streaming. `ffmpeg` no pasa por `heavy_compute_semaphore` (carga de CPU corta, mismo criterio que el fallback de `ffmpeg` del decoder de Fase 2).

```bash
curl -i "http://localhost:3000/api/jobs/1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a/audio-segment?start=90&end=120" \
  -H "Authorization: Bearer <token>" -o segmento.mp3
```

### `POST /api/search`

Wrapper REST de la tool MCP `search_transcript` — retrieval puro, sin generación. Mismo body/output que la tool MCP (ver sección `/mcp` más abajo), mismo `heavy_compute_semaphore`, mismo top-k (`SEARCH_TOP_K = 8`).

```bash
curl -i -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "¿qué se acordó sobre el presupuesto?", "scope": "audio", "audio_id": "1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a"}'
```

### `POST /api/rag/answer`

Wrapper REST de la tool MCP `rag_answer` — retrieval + generación server-side vía Ollama. Mismo body shape (con `question` en vez de `query`), mismo semáforo.

- **Output**: `{"answer": "..."}` (texto plano generado por el modelo, dentro de un JSON de un solo campo).
- **Nota de latencia sin resolver**: una llamada de generación con el modelo de 7-8B en CPU puede tardar bastante — no hay timeout de request explícito ni streaming (SSE/WebSocket) implementado todavía.

---

## `/mcp` — servidor MCP de solo lectura

Transporte Streamable HTTP (`rmcp` 2.2.0), montado con `Router::nest_service("/mcp", ...)`. Pensado para clientes MCP genéricos (Claude Desktop, `@modelcontextprotocol/inspector`, o un cliente propio) — no es la superficie que la interfaz web debería llamar directo desde el browser sin más consideración (ver "Gaps relevantes para el cliente web" más abajo, en particular CORS).

Handshake estándar MCP: `initialize` → `notifications/initialized` → `tools/list` → `tools/call`. Ejemplos reproducibles en `docs/mcp-requests.http` (extensión REST Client de VSCode) y documentados paso a paso en `docs/MCP_Testing.md`.

### Autenticación

Middleware `exigir_bearer_token` (`router.rs`), aplicado a `/mcp` **y** a los endpoints REST de lectura de arriba (`GET /api/jobs`, `GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/transcript`, `GET /api/jobs/{job_id}/metrics`, `GET /api/jobs/{job_id}/audio-segment`, `POST /api/search`, `POST /api/rag/answer` — todos montados vía `router::crear_router_protegido`, mismo `route_layer`). `POST /api/upload-audio` y `POST /api/jobs/{job_id}/resume` (endpoints de escritura) quedan fuera de este grupo, sin autenticación, sin cambios.

- Si la variable de entorno `MCP_BEARER_TOKEN` está seteada: toda request a cualquiera de esas 8 rutas debe traer `Authorization: Bearer <token>` con el valor exacto, o responde `401 Unauthorized`.
- Si **no** está seteada: esas rutas quedan completamente sin autenticación (cualquiera que llegue a la IP:puerto puede llamarlas). Esto es aceptable solo mientras el servidor no se expone más allá de `localhost` — pero como se indicó arriba, el bind real hoy ya es `0.0.0.0`, así que en la práctica esto es un gap de seguridad activo, no solo teórico, en cuanto la máquina esté en una red no confiable.

### Tools disponibles

Las 4 tools son de **solo lectura** por diseño — ninguna escribe/borra en Qdrant ni dispara transcripciones nuevas (regla dura, ver `CLAUDE.local.md`).

**`search_transcript` y `rag_answer` esperan el mismo `heavy_compute_semaphore` que usa Whisper** (2026-07-29) — si hay una transcripción en curso, la tool queda esperando el permiso antes de correr, potencialmente varios minutos en un audio largo. `list_audios`/`get_audio_metadata` no esperan nada (solo leen `job.json` de disco).

#### `search_transcript`

Retrieval puro (sin generación) — devuelve los chunks más relevantes con su score.

- **Input** (JSON, `#[serde(tag = "scope")]` — el `scope` es obligatorio, sin default):
  ```json
  {"query": "texto de la pregunta", "scope": "audio", "audio_id": "<uuid>"}
  ```
  o
  ```json
  {"query": "texto de la pregunta", "scope": "all_corpus"}
  ```
- `scope: "all_corpus"` activa el reranker de cross-encoder (`BAAI/bge-reranker-v2-m3` vía `candle-transformers`) sobre un pool de 30 candidatos antes de recortar al top-k. `scope: "audio"` nunca rerankea (bi-encoder solo).
- Top-k fijo: 8 (`SEARCH_TOP_K`, más generoso que el de `rag_answer` porque acá el caller ve los chunks crudos).
- **Output** (`CallToolResult` con un `ContentBlock::text` de JSON serializado):
  ```json
  {
    "hits": [
      {
        "audio_id": "...",
        "chunk_id": 3,
        "start": 90.0,
        "end": 120.0,
        "text": "...",
        "speaker": "unknown",
        "avg_logprob": -0.42,
        "score": 0.83
      }
    ]
  }
  ```
  `score` es `null` en los chunks vecinos agregados por context assembly (`chunk_id ± 1`), que nunca pasaron por la búsqueda vectorial ni el reranker.

#### `rag_answer`

Retrieval + generación server-side vía Ollama (`qcwind/qwen2.5-7b-instruct-Q4_K_M:latest`) — el motor de RAG por defecto del proyecto.

- **Input**: mismo patrón de `scope` obligatorio, con `question` en vez de `query`:
  ```json
  {"question": "texto de la pregunta", "scope": "audio", "audio_id": "<uuid>"}
  ```
- **Output**: `CallToolResult` con un único `ContentBlock::text` — la respuesta en texto plano generada por el modelo (no un JSON estructurado).
- El contexto interno que arma el prompt incluye `[audio | chunk | start-end]` por pasaje y marca `[BAJA CONFIANZA]` los tramos con `avg_logprob < -0.8`, pero eso no es visible al caller de la tool — solo llega la respuesta final.

#### `list_audios`

Sin parámetros. Enumera `./jobs/*`, lee cada `job.json` vía el mismo loader validado que usa el endpoint de resume (`audio_pipeline::job::load_job`). Un directorio con `job.json` ilegible se saltea (con log en stderr del servidor) en vez de abortar el listado completo.

- **Output**: array JSON de `JobMetadata` (ver schema abajo) — **incluye las 3 rutas de disco internas tal cual** (`audio_path`, `transcript_path`, `checkpoint_path`). Ver "Gaps conocidos" — esto está identificado como pendiente de filtrar antes de exponer en LAN.

#### `get_audio_metadata`

- **Input**: `{"audio_id": "<uuid>"}`
- **Output**: un único `JobMetadata` (mismo schema completo, mismas rutas de disco expuestas).
- Si `audio_id` no es un UUID válido o no existe el job: error JSON-RPC `-32602` (`invalid_params`), no un `CallToolResult` de error — importante para el cliente: hay que manejar el error a nivel de protocolo MCP, no solo inspeccionar el body de un `CallToolResult`.

### Schema de `JobMetadata` (el que devuelven `list_audios`/`get_audio_metadata`)

```json
{
  "job_id": "1b6f5e2a-4c3d-4a11-9e2b-0a1c2d3e4f5a",
  "audio_path": "./jobs/1b6f5e2a-.../audio.mp4",
  "transcript_path": "./jobs/1b6f5e2a-.../transcript.jsonl",
  "checkpoint_path": "./jobs/1b6f5e2a-.../checkpoint.json",
  "status": "Processing",
  "created_at": "1753700000",
  "transcript_ready": false,
  "processing_started_at": "1753700005",
  "transcript_ready_at": null,
  "completed_at": null,
  "summary_status": "NotStarted"
}
```

- `status` es uno de `"Pending" | "Processing" | "Completed" | "Failed"` (enum `JobStatus`, `src/audio_pipeline/models.rs`) — actualizado en las transiciones reales del pipeline desde 2026-07-30 (ver `docs/TODO.md`). `Pending` mientras espera el permiso de `heavy_compute_semaphore`, `Processing` mientras corre activamente, `Completed`/`Failed` al terminar.
- `summary_status` (agregado 2026-08-01) es uno de `"NotStarted" | "Generating" | "Ready" | "Failed"` — a diferencia de los endpoints REST (`JobSummary`), `list_audios`/`get_audio_metadata` de MCP **no** incluyen el texto del resumen en sí (`summary`), solo este bookkeeping — leerlo requeriría un campo adicional a agregar a las tools de MCP si hiciera falta a futuro.
- `transcript_ready` (agregado 2026-07-30, `#[serde(default)] = false` para compatibilidad con `job.json` viejos sin el campo): `true` en cuanto Fase 2/3 (Whisper + persistencia) termina bien, independiente de si Fase 4 (embeddings) todavía está en curso o falla después.
- `created_at` es un epoch en segundos, serializado como **string**, no número.
- Los endpoints REST (`GET /api/jobs`, `GET /api/jobs/{job_id}`) devuelven un `JobSummary` filtrado (sin las 3 rutas de disco) en vez de este schema completo — ver sección propia más arriba. `list_audios`/`get_audio_metadata` de MCP siguen devolviendo el `JobMetadata` completo, con las rutas de disco sin filtrar (ver "Gaps conocidos").

---

## Schemas de datos relevantes para el cliente

### `TranscriptEntry` (una línea de `transcript.jsonl`, no expuesto directo por ningún endpoint hoy — solo en disco)

```json
{"chunk": 3, "start": 90.0, "end": 120.0, "text": "...", "avg_logprob": -0.42}
```

### Payload de un punto en Qdrant (lo que subyace a `ChunkPayload`/los `hits` de `search_transcript`)

```json
{
  "audio_id": "...",
  "chunk_id": 0,
  "start": 0.0,
  "end": 30.0,
  "text": "...",
  "speaker": "unknown",
  "avg_logprob": 0.0
}
```

`speaker` está fijo en `"unknown"` en toda la base actual — tdrz (turn-detection) no está activo con el modelo `ggml-small-q5_1.bin` en uso (ver `CLAUDE.local.md`).

---

## Gaps conocidos relevantes para el cliente web y el testing (no implementados todavía)

Extraído de `docs/TODO.md` y verificado contra el código actual — listado acá porque afecta directamente qué puede hacer un cliente HTTP/MCP hoy:

- **`list_audios`/`get_audio_metadata` de MCP exponen rutas de disco del servidor** (`audio_path`/`transcript_path`/`checkpoint_path`) sin filtrar. No es una fuga de secretos, pero es estructura interna innecesaria para un cliente remoto. Los endpoints REST equivalentes (`GET /api/jobs`, `GET /api/jobs/{job_id}`) ya filtran esos campos desde el día uno — este gap sigue existiendo solo en el lado MCP.
- **Sin CORS configurado**: ni en `/mcp` ni en los endpoints REST nuevos — si el cliente web futuro los llama directo desde JS del browser (no a través de un backend propio), va a necesitar headers CORS (`Access-Control-Allow-Origin`, exponer `Mcp-Session-Id` para `/mcp`) que hoy no existen en el router.
- **`allowed_hosts` de `rmcp` limitado a `localhost`/`127.0.0.1`/`::1`** (default del SDK, protección anti DNS-rebinding): un cliente conectando por la IP de LAN real recibe `403 Forbidden` hasta que se amplíe explícitamente con `.with_allowed_hosts([...])`. Solo afecta a `/mcp` (transporte `rmcp`), no a los endpoints `/api/*` (Axum plano, sin ese chequeo). Relevante ni bien se pruebe el servidor desde otro dispositivo en la misma red.
- **Bearer token opcional, no obligatorio** — y el servidor ya bindea `0.0.0.0` (ver "Arranque del servidor"). Antes de probar desde otro dispositivo en la LAN, como mínimo setear `MCP_BEARER_TOKEN` (cubre `/mcp` y los endpoints REST protegidos por igual, ver "Autenticación").
- **"Jobs atascados"**: si el proceso completo del servidor muere a mitad de una transcripción (no un `Err` capturado dentro de Rust, sino el binario cayendo), ningún código marca el job como `Failed` — queda en `Processing` indefinidamente. Detectar esto por `mtime` de `checkpoint.json` está documentado como enfoque en `docs/TODO.md`, sin umbral fijado ni implementado todavía.
- **`job.json` no tiene ningún campo "amigable" para UI** (nombre de archivo original, título): un selector de audios en el cliente web va a mostrar UUIDs pelados hasta que se agregue.
- **Sin `CancellationToken`/graceful shutdown** para las sesiones MCP — matar el proceso no las cierra prolijamente; no bloquea testing manual pero puede dar falsos timeouts en un cliente si el proceso se reinicia con una sesión abierta.
- ~~`POST /api/upload-audio` no tiene límite de tamaño de archivo explícito~~ — **resuelto 2026-07-31**: límite explícito de 1 GiB, scoped solo a esta ruta (ver sección propia más arriba). Encontrado como bug real, no solo teórico: el default de Axum (2 MiB) truncaba en silencio audios de más de un par de minutos, y el handler trataba el error de lectura resultante como fin del stream normal en vez de reportarlo — el archivo truncado se guardaba con `202 Accepted` como si la subida hubiera sido exitosa, y solo fallaba horas después al decodificarlo en Fase 2 (`Symphonia`/`ffprobe`: "missing moov atom" / "Invalid data found"). Ambos bugs corregidos juntos.
- **`POST /api/rag/answer` (REST y MCP) sin timeout de request ni streaming**: una generación con el modelo de 7-8B en CPU puede tardar bastante; no hay número medido documentado ni mitigación implementada.

## Cómo probar el estado actual

- **REST**: `docs/mcp-requests.http` (extensión REST Client de VSCode) cubre el handshake MCP completo. Para `/api/upload-audio` y `/api/jobs/{job_id}/resume` no hay un `.http` documentado todavía — usar los ejemplos `curl` de arriba.
- **Audios de prueba**: `sample_Media/` (no versionado en git, son grabaciones reales). El 100% de las muestras verificadas son en realidad AMR-NB a 8kHz mono dentro de un contenedor MP4/3GP, sin importar la extensión declarada (`.mp3` o `.m4a`) — ver `CLAUDE.local.md`. `ffmpeg`/`ffprobe` tienen que estar en el `PATH` para que estos archivos se puedan decodificar (Symphonia no tiene decoder de AMR-NB).
- **Tests de integración `#[ignore]`** (requieren Whisper + Ollama + Qdrant reales corriendo, no corren en CI por default):
  - `cargo test pipeline_hardcodeado -- --ignored --nocapture` (`router.rs`) — sube un audio real vía el endpoint HTTP, espera la transcripción, confirma los embeddings en Qdrant, limpia sus propios puntos.
  - `cargo test embeddings -- --ignored --nocapture` (`audio_pipeline/embeddings.rs`).
  - `cargo test rag_answer -- --ignored --nocapture` (`rag/generation.rs`).
  - `cargo test rerank_reordena -- --ignored --nocapture` (`rag/reranker.rs`, no depende de Qdrant/Ollama).
  - Variable `TEST_AUDIO_PATH` permite apuntar `pipeline_hardcodeado` a cualquier archivo de `sample_Media/` sin tocar código (default: `sample_Media/Muestra2_02min.m4a`, ~2 min, para iteración rápida).
- **Requisitos previos para cualquier prueba end-to-end**: Ollama corriendo en `127.0.0.1:11434` con `bge-m3` y `qcwind/qwen2.5-7b-instruct-Q4_K_M:latest` ya descargados (`ollama pull`), y Qdrant corriendo en `127.0.0.1:6333`/`6334` (contenedor Docker `qdrant/qdrant`, bindeado solo a localhost — nunca `0.0.0.0`).
