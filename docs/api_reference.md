# Referencia de API

Estado actual de la superficie HTTP del servidor (`src/main.rs`, `src/router.rs`, `src/handlers/audio_handler.rs`, `src/mcp/mod.rs`), documentado como base para: (1) construir el cliente web, (2) probar el servidor con distintos audios, (3) tener un punto de referencia único de qué existe hoy realmente, sin desactualizarse con lo que dice `docs/Arquitechture.md` a nivel de diseño. Para el "por qué" de cada decisión, ver `CLAUDE.local.md` y `docs/Arquitechture.md`; este documento se limita al "qué" — rutas, payloads, códigos de estado.

## Arranque del servidor

`src/main.rs`:

- Bind: `0.0.0.0:3000` — el servidor ya escucha en todas las interfaces (LAN incluida), no solo `localhost`. Ver "Exposición en LAN — gaps" más abajo antes de considerar esto seguro para un cliente remoto.
- Clientes construidos una sola vez en `main.rs` y compartidos vía `Arc<AppState>` (`src/state.rs`): `Ollama::default()` (`http://127.0.0.1:11434`) y `Qdrant::from_url("http://localhost:6334")`. Son clientes livianos (no cargan modelos) — construirlos al boot no viola la regla de "nunca dejar Whisper/Ollama residentes".
- `transcription_semaphore` (`Semaphore::new(1)`): como máximo una transcripción pesada corriendo a la vez, en todo el proceso — compartido entre `POST /api/upload-audio` y `POST /api/jobs/{job_id}/resume`.
- Si `MCP_BEARER_TOKEN` no está seteado en el entorno, el proceso imprime una advertencia por consola al arrancar (no rechaza arrancar, no rechaza bindear a `0.0.0.0`) — ver "Autenticación" más abajo.

No hay healthcheck (`GET /health` o similar) implementado todavía.

## Rutas registradas

`src/router.rs::crear_router` monta dos grupos de rutas al mismo nivel (`Router::merge`, no `nest`):

| Método | Ruta | Handler | Descripción corta |
|---|---|---|---|
| `POST` | `/api/upload-audio` | `recibir_y_procesar_audio` | Sube un audio nuevo, crea un job, dispara el pipeline. |
| `POST` | `/api/jobs/{job_id}/resume` | `reanudar_job` | Reanuda un job existente cortado a mitad de camino. |
| `*` | `/mcp` | `mcp::build_service` (Streamable HTTP, `rmcp`) | Servidor MCP de solo lectura — ver sección propia. |

Cualquier otra ruta devuelve `404` (comportamiento default de Axum, verificado en `router::tests::ruta_desconocida_devuelve_404`).

---

## `POST /api/upload-audio`

Sube un archivo de audio, crea un job nuevo (`job_id` UUID v4) y dispara el pipeline completo en background (Fase 2 Whisper → Fase 3 persistencia → Fase 4 embeddings). No espera a que el pipeline termine — responde apenas el archivo terminó de escribirse a disco.

### Request

- `Content-Type: multipart/form-data; boundary=...`
- Un único campo de archivo (el nombre del campo no se valida — el handler simplemente toma el primer campo del multipart con `multipart.next_field()`). El `filename` que manda el cliente se usa solo para logging, **nunca** para construir una ruta de disco.
- El **formato real se detecta por magic bytes del primer chunk recibido**, no por la extensión del nombre de archivo ni por el `Content-Type` declarado:
  - `ID3` al inicio, o frame sync MPEG (`0xFF` seguido de un byte con los 3 bits altos en `1`) → tratado como `mp3`.
  - Caja ISO-BMFF `ftyp` en el offset 4 (`bytes[4..8] == "ftyp"`) → tratado como `mp4` (cubre también `.m4a`, mismo contenedor).
  - Cualquier otra cosa → `400`, sin crear directorio de job ni escribir nada a disco.
- No hay límite de tamaño de archivo explícito en el handler (Axum tiene un límite default de body, ver `DefaultBodyLimit` — no está configurado explícitamente en `router.rs`, así que corre con el default de Axum). El archivo se escribe a disco en streaming, chunk por chunk (`campo.chunk().await`), nunca se buferea completo en memoria.

### Respuestas

| Código | Cuándo | Body |
|---|---|---|
| `202 Accepted` | Archivo válido, guardado, pipeline lanzado en background. | `{"job_id": "<uuid>", "status": "processing"}` |
| `400 Bad Request` | Multipart ilegible, campo vacío, o magic bytes no reconocidos (ni mp3 ni mp4). | Texto plano (`"Error al leer el archivo multipart"` / `"Archivo vacío o inválido"` / `"Formato no soportado: solo se aceptan mp3 y mp4/m4a"`). |
| `500 Internal Server Error` | Falla creando el directorio del job, creando el archivo en disco, o escribiendo un chunk. | Texto plano (`"Error interno del servidor"` / `"Error al guardar el archivo"`). Si falla escribiendo, se intenta limpiar (`remove_dir_all`) el directorio del job a medio crear. |

`status: "processing"` en la respuesta es un valor fijo del handler, **no** refleja `job.json.status` (que en la práctica queda en `"Pending"` para siempre hoy — ver "Gaps conocidos" más abajo). No confundir uno con otro al construir el cliente.

### Qué dispara en background (no forma parte de la respuesta HTTP)

1. Adquiere 1 permiso de `transcription_semaphore` (bloquea si ya hay una transcripción corriendo).
2. `run_pipeline` (Whisper, Fase 2/3) dentro de `spawn_blocking` — escribe `transcript.jsonl` incrementalmente y `checkpoint.json` tras cada chunk.
3. Si `run_pipeline` termina en `Ok(Ok(()))`, encadena `run_embedding_phase` (Fase 4, embeddings a Qdrant) como tarea async normal.
4. Cualquier error en cualquiera de los dos pasos solo se loguea por `eprintln!` — no hay reintento automático ni actualización de `job.json.status` a `Failed` (gap conocido, ver `docs/TODO.md` → Housekeeping).

No hay ningún endpoint todavía para consultar el progreso/resultado de un job vía HTTP — el único forma de ver el resultado hoy es leer `./jobs/{job_id}/transcript.jsonl` en disco directamente, o usar las tools de MCP `list_audios`/`get_audio_metadata` (que exponen `job.json`, no el transcript).

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

## `/mcp` — servidor MCP de solo lectura

Transporte Streamable HTTP (`rmcp` 2.2.0), montado con `Router::nest_service("/mcp", ...)`. Pensado para clientes MCP genéricos (Claude Desktop, `@modelcontextprotocol/inspector`, o un cliente propio) — no es la superficie que la interfaz web debería llamar directo desde el browser sin más consideración (ver "Gaps relevantes para el cliente web" más abajo, en particular CORS).

Handshake estándar MCP: `initialize` → `notifications/initialized` → `tools/list` → `tools/call`. Ejemplos reproducibles en `docs/mcp-requests.http` (extensión REST Client de VSCode) y documentados paso a paso en `docs/MCP_Testing.md`.

### Autenticación

Middleware `exigir_bearer_token` (`router.rs`), aplicado solo a `/mcp` (`route_layer`, no afecta `/api/*`):

- Si la variable de entorno `MCP_BEARER_TOKEN` está seteada: toda request a `/mcp` debe traer `Authorization: Bearer <token>` con el valor exacto, o responde `401 Unauthorized`.
- Si **no** está seteada: `/mcp` queda completamente sin autenticación (cualquiera que llegue a la IP:puerto puede llamar las tools). Esto es aceptable solo mientras el servidor no se expone más allá de `localhost` — pero como se indicó arriba, el bind real hoy ya es `0.0.0.0`, así que en la práctica esto es un gap de seguridad activo, no solo teórico, en cuanto la máquina esté en una red no confiable.

### Tools disponibles

Las 4 tools son de **solo lectura** por diseño — ninguna escribe/borra en Qdrant ni dispara transcripciones nuevas (regla dura, ver `CLAUDE.local.md`).

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
  "status": "Pending",
  "created_at": "1753700000"
}
```

- `status` es uno de `"Pending" | "Processing" | "Completed" | "Failed"` (enum `JobStatus`, `src/audio_pipeline/models.rs`) — **en la práctica hoy siempre vale `"Pending"`**, ver "Gaps conocidos".
- `created_at` es un epoch en segundos, serializado como **string**, no número.

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

Extraído de `docs/TODO.md` (sección Fase 6, gaps) y verificado contra el código actual — listado acá porque afecta directamente qué puede hacer un cliente HTTP/MCP hoy:

- **`job.json.status` nunca cambia de `"Pending"`**: ni `run_pipeline` ni `run_embedding_phase` actualizan el campo `status` en las transiciones reales (arranca/termina Whisper, terminan embeddings, falla algo). Un cliente no puede hoy distinguir "recién subido" de "transcripción completa" mirando `status` — tiene que inferirlo de otra forma (p.ej. polling de `transcript.jsonl`, o contar puntos en Qdrant vía `search_transcript`).
- **`list_audios`/`get_audio_metadata` exponen rutas de disco del servidor** (`audio_path`/`transcript_path`/`checkpoint_path`) sin filtrar. No es una fuga de secretos, pero es estructura interna innecesaria para un cliente remoto.
- **Sin CORS configurado en `/mcp`**: si el cliente web futuro llama a `/mcp` directo desde JS del browser (no a través de un backend propio), va a necesitar headers CORS (`Access-Control-Allow-Origin`, exponer `Mcp-Session-Id`) que hoy no existen en el router.
- **`allowed_hosts` de `rmcp` limitado a `localhost`/`127.0.0.1`/`::1`** (default del SDK, protección anti DNS-rebinding): un cliente conectando por la IP de LAN real recibe `403 Forbidden` hasta que se amplíe explícitamente con `.with_allowed_hosts([...])`. Relevante ni bien se pruebe el servidor desde otro dispositivo en la misma red.
- **Bearer token opcional, no obligatorio** — y el servidor ya bindea `0.0.0.0` (ver "Arranque del servidor"). Antes de probar desde otro dispositivo en la LAN, como mínimo setear `MCP_BEARER_TOKEN`.
- **Sin endpoint de progreso/estado para `POST /api/upload-audio`**: no hay forma de saber por HTTP si un job terminó sin leer archivos en disco o sin usar `search_transcript`/`get_audio_metadata` de MCP como proxy indirecto.
- **`job.json` no tiene ningún campo "amigable" para UI** (nombre de archivo original, título): un selector de audios en el cliente web va a mostrar UUIDs pelados hasta que se agregue.
- **Sin `CancellationToken`/graceful shutdown** para las sesiones MCP — matar el proceso no las cierra prolijamente; no bloquea testing manual pero puede dar falsos timeouts en un cliente si el proceso se reinicia con una sesión abierta.
- **`POST /api/upload-audio` no tiene límite de tamaño de archivo explícito** — corre con el `DefaultBodyLimit` de Axum, no configurado a mano en `router.rs`. Verificar el valor default de la versión de Axum en uso (0.8.9) antes de probar con audios de varias horas/GB, o subirlo explícitamente si hace falta.

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
