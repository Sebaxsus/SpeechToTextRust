# Probar el servidor MCP con REST Client (VSCode)

Guía para probar manualmente el servidor MCP de Fase 6 (`src/mcp/mod.rs`, montado en `/mcp` sobre
el Axum existente) usando la extensión **REST Client** de VSCode (`humao.rest-client`) y el
archivo [`docs/mcp-requests.http`](./mcp-requests.http) que acompaña esta guía. No hace falta un
cliente MCP real (Claude Desktop, etc.) para este flujo — son peticiones HTTP planas.

## Prerrequisitos

1. Extensión **REST Client** instalada en VSCode (`ext install humao.rest-client`).
2. Ollama corriendo con `bge-m3` y `qcwind/qwen2.5-7b-instruct-Q4_K_M` ya descargados (`ollama list` para confirmar).
3. Qdrant corriendo y accesible en `localhost:6334` (gRPC) / `localhost:6333` (REST).
4. Al menos un audio ya procesado (`./jobs/{job_id}/job.json` + embeddings en Qdrant) si querés
   probar `search_transcript`, `rag_answer`, `get_audio_metadata` o `get_transcript` con datos
   reales — `list_audios` funciona igual sin esto (devuelve una lista vacía). `get_transcript`
   además necesita que `transcript.jsonl` ya exista (no solo `job.json`), o devuelve `entries: []`.
5. Levantar el servidor: `cargo run`. La consola imprime la URL del endpoint MCP y si
   `MCP_BEARER_TOKEN` está configurado o no (ver "Autenticación" más abajo).

## Por qué el `.http` no es un simple GET/POST con JSON suelto

MCP (Model Context Protocol) sobre transporte **Streamable HTTP** no es un REST API convencional:
es **JSON-RPC 2.0** viajando por `POST` a una única ruta (`/mcp`), con una sesión que se abre con
un handshake de dos pasos y se referencia en cada pedido posterior por un header
`Mcp-Session-Id`. Por eso el archivo `.http` tiene las peticiones numeradas y hay que ejecutarlas
**en orden**: cada una (salvo la primera) depende del `Mcp-Session-Id` que abrió la petición 1.

REST Client resuelve esto con su sintaxis de "peticiones con nombre": la primera petición se marca
con `# @name initialize`, y las siguientes referencian su respuesta con
`{{initialize.response.headers.mcp-session-id}}`. No hay que copiar el session id a mano.

### El handshake (peticiones 1 y 2 del `.http`)

1. **`initialize`**: el cliente declara su `protocolVersion` y `clientInfo`. El servidor responde
   `200` con el header `Mcp-Session-Id` (la sesión recién creada) y confirma sus capacidades
   (`get_info` en `src/mcp/mod.rs` — este servidor solo declara `enable_tools()`, sin prompts ni
   resources).
2. **`notifications/initialized`**: notificación (sin `id`, no espera una respuesta con resultado)
   que cierra el handshake. El servidor responde `202 Accepted` con cuerpo vacío. A partir de acá
   la sesión queda lista para `tools/list` y `tools/call`.

### Por qué las respuestas se ven como `data: {...}` en vez de JSON plano

El servidor corre en **modo stateful** (`StreamableHttpServerConfig::default()`, sin overrides —
ver `mcp::build_service`), que es el modo estándar para sesiones persistentes multi-request. En
este modo las respuestas siempre vienen enmarcadas como **Server-Sent Events**
(`Content-Type: text/event-stream`), incluso para una respuesta única de request/response simple —
es una decisión del propio SDK (`rmcp`), no algo a corregir. REST Client las muestra igual como
texto crudo, ejemplo real de una respuesta a `tools/call`:

```
id: 0/0
retry: 3000
data: {"jsonrpc":"2.0","method":"notifications/message", ...}

id: 1/0
data: {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}]}}
```

El JSON-RPC que importa (la respuesta real a tu request) está en el **último bloque `data:`** — el
campo `"id"` ahí adentro es el `id` de JSON-RPC que vos mandaste (`3`, `4`, etc. en el `.http`), no
el `id: N/N` de la línea de SSE que lo envuelve (eso es un detalle interno del framing, no del
protocolo MCP).

## Las 5 tools expuestas

Definidas en `src/mcp/mod.rs`, todas de solo lectura (nunca escriben ni borran en Qdrant, nunca
disparan una transcripción):

| Tool | Argumentos | Qué hace |
|---|---|---|
| `search_transcript` | `query`, `scope` | Retrieval puro (Fase 5, sin generación): devuelve los chunks más relevantes con su score. |
| `rag_answer` | `question`, `scope` | Retrieval + generación server-side vía Ollama (el motor de RAG por defecto del proyecto). |
| `list_audios` | (ninguno) | Lista todos los `./jobs/*/job.json`. |
| `get_audio_metadata` | `audio_id` | Devuelve el `job.json` completo de un audio puntual. |
| `get_transcript` | `audio_id` | Devuelve el transcript **completo** (no un resumen, no un retrieval top-k) — pensada para que un cliente MCP externo (otro asistente de IA) genere su propio análisis del texto entero. |

### El campo `scope` es obligatorio, sin default a "buscar todo"

`search_transcript` y `rag_answer` requieren `scope` explícito, reflejando `SearchScope` de
`rag::retrieval` (ver `CLAUDE.local.md`: "Scope forzado, sin default implícito a todo el corpus").
Dos formas válidas, nunca una tercera implícita:

```json
{ "scope": "audio", "audio_id": "<job_id>" }
```
```json
{ "scope": "all_corpus" }
```

`all_corpus` activa el reranker de cross-encoder (`rag::reranker`, `BAAI/bge-reranker-v2-m3`) sobre
un pool de 30 candidatos — más lento, y la primera vez descarga los pesos (~2.2GB) si no están
todavía en la cache de Hugging Face (`~/.cache/huggingface/hub`).

## Autenticación (`MCP_BEARER_TOKEN`)

Si arrancaste el servidor con la variable de entorno `MCP_BEARER_TOKEN` configurada, `/mcp` exige
el header `Authorization: Bearer <token>` en **todas** las peticiones (incluida `initialize`) y
devuelve `401` sin ese header. El `.http` ya trae una línea `Authorization: Bearer {{token}}`
comentada en cada petición — completá `@token` al principio del archivo y descomentá esas líneas.

Sin `MCP_BEARER_TOKEN` configurado, `/mcp` queda sin autenticación — el servidor lo advierte por
consola al arrancar. Aceptable solo para pruebas locales; ver `CLAUDE.local.md` (sección "MCP de
solo lectura") antes de exponer el servidor más allá de `localhost`.

## Troubleshooting

- **`403 Forbidden`**: el header `Host` de la petición no está en la lista `allowed_hosts` del
  servidor (por defecto solo `localhost` / `127.0.0.1` / `::1`, protección anti DNS-rebinding del
  propio SDK). Usá `http://localhost:3000` en `@baseUrl`, no una IP de LAN, mientras se prueba en
  local.
- **`401 Unauthorized`**: falta o no coincide el header `Authorization` — ver "Autenticación" arriba.
- **`400` con `"Invalid Request"` mencionando `MCP-Protocol-Version`**: si agregás ese header a
  mano, tiene que coincidir exactamente con el `protocolVersion` que mandaste en el body de
  `initialize`. Más simple: no mandar ese header y dejar que el servidor asuma el default.
- **`tools/call` devuelve un error JSON-RPC en vez de un resultado**: por ejemplo
  `get_audio_metadata` con un `audio_id` inexistente responde con un error (`invalid_params`,
  mensaje `no existe el job '...'`) en vez de un `result` — es el comportamiento esperado, no un
  bug del cliente.
- **`rag_answer` tarda varios segundos**: es normal — está generando con un modelo de 7-8B en CPU
  (`qcwind/qwen2.5-7b-instruct-Q4_K_M`, sin `keep_alive`, se descarga de memoria al terminar).
