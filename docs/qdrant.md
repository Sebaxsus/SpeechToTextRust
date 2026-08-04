# Administrar Qdrant de este proyecto

Guía rápida para revisar a mano que Qdrant esté guardando la información que genera la Fase 4
(embeddings). Complementa `README.md` (instalación del contenedor) — esto es sobre *inspeccionar*
lo que ya está corriendo, no sobre instalarlo.

Todo lo de acá asume el contenedor levantado como indica el README:

```bash
docker run -d --name qdrant_local \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

El puerto `6333` es la API REST (HTTP), el `6334` es gRPC (el que usa `qdrant-client` desde el
código Rust). El dashboard y los `curl` de esta guía van todos contra `6333`. Como el contenedor
está bindeado solo a `127.0.0.1`, todo esto solo funciona desde la misma máquina donde corre
Docker — es la topología deliberada del proyecto (ver `CLAUDE.local.md`: Qdrant nunca se expone en
la LAN).

## 1. Abrir el dashboard

Con el contenedor corriendo, abrir en el navegador:

```
http://127.0.0.1:6333/dashboard
```

Es una UI web que Qdrant sirve solo desde el mismo puerto REST — no hace falta instalar nada
aparte. Ahí vas a ver la lista de colecciones; el proyecto usa una sola, llamada **`transcripts`**
(ver `src/audio_pipeline/embeddings.rs`, `COLLECTION_NAME`).

## 2. Ver la colección `transcripts`

Al entrar a la colección vas a encontrar, entre otras cosas:

- **`points_count`**: cuántos vectores hay guardados en total (todos los audios juntos — la
  colección es única para todo el corpus, no una por audio, ver `CLAUDE.local.md`: Qdrant —
  topología). Cada punto es un chunk de 30s de algún audio que ya pasó por Fase 4.
- **`segments_count`**: partes internas en las que Qdrant divide el almacenamiento para poder
  indexar/compactar en paralelo. No es algo que tengas que administrar a mano; crece solo con el
  volumen de datos.
- **`status`**: `green` = todo indexado y estable, sin trabajo pendiente. `yellow` = está
  optimizando/indexando en background (normal justo después de un upsert grande). `red` indicaría
  un problema real (poco frecuente, revisar logs del contenedor con `docker logs qdrant_local` si
  aparece).
- **Config de vectores**: `size: 1024` (dimensión de `bge-m3`), `distance: Cosine`, y vectores +
  payload marcados `on_disk` — coherente con lo que fija `ensure_collection()` en el código.

Si `transcripts` no aparece en la lista, es que Fase 4 todavía no corrió ni una vez con éxito (la
colección se crea perezosamente, la primera vez que `run_embedding_phase` llama a
`ensure_collection`).

## 3. Ver los puntos guardados (payload)

Dentro de la colección, la pestaña de puntos te deja navegar los vectores uno por uno. Cada punto
tiene un `payload` — los metadatos legibles que el pipeline le agrega a cada chunk (ver
`run_embedding_phase`):

```json
{
  "audio_id": "6be8a34c-2619-486d-ae29-98e0c6a028c8",
  "chunk_id": 12,
  "start": 360.0,
  "end": 390.0,
  "text": "...transcripción de ese chunk...",
  "speaker": "unknown",
  "avg_logprob": -0.42
}
```

El vector en sí (1024 floats) no es legible a simple vista — lo que importa para verificar que
"algo se guardó bien" es que el `payload` tenga el `text` correcto y el `audio_id` que esperás.

Para no scrollear a mano entre miles de puntos, la mayoría de los dashboards de Qdrant traen una
pestaña de **Console** (una consola REST embebida) donde podés escribir directamente el body de un
`scroll`/`search`/`count` y ver el resultado ahí mismo, sin salir del navegador — es el mismo tipo
de request que los `curl` de la sección siguiente.

## 4. Verificar por comando (sin abrir el navegador)

Info general de la colección:

```bash
curl http://127.0.0.1:6333/collections/transcripts
```

Contar cuántos puntos hay en total:

```bash
curl -X POST http://127.0.0.1:6333/collections/transcripts/points/count \
  -H "Content-Type: application/json" \
  -d '{"exact": true}'
```

Contar (o listar) solo los puntos de un `job_id`/`audio_id` puntual — útil después de subir un
audio, para confirmar que Fase 4 efectivamente insertó sus chunks:

```bash
# Contar
curl -X POST http://127.0.0.1:6333/collections/transcripts/points/count \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "audio_id", "match": {"value": "<job_id>"}}]}, "exact": true}'

# Traer los puntos completos (payload incluido, sin el vector)
curl -X POST http://127.0.0.1:6333/collections/transcripts/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "audio_id", "match": {"value": "<job_id>"}}]}, "with_payload": true, "with_vector": false, "limit": 10}'
```

Si el `count` coincide aproximadamente con la cantidad de chunks no-vacíos de
`GET /api/jobs/{job_id}/transcript` para ese mismo audio, Fase 4 corrió bien sobre ese job.

## 5. Persistencia real en disco

Los datos viven en el volumen Docker nombrado `qdrant_storage` (no en el filesystem del
contenedor), así que sobreviven a `docker stop`/`docker restart qdrant_local`. Para ver dónde está
ese volumen en el disco real de Windows:

```bash
docker volume inspect qdrant_storage
```

Los datos solo se pierden si el volumen se borra explícitamente (`docker volume rm qdrant_storage`)
— nunca por reiniciar el contenedor o la máquina.
