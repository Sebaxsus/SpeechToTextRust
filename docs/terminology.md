# Glosario

Definiciones de los términos técnicos usados en el proyecto y en `CLAUDE.local.md`, pensadas para alguien que retoma el código sin el contexto completo.

## Magic bytes

Los primeros bytes de un archivo, que identifican su formato *real* (contenedor/códec) independientemente de la extensión que declare el nombre de archivo — ej. `ID3` o un frame sync MPEG al inicio indican mp3; la caja `ftyp` de ISO-BMFF indica mp4/m4a. La Fase 1 de este proyecto valida por magic bytes, no por la extensión que manda el cliente, precisamente porque las grabaciones reales del dataset tienen extensiones engañosas (`.mp3`/`.m4a` que en realidad contienen AMR-NB dentro de un contenedor MP4/3GP — ver "Códec real de las grabaciones" en `CLAUDE.local.md`).

## ASR (Automatic Speech Recognition)

Reconocimiento automático de voz: convertir una señal de audio en texto. Whisper (y por lo tanto `whisper-rs`) es un modelo de ASR. En este proyecto es la Fase 2 del pipeline.

## Log-mel spectrogram

Representación tiempo-frecuencia del audio que espera Whisper como entrada (en vez de la forma de onda cruda). Se calcula con una STFT (ver abajo) y se mapea a 80 bandas de frecuencia según la escala mel (que aproxima cómo el oído humano percibe el tono). Whisper nunca "escucha" samples PCM directamente — siempre este espectrograma.

## STFT (Short-Time Fourier Transform)

Transformada de Fourier aplicada a ventanas cortas y solapadas de la señal (en vez de a la señal completa), para obtener cómo cambia el contenido frecuencial en el tiempo. Whisper usa ventanas de 25ms con salto (hop) de 10ms. Un "windowing consistente" entre chunks es necesario para que el espectrograma no tenga discontinuidades artificiales en los bordes.

## Hop size / window size

Parámetros de la STFT: `window` es el largo de cada segmento analizado (25ms en Whisper), `hop` es cuánto avanza la ventana entre un cálculo y el siguiente (10ms, o sea con solape). El tamaño de chunk de audio que se le pasa a Whisper debe ser múltiplo del hop size para no producir *spectral leakage* (energía de una frecuencia "filtrándose" a bandas vecinas por culpa de un corte abrupto).

## Chunk

Fragmento de audio de duración fija (30s en este proyecto) en el que se divide un audio largo antes de pasarlo a Whisper, porque no es viable ni deseable decodificar 5 horas de audio de una sola vez (ni en RAM ni en calidad de transcripción).

## Overlap

Región de audio compartida entre el final de un chunk y el comienzo del siguiente (2-3s en este proyecto), para que una palabra que cae justo en el borde del corte no se pierda o se corte a la mitad en ambos chunks.

## Crossfade

Técnica para suavizar la transición en la zona de overlap: se atenúa gradualmente el final del chunk A (`fade_out`) mientras se incrementa gradualmente el comienzo del chunk B (`fade_in`), típicamente con una ventana Hann o Tukey, para evitar un salto abrupto de energía en la costura.

## Nyquist (frecuencia de Nyquist)

La mitad de la frecuencia de muestreo de una señal digital. Es la frecuencia máxima que se puede representar sin *aliasing*. Si se resamplea audio de 44.1/48kHz a 16kHz sin filtrar antes las frecuencias por encima de 8kHz (la nueva Nyquist), esas frecuencias altas se "pliegan" sobre el espectro útil y contaminan el resultado — por eso el resampling a 16kHz para Whisper es obligatorio con filtro anti-aliasing (sinc, vía `rubato`), nunca decimación simple.

## Aliasing

Distorsión que ocurre cuando una señal se muestrea (o resamplea) por debajo del doble de su frecuencia máxima sin filtrar antes esas frecuencias altas: componentes de alta frecuencia aparecen disfrazados como frecuencias más bajas, indistinguibles del contenido real. En este proyecto, el riesgo concreto es degradar el mel-spectrogram que ve Whisper y por lo tanto la precisión de la transcripción.

## Resampling

Cambiar la frecuencia de muestreo de una señal (ej: de 48kHz a 16kHz). Symphonia decodifica el contenedor de audio pero no resamplea — eso lo hace explícitamente el pipeline con `rubato`.

## GGML / cuantización

Formato de pesos de modelo (usado por whisper.cpp / `whisper-rs`) optimizado para inferencia CPU-eficiente. La cuantización (ej. `q5_1`) reduce la precisión numérica de los pesos (de float32 a un formato de ~5 bits efectivos) para bajar el uso de RAM y acelerar la inferencia, a costa de algo de precisión — un trade-off relevante dado el límite de 16 GB de RAM del hardware objetivo.

## tinydiarize / tdrz

Variante de fine-tuning de Whisper (mismo modelo, mismos pesos base, reentrenado ligeramente) que reutiliza un token especial para marcar cambios de turno de hablante (*speaker turn detection*) dentro del texto decodificado. No es diarización completa: no asigna una identidad consistente a cada speaker (eso sería *speaker clustering*), solo indica "acá cambió quien habla". En `whisper-rs` se activa con `set_tdrz_enable(true)` y se lee con `next_segment_speaker_turn()` por segmento.

## Diarización (speaker diarization)

Proceso de determinar "quién habló cuándo" en un audio con múltiples hablantes. tdrz hace la parte de detectar *que* hubo un cambio de turno, pero no resuelve la identidad global del hablante (eso requeriría clustering de embeddings de voz, fuera del alcance actual).

## Cross-talk

Dos o más personas hablando al mismo tiempo. Es la condición normal esperada en las grabaciones de reuniones por teléfono de este proyecto (no un edge case). Whisper decodifica un solo stream de texto por segmento, así que en cross-talk real el resultado mezcla o pierde palabras de uno de los hablantes, y tdrz (que asume turnos discretos) es poco confiable ahí por diseño.

## avg_logprob

Media de `ln(token_probability())` sobre todos los tokens de un chunk transcrito por Whisper — mide qué tan "segura" estuvo la decodificación de que el texto generado es correcto, no la calidad objetiva de la transcripción en sí. Se calcula en `WhisperRunner::transcribe_chunk` y se persiste en el payload de Qdrant (Fase 4), para que la Fase 5 (RAG) pueda marcar un tramo como "BAJA CONFIANZA" (probable cross-talk o ruido de fondo) en vez de citarlo como transcripción limpia. No confundir con `logprob_thold` (abajo): `avg_logprob` es el valor final por chunk que queda persistido; `logprob_thold` es el umbral que usa Whisper *internamente* para decidir si reintentar una decodificación.

## logprob_thold

Parámetro de `FullParams` en whisper.cpp/`whisper-rs` (`set_logprob_thold`, fijado en `-0.8` en este proyecto): si el promedio de log-probabilidad de una decodificación cae por debajo de este umbral, whisper.cpp la descarta y reintenta con otra temperatura (se ve en los logs como `failed due to avg_logprobs ... < -0.80000`). Junto con `no_speech_thold` y `entropy_thold`, es uno de los mecanismos que evitan quedarse con una transcripción de baja confianza — relevante en este proyecto porque el audio de origen (teléfono, cross-talk, ruido) no es señal limpia.

## VAD (Voice Activity Detection)

Detección de qué partes de una señal de audio contienen voz humana vs. silencio/ruido. Permite saltar tramos vacíos antes de decodificarlos con Whisper, ahorrando CPU en audios largos con silencios. `whisper-rs` 0.16.0 expone `set_vad_model_path` / `enable_vad` para esto (no aplicado todavía, ver `docs/TODO.md`).

## RAG (Retrieval-Augmented Generation)

Técnica que combina búsqueda semántica (recuperar los fragmentos de texto más relevantes de una base de datos vectorial) con generación de texto por un LLM, para responder preguntas ancladas en un corpus específico (acá: las transcripciones) en vez de depender solo del conocimiento paramétrico del modelo.

## Embedding

Representación vectorial (de N dimensiones) de un fragmento de texto, generada por un modelo (acá, vía Ollama), tal que fragmentos con significado similar quedan cerca en ese espacio vectorial. Es lo que permite la búsqueda semántica en Qdrant.

## Bi-encoder

Modelo que codifica la query y el pasaje **por separado**, cada uno en un embedding independiente, y compara esos vectores después (típicamente con similitud del coseno). Es rápido porque los embeddings de los documentos se precalculan una sola vez (Fase 4) y quedan indexados en Qdrant — en tiempo de consulta solo hay que embeber la query y comparar contra vectores ya guardados. `bge-m3` es el bi-encoder de este proyecto. Contrastar con "Cross-encoder" abajo, mucho más preciso pero que no admite precálculo.

## Cross-encoder

Modelo que recibe la query y el pasaje **juntos**, concatenados en una sola entrada, y calcula su relevancia con atención cruzada completa entre ambos — más preciso que comparar dos embeddings independientes (bi-encoder), porque puede atender directamente a qué palabras de la query matchean con cuáles del pasaje. El costo es que no admite precálculo: hay que correr el modelo una vez por cada par (query, pasaje) candidato, así que solo se usa para reordenar un puñado de candidatos ya filtrados por el bi-encoder, nunca para buscar en todo el corpus desde cero. `bge-reranker-v2-m3` (ver "Reranker" abajo) es el cross-encoder de este proyecto.

## Similitud del coseno (cosine similarity)

Medida de qué tan parecida es la *dirección* de dos vectores (embeddings), ignorando su magnitud: `1.0` significa misma dirección (máxima similitud semántica según el bi-encoder), `0` ortogonales/sin relación, `-1.0` dirección opuesta. Es la métrica de distancia configurada en la colección `transcripts` de Qdrant (`Distance::Cosine`) para rankear qué chunks son más relevantes a una query antes de cualquier reranking.

## Reranker

Paso opcional de reordenamiento que corre **después** de la búsqueda vectorial inicial (bi-encoder) y **antes** de armar el contexto final para el LLM: toma los candidatos que ya trajo Qdrant y los reordena con un cross-encoder, más lento pero más preciso. Implementado en `src/rag/reranker.rs` (`BAAI/bge-reranker-v2-m3` corriendo in-process vía `candle-transformers`) y gateado a `SearchScope::AllCorpus` únicamente: con un solo audio (~600 chunks) el ranking del bi-encoder ya es casi exhaustivo, no vale la pena el costo de cargar el cross-encoder.

## top-k

La cantidad de resultados que se piden/conservan de una búsqueda — "las k mejores coincidencias". En este proyecto, `top_k` es un parámetro de `rag::retrieval::search` (default 6, rango 5-8 fijado en `CLAUDE.local.md`): con `SearchScope::Audio` es directamente cuántos puntos trae Qdrant; con `SearchScope::AllCorpus` se traen más candidatos de entrada (30, el pool para el reranker) y recién se recorta a `top_k` después de reordenar.

## Hits

Término genérico para "los resultados de una búsqueda" — cada punto que Qdrant devuelve para una query, antes o después de reranking/expansión de contexto. Ver `ChunkHit` abajo para el tipo concreto que usa el código de este proyecto.

## ChunkHit

Struct de `rag::retrieval` que envuelve un `ChunkPayload` (el chunk tal como está en Qdrant: `audio_id`, `chunk_id`, texto, timestamps, `avg_logprob`...) junto a un `score: Option<f32>`. El significado de `score` cambia según de dónde vino el hit: similitud coseno del bi-encoder si nunca pasó por el reranker, relevancia del cross-encoder si sí (el score se **reemplaza**, no se promedia), o `None` si es un chunk vecino agregado por context assembly — un chunk que nunca compitió en ninguna búsqueda, solo se sumó como contexto extra alrededor de un hit real.

## Corpus

El conjunto completo de transcripciones de todos los audios/jobs procesados por el sistema, todos indexados en la misma colección `transcripts` de Qdrant (ver "Point / Payload" y "HNSW" abajo). Es la unidad sobre la que opera `SearchScope::AllCorpus`, en contraposición a limitarse a un solo audio (`SearchScope::Audio`).

## SearchScope

Tipo Rust (`rag::retrieval::SearchScope`) que fuerza explícitamente el alcance de cualquier búsqueda RAG: `Audio(audio_id)` (buscar solo dentro de un audio específico, el default esperado de las tools de RAG/MCP) o `AllCorpus` (buscar en todo el corpus indexado). Deliberadamente no es un `audio_id: Option<String>` — con `None` significando "buscar todo" sería fácil terminar filtrando cross-audio sin querer; con `SearchScope`, el caller tiene que pedir `AllCorpus` a mano para salir del audio seleccionado.

## keep_alive (Ollama)

Parámetro de las llamadas a Ollama (`ollama-rs`) que controla cuánto tiempo un modelo queda cargado en memoria después de responder, antes de descargarse solo (default: 5 minutos de inactividad). Este proyecto lo fuerza explícitamente a `KeepAlive::UnloadOnCompletion` en la última línea de cada job de embeddings (Fase 4) y en cada respuesta de generación RAG (Fase 5) — nunca depender del timeout default para modelos pesados (embeddings de ~1.2GB, generación de 7-8B), sino liberar la memoria apenas se sabe que no hace falta más el modelo (ver `CLAUDE.local.md`: "Ollama — los modelos NO deben permanecer cargados permanentemente").

## Qdrant

Base de datos vectorial usada para indexar los embeddings de las transcripciones y hacer búsqueda semántica sobre ellos.

## Point / Payload (Qdrant)

Un **point** es la unidad básica de Qdrant: un ID + un vector (el embedding) + un **payload** (metadata arbitraria en formato JSON asociada a ese vector — en este proyecto, `audio_id`, `chunk_id`, `start`, `end`, `text`, `speaker`, `avg_logprob`). Buscar en Qdrant es, en el fondo, encontrar los points cuyo vector es más similar al de la query, y devolver su payload para poder usar el texto real. El ID de cada point en este proyecto es determinístico (`Uuid::new_v5` sobre `audio_id:chunk_id`), no aleatorio, para que reintentar la Fase 4 tras un crash haga upsert idempotente en vez de duplicar points.

## HNSW (Hierarchical Navigable Small World)

Algoritmo de índice que usa Qdrant (y la mayoría de bases de datos vectoriales) para buscar los vecinos más cercanos sin comparar la query contra *todos* los vectores uno por uno — arma un grafo en capas que permite saltar rápido hacia las regiones más prometedoras del espacio vectorial. Cada colección en Qdrant paga un costo fijo de memoria/CPU por mantener este índice (además de segmentos y WAL) independientemente de cuántos vectores tenga — por eso este proyecto usa **una sola colección** para todo el corpus en vez de una por audio (ver `CLAUDE.local.md`: "Qdrant — topología"): dividir multiplicaría ese overhead fijo sin reducir el volumen total de datos a servir.

## Checkpoint / checkpointing

Persistir periódicamente el progreso de un proceso largo (acá: qué chunk y qué segundo del audio ya se procesó) para poder retomarlo desde ahí si el proceso crashea, en vez de reprocesar un audio de varias horas desde el principio.

## Backpressure

Mecanismo para que un productor (ej: llegada de audios) no sobrepase la capacidad de un consumidor (ej: la transcripción con Whisper), típicamente vía canales acotados (`mpsc` bounded) o semáforos (`tokio::sync::Semaphore`) en vez de encolar trabajo sin límite.

## `spawn_blocking` vs `tokio::spawn`

En Tokio, `tokio::spawn` corre una tarea *async* en el pool de workers async — bloquear ahí (con trabajo CPU-bound síncrono, como la inferencia de Whisper) traba el runtime y afecta a las demás tareas. `tokio::task::spawn_blocking` corre la tarea en un thread pool separado, dedicado a trabajo bloqueante/CPU-bound. Por eso Whisper siempre debe correr dentro de `spawn_blocking`.
