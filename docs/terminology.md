# Glosario

Definiciones de los términos técnicos usados en el proyecto y en `CLAUDE.local.md`, pensadas para alguien que retoma el código sin el contexto completo.

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

## VAD (Voice Activity Detection)

Detección de qué partes de una señal de audio contienen voz humana vs. silencio/ruido. Permite saltar tramos vacíos antes de decodificarlos con Whisper, ahorrando CPU en audios largos con silencios. `whisper-rs` 0.16.0 expone `set_vad_model_path` / `enable_vad` para esto (no aplicado todavía, ver `docs/TODO.md`).

## RAG (Retrieval-Augmented Generation)

Técnica que combina búsqueda semántica (recuperar los fragmentos de texto más relevantes de una base de datos vectorial) con generación de texto por un LLM, para responder preguntas ancladas en un corpus específico (acá: las transcripciones) en vez de depender solo del conocimiento paramétrico del modelo.

## Embedding

Representación vectorial (de N dimensiones) de un fragmento de texto, generada por un modelo (acá, vía Ollama), tal que fragmentos con significado similar quedan cerca en ese espacio vectorial. Es lo que permite la búsqueda semántica en Qdrant.

## Qdrant

Base de datos vectorial usada para indexar los embeddings de las transcripciones y hacer búsqueda semántica sobre ellos.

## Checkpoint / checkpointing

Persistir periódicamente el progreso de un proceso largo (acá: qué chunk y qué segundo del audio ya se procesó) para poder retomarlo desde ahí si el proceso crashea, en vez de reprocesar un audio de varias horas desde el principio.

## Backpressure

Mecanismo para que un productor (ej: llegada de audios) no sobrepase la capacidad de un consumidor (ej: la transcripción con Whisper), típicamente vía canales acotados (`mpsc` bounded) o semáforos (`tokio::sync::Semaphore`) en vez de encolar trabajo sin límite.

## `spawn_blocking` vs `tokio::spawn`

En Tokio, `tokio::spawn` corre una tarea *async* en el pool de workers async — bloquear ahí (con trabajo CPU-bound síncrono, como la inferencia de Whisper) traba el runtime y afecta a las demás tareas. `tokio::task::spawn_blocking` corre la tarea en un thread pool separado, dedicado a trabajo bloqueante/CPU-bound. Por eso Whisper siempre debe correr dentro de `spawn_blocking`.
