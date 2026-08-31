# Benchmark: Greedy vs BeamSearch

Registro de las pruebas comparativas entre `SamplingStrategy::Greedy { best_of: 5 }` y
`SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 }`. Para el razonamiento completo de
por qué se evalúa esto y el detalle técnico de cada cambio, ver `WHISPER_TUNING_LOG.md` — este
documento es la vista consolidada orientada a los resultados de hardware/precisión, no un
changelog de parámetros.

**Nota (2026-08-07):** a partir de los resultados de la sección 4 (costo térmico/CPU sostenido
negligible) y de la sección 3 (señales de mejor precisión), el default se invirtió:
**`BeamSearch` es ahora el modo por defecto**, `cargo run -- --greedy` queda como flag opcional
para el modo más rápido. Las tablas de abajo se dejan tal cual se midieron (con los nombres de
flag/rol vigentes en el momento de cada prueba) — leer "default" en las secciones 1-2 como
"lo que corría sin flags en ese momento" (Greedy), no como el default actual del proyecto.

Hardware de referencia en todas las pruebas: AMD Ryzen 5 5600X (6 núcleos/12 hilos), sin GPU
dedicada obligatoria, métricas capturadas con HWiNFO.

## 1. Línea base — job real de ~3.6h en Greedy (`Prueba2.CSV`)

Job `51b27211-7649-42ec-b003-142aa3822d1f` (reunión real, 3.6h), `Greedy { best_of: 1 }` (config
vigente en ese momento, antes de subir el default a 5). 4015 muestras de HWiNFO, promedio sobre
toda la duración del job (incluye tramos de silencio/pausas, no solo carga activa de Whisper).

| Métrica | Promedio | Máximo |
|---|---|---|
| Uso de CPU | 35.8% | 98.2% |
| Temperatura (Tctl/Tdie) | 55.3°C | 71°C |
| Núcleos activos | 3.18 de 6 | 6 |
| Consumo de CPU | 41.2 W | 73.9 W |

Sin throttling térmico registrado (`HTC`/`PROCHOT` en 0 de 4015 muestras), pero con saltos
abruptos de uso total de CPU (>25 puntos porcentuales en menos de 2s, 487 ocurrencias) y uso
desparejo entre núcleos.

## 2. Prueba corta A/B — mismo audio de 2 minutos, Greedy vs BeamSearch

`sample_Media/Muestra2_02min.m4a` (2 min), misma máquina, mismo entorno de prueba que
`Prueba2.CSV`, HWiNFO grabando ambas corridas por separado:

- Greedy (`best_of: 5`, default): job `15e77ca8-4fca-4be7-a890-9b1199e14e80` — `TestGreedy.csv`.
- BeamSearch (`--beam-search`): job `6d704933-e589-4f99-972f-6973e08c5437` — `TestBeamSearch.csv`.

**Reloj de pared (fase Whisper):** Greedy 43s vs BeamSearch 48s (**+12%**) — muy por debajo del
"~3-5x" documentado como advertencia teórica antes de esta prueba.

**CPU/térmica (ventana activa de cada job):**

| Métrica | Prueba2.CSV (job real 3.6h, avg con silencios) | TestGreedy (ventana activa) | TestBeamSearch (ventana activa) |
|---|---|---|---|
| Uso CPU avg/max | 35.8% / 98.2% | 60% / 92.2% | 66.6% / 97% |
| Tctl/Tdie avg/max | 55.3°C / 71°C | 64.1°C / 71.3°C | 66.3°C / 71.3°C |
| Núcleos activos avg/max | 3.18 / 6 | 4.44 / 6 | 4.85 / 6 |
| Consumo CPU avg/max | 41.2W / 73.9W | 53.1W / 73.0W | 58W / 73.2W |

BeamSearch corrió ~11% más de uso promedio, ~2-2.5°C más caliente en promedio y ~9% más consumo
que Greedy en la misma máquina/audio, pero el **pico** (temperatura/consumo máximo) fue
prácticamente idéntico entre los tres casos — y ese pico ya lo alcanzaba un job Greedy real de
horas (`Prueba2.CSV`).

**Hipótesis:** en un Ryzen 5 5600X (6C/12T) que normalmente usa menos de la mitad de su
capacidad (~3.18-4.44 de 6 núcleos), whisper.cpp parece repartir los 5 decodificadores del beam
en núcleos que ya estaban ociosos, así que el costo se nota como "más ocupado más seguido" en vez
de "Nx más lento/caliente".

**Caveat — no concluyente para audios de 5h+:** la muestra es de solo ~48s de carga activa sobre
un audio de 2 min. El riesgo que motivó mantener BeamSearch opt-in es el *sostenido* durante
horas (acumulación térmica/soak), no un pico corto. Ver sección 4 para la prueba de larga
duración que resuelve esta pregunta.

## 3. Comparación de precisión — mismo audio de 2 minutos, Greedy vs BeamSearch

**Advertencia sobre esta sección:** no existe una transcripción de referencia ("ground truth")
para este audio, así que lo que sigue es una comparación **objetiva de métricas** (segmentación,
longitud de texto, confianza) más una lectura cualitativa del texto — no un porcentaje de
accuracy verificado. El audio es una lista de asistencia (nombres propios + códigos numéricos),
un caso particularmente difícil para Whisper: son palabras fuera de vocabulario que el modelo
tiene que adivinar fonéticamente en ambos modos por igual.

| Chunk | Segmentos (Greedy → Beam) | Caracteres (Greedy → Beam) | `avg_logprob` (Greedy → Beam) |
|---|---|---|---|
| 0 (0-30s) | 1 → 7 | 204 → 237 | -0.071 → -0.728 |
| 1 (28-58s) | 1 → 8 | 178 → 236 | -0.404 → -0.543 |
| 2 (56-86s) | 1 → 5 | 197 → 187 | -0.354 → -0.391 |
| 3 (84-114s) | 1 → 7 | 126 → 221 | -0.637 → -0.403 |
| 4 (112-122.7s) | 3 → 1 | 89 → 98 | -0.762 → -0.071 |
| **Total** | **7 → 28** | **794 → 979** | — |

Observaciones:

- **Segmentación:** BeamSearch produce 4x más segmentos internos (28 vs 7) para el mismo audio.
  Dado que el contenido son nombres/códigos separados por pausas discretas (turno de asistencia),
  más segmentos tiende a alinear mejor con la estructura real del habla — Greedy tiende a fundir
  varios nombres en un único segmento corrido.
- **Longitud de texto:** BeamSearch capturó ~23% más texto total (979 vs 794 caracteres) en el
  mismo tramo de audio, notablemente en el chunk 3 (221 vs 126 caracteres) — sugiere que Greedy
  está comprimiendo o descartando parte del habla real en ese tramo, no solo transcribiéndola más
  corto.
- **Confianza (`avg_logprob`) no es un proxy confiable acá:** en 3 de 5 chunks Greedy fue más
  "confiado" que BeamSearch y en los otros 2 al revés, sin relación aparente con qué texto se lee
  mejor — con nombres propios fuera de vocabulario, el modelo puede estar muy seguro de una
  alucinación fonéticamente plausible en cualquiera de los dos modos.
- **Lectura cualitativa (ejemplo, chunk 0):**
  - Greedy: *"La Navalentina Carreño 1, la Comunidad de Cis, Chabarrobeña 2-5, Santiago Cortés,
    Parra 5..."* — el fragmento "la Comunidad de Cis" no encaja gramaticalmente con el resto (lista
    de nombres), un indicio de relleno/alucinación.
  - BeamSearch: *"Dana Valentina Carreño, uno. Maria Alexis Chabarrobeña, dos, cinco. Santiago
    Cortés, parra, cinco, muy bien..."* — se lee como una lista de nombre + código, consistente
    con el resto del audio, sin el fragmento incoherente que aparece en Greedy.
- **Conclusión parcial:** con esta única muestra de 2 minutos, BeamSearch da señales de mejor
  segmentación y mayor cobertura de texto, y al menos un caso concreto donde el texto de Greedy
  incluye una frase incoherente que BeamSearch no repite. No alcanza para una conclusión
  estadística — es consistente con la razón por la que se investiga BeamSearch, pero se necesita
  una muestra más grande (ver sección 4) y, si es posible, contrastar contra la lista real de
  asistencia para tener una medición de precisión real.

## 4. Prueba de larga duración (en curso / pendiente)

**Objetivo:** confirmar si el costo de CPU/térmico de BeamSearch medido en la sección 2 (prueba
de 48s) se sostiene sin acumulación térmica en un audio real de duración comparable a los jobs de
producción (horas, no minutos).

**Audio de prueba:** `sample_Media/comité de obra 11 de junio 2026.wav` — 49 min 9s
(2949.14s), reunión real.

| Corrida | Job ID | Modo | Inicio | Fin fase Whisper | Fin total (+ embeddings) | Estado |
|---|---|---|---|---|---|---|
| 1 | `f109206a-632f-4279-911f-b36396362d22` | BeamSearch (`--beam-search`) | 07.08.2026 16:34:03 | 07.08.2026 16:58:34 | 07.08.2026 16:59:18 | Completado |
| 2 | `f2104127-6115-40de-9459-7f8421333c3d` | Greedy (default) | 07.08.2026 17:18:50 | 07.08.2026 17:41:31 | 07.08.2026 17:42:16 | Completado |

**Corrida 1 (BeamSearch) — resultado:**

- Fase Whisper: **24 min 31s** (1471s) para 49 min 9s de audio → factor tiempo-real **~0.50x**
  (procesa a ~2x la velocidad de reproducción del audio).
- Fase embeddings (Ollama + Qdrant): 44s adicionales.
- 106 chunks en total. El loop-breaker se activó **una sola vez** (chunk 42, minuto ~19.6):
  detectó una repetición sostenida, la cortó (texto suprimido, `avg_logprob` residual -0.456
  sobre 1 solo carácter) y el pipeline continuó con contexto fresco sin quedar atrapado — la
  misma clase de bug documentada en `WHISPER_TUNING_LOG.md` (2026-08-06), ahora resuelta en
  producción sobre un audio real nuevo, no solo en los tests unitarios.
- Sin chunks con `avg_logprob` por debajo de -0.79 (el peor caso fue -0.78) — no hay evidencia de
  los tramos muy garbled (`avg_logprob` -4 a -9.4) que sí aparecían en los jobs de `51b27211-...`/
  `e2ce31cc-...` antes de estos cambios, aunque no es una comparación directa (audio distinto).

**Corrida 2 (Greedy) — resultado:**

- Fase Whisper: **22 min 41s** (1361s) para 49 min 9s de audio → factor tiempo-real **~0.46x**.
- Fase embeddings: 45s adicionales (prácticamente igual que BeamSearch).
- 106 chunks en total (mismo audio, mismo chunking). **0 activaciones del loop-breaker** — a
  diferencia de BeamSearch (1 activación en el chunk 42), Greedy no entró en ningún loop de
  alucinación sostenido en este audio.
- Peor `avg_logprob`: -0.766 (similar al peor caso de BeamSearch, -0.782) — ninguno de los dos
  modos produjo chunks muy garbled en este audio real.

**Comparación directa BeamSearch vs Greedy — mismo audio de 49 min, mismo servidor/máquina:**

| Métrica | Greedy (default) | BeamSearch | Diferencia |
|---|---|---|---|
| Fase Whisper (reloj de pared) | 22min 41s (1361s) | 24min 31s (1471s) | **+8.1%** |
| Factor tiempo-real | 0.46x | 0.50x | — |
| Fase embeddings | 45s | 44s | ~igual |
| Activaciones del loop-breaker | 0 | 1 | — |
| Peor `avg_logprob` del job | -0.766 | -0.782 | ~igual |

**Lectura:** el +8.1% de costo en reloj de pared para un audio real de 49 minutos confirma —
ahora con una muestra representativa, no solo los 48s de la sección 2 — que el "~3-5x" de la
advertencia teórica inicial (basada en el costo de cómputo puro de `beam_size=5` decodificadores)
no se materializa en este hardware: el margen ocioso del Ryzen 5 5600X (normalmente
~3.2-4.4 de 6 núcleos activos, ver sección 1) absorbe casi todo el costo adicional de BeamSearch.
Falta correlacionar esto con las curvas de HWiNFO de ambas corridas (grabadas en paralelo por el
usuario) para confirmar que la temperatura/consumo tampoco se acumula de forma distinta a lo ya
visto en la sección 2 — pendiente de análisis una vez se compartan los `.csv` de esta prueba.

Sobre el único dato de precisión disponible en esta corrida (activación única del loop-breaker en
BeamSearch, cero en Greedy): con una sola ocurrencia no alcanza para concluir que BeamSearch sea
más o menos propenso a loops de alucinación en general — es la misma clase de evento que ya se
vio en ambos modos durante la prueba corta de la sección 3, solo que esta vez el loop-breaker lo
contuvo automáticamente antes de que se sostuviera.

### CPU/térmica sostenida — HWiNFO de ambas corridas (`TestBeamSearch49min.CSV` / `TestGreedy49min.csv`)

Ventana analizada: desde `processing_started_at` hasta `transcript_ready_at` de cada job (la
fase Whisper completa, ~23-25 minutos de carga real, no los ~48s de la sección 2).

| Métrica | Greedy (681 muestras) | BeamSearch (736 muestras) | Diferencia |
|---|---|---|---|
| Uso CPU avg/max | 56.6% / 77.6% | 56.3% / 79.0% | ~igual (avg), +1.4pp (max) |
| Tctl/Tdie avg/max | 64.8°C / 69.6°C | 65.1°C / 70.6°C | +0.3°C (avg), +1.0°C (max) |
| Caja de CPU avg/max | 57.9°C / 67.8°C | 58.6°C / 69.4°C | +0.7°C (avg), +1.6°C (max) |
| Consumo CPU avg/max | 54.4W / 72.5W | 54.4W / 73.1W | igual (avg), +0.6W (max) |
| Núcleos activos avg/max | 4.79 / 5.90 | 4.75 / 6.00 | ~igual |

**Conclusión — ahora sí concluyente:** sobre ~23-25 minutos de carga sostenida real (no un pico
de 48s), la diferencia de CPU/temperatura/consumo entre Greedy y BeamSearch es **prácticamente
ruido de medición**, muy por debajo de las diferencias del +8-11% observadas en la prueba corta
de la sección 2. Ninguna métrica muestra la acumulación térmica que motivó mantener BeamSearch
como opt-in en primer lugar: el pico de temperatura (70.6°C) queda incluso por debajo del pico ya
registrado en un job Greedy real de 3.6h (`Prueba2.CSV`, 71°C), y el consumo máximo (73.1W) es
prácticamente igual al de esa misma referencia (73.9W).

Esto **revierte la cautela** de la sección 2: con una muestra representativa (23+ minutos en vez
de 48s), el "costo" de BeamSearch que parecía notorio en la prueba corta se diluye dentro de la
variabilidad normal entre corridas — probablemente el margen de núcleos ociosos del Ryzen 5 5600X
(sección 1: ~3.2-4.4 de 6 núcleos activos en promedio en producción real) es suficiente para
absorber los decodificadores extra de BeamSearch sin que el die tenga que trabajar
sostenidamente más caliente. **Recomendación actualizada:** con esta evidencia, BeamSearch ya no
necesita quedar restringido a un flag opt-in por motivos térmicos/CPU — el costo real medido en
audios de duración representativa (49 min) es marginal. Se mantiene como flag explícito (no como
nuevo default) por prudencia y porque la ganancia de precisión medida hasta ahora (sección 3) es
cualitativa sobre una sola muestra corta, no una razón contundente para cambiar el default —pero
el argumento térmico/CPU que originalmente lo descartaba como default queda desactivado por esta
prueba.
