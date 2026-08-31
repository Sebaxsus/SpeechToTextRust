use std::time::Instant;

use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::decoder::StreamingDecoder;
use crate::audio_pipeline::jsonl_writer::JsonlWriter;
use crate::audio_pipeline::models::{ChunkMetrics, JobMetadata, TranscriptEntry};
use crate::audio_pipeline::whisper_runner::WhisperRunner;

/// Cantidad de chunks consecutivos con texto idéntico que confirman un loop de alucinación (no
/// habla real distinta) — ver `WHISPER_TUNING_LOG.md`, hallazgo 2026-08-06: en los dos jobs
/// reales que originaron este fix, los loops encontrados iban de 3 a 325 chunks consecutivos,
/// nunca menos de 3, así que este umbral detecta todos los casos reales conocidos sin falsos
/// positivos (3 chunks de 30s con texto *idéntico* ya es prácticamente imposible en habla
/// distinta).
const REPETICIONES_PARA_LOOP: u32 = 3;

/// Colapsa espacios (incluye el `\n` que separan segmentos dentro de un mismo chunk) para que la
/// comparación de repetición no falle por diferencias triviales de espaciado — mismo criterio
/// que ya se usó para detectar los loops reales en los transcripts de `51b27211-...`/`e2ce31cc-...`.
fn normalizar_texto(texto: &str) -> String {
    texto.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Actualiza el contador de repeticiones consecutivas de texto normalizado. Un texto vacío
/// (silencio real, `no_speech`) nunca cuenta como repetición — ver CLAUDE.local.md: "Condiciones
/// reales de grabación". Devuelve el nuevo contador: `1` en la primera aparición de un texto no
/// vacío, incrementado mientras se repita, `0` en cuanto cambia o el chunk queda vacío.
fn actualizar_contador_repeticion(anterior: &str, actual: &str, contador_anterior: u32) -> u32 {
    if actual.is_empty() {
        0
    } else if actual == anterior {
        contador_anterior + 1
    } else {
        1
    }
}

pub fn run_pipeline(metadata: JobMetadata) -> anyhow::Result<()> {
    let mut decoder = StreamingDecoder::new(&metadata.audio_path)?;

    let mut writer = JsonlWriter::<TranscriptEntry>::new(&metadata.transcript_path)?;

    // Diagnóstico del pipeline (tiempos por etapa + señales de calidad de whisper-rs), separado
    // de `transcript.jsonl`/el payload de Qdrant a propósito (ver CLAUDE.local.md: logger de
    // métricas). Nunca bloquea el pipeline si algo raro pasa con este archivo — es información
    // best-effort, no un requisito para que la transcripción avance.
    let metrics_path = metadata.metrics_path();
    let mut metrics_writer = JsonlWriter::<ChunkMetrics>::new(&metrics_path.to_string_lossy())?;

    let mut checkpoint = CheckpointManager::new(&metadata.checkpoint_path)?;

    let cp = checkpoint.load()?;

    // `last_chunk` es el último chunk que terminó de procesarse antes del corte (o 0 si no hay
    // checkpoint todavía) — el resume retoma en el siguiente. Sin checkpoint, `processed_seconds`
    // también es 0.0, así que `seek_seconds` hace early-return y `resume_from_chunk` se ignora.
    decoder.seek_seconds(cp.processed_seconds, cp.last_chunk + 1)?;

    let cfg = crate::config::get();
    let mut runner = WhisperRunner::new(
        &cfg.paths.whisper_model_path.to_string_lossy(),
        cfg.whisper.clone(),
    )?;

    // Estado del loop-breaker (ver `REPETICIONES_PARA_LOOP`) — vive en el scope de `run_pipeline`,
    // no persiste entre resumes (un resume construye un `WhisperRunner`/loop nuevo desde cero, así
    // que en el peor caso tarda hasta `REPETICIONES_PARA_LOOP` chunks en re-detectar un loop que
    // ya estaba activo al momento del corte — no vale la pena persistirlo en `checkpoint.json`
    // solo para ese caso borde).
    let mut ultimo_texto_normalizado = String::new();
    let mut repeticiones_consecutivas: u32 = 0;
    let mut forzar_contexto_fresco = false;

    loop {
        let t_decode = Instant::now();
        let Some(chunk) = decoder.next_chunk()? else {
            break;
        };
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        let t_whisper = Instant::now();
        let transcription = runner.transcribe_chunk(&chunk.samples, forzar_contexto_fresco)?;
        let whisper_ms = t_whisper.elapsed().as_millis() as u64;

        let text_len = transcription.text.chars().count();

        let texto_normalizado = normalizar_texto(&transcription.text);
        repeticiones_consecutivas = actualizar_contador_repeticion(
            &ultimo_texto_normalizado,
            &texto_normalizado,
            repeticiones_consecutivas,
        );
        ultimo_texto_normalizado = texto_normalizado;

        // Ver `WHISPER_TUNING_LOG.md` (hallazgo 2026-08-06): una vez confirmado el loop, se
        // fuerza `no_context` en el chunk siguiente (rompe la realimentación de contexto que lo
        // sostiene) y se blanquea el texto de este chunk en adelante mientras persista — mismo
        // tratamiento que un chunk de silencio real (`run_embedding_phase` ya saltea texto
        // vacío), en vez de repetir la alucinación cientos de veces en `transcript.jsonl`.
        let loop_detectado = repeticiones_consecutivas >= REPETICIONES_PARA_LOOP;
        forzar_contexto_fresco = loop_detectado;

        let texto_final = if loop_detectado {
            tracing::warn!(
                chunk = chunk.index,
                repeticiones = repeticiones_consecutivas,
                texto_original = %transcription.text,
                "posible loop de alucinación detectado (texto repetido), se suprime en transcript.jsonl"
            );
            String::new()
        } else {
            transcription.text
        };

        let t_persist = Instant::now();
        writer.append(TranscriptEntry {
            chunk: chunk.index,
            start: chunk.start_sec,
            end: chunk.end_sec,
            text: texto_final,
            avg_logprob: transcription.avg_logprob,
        })?;
        checkpoint.save(chunk.index, chunk.end_sec)?;
        let persist_ms = t_persist.elapsed().as_millis() as u64;

        let metrics = ChunkMetrics {
            chunk: chunk.index,
            start: chunk.start_sec,
            end: chunk.end_sec,
            decode_ms,
            whisper_ms,
            persist_ms,
            total_ms: decode_ms + whisper_ms + persist_ms,
            text_len,
            avg_logprob: transcription.avg_logprob,
            score: transcription.avg_logprob.exp(),
            no_speech_prob: transcription.no_speech_prob,
            entropy: transcription.entropy,
            segment_count: transcription.segment_count,
            loop_suppressed: loop_detectado,
        };

        // `tracing`'s `Value` solo cubre i64/u64/f64/bool/&str como primitivos (ver
        // tracing::field, doc del módulo) — los campos f32 se castean a f64 para el log
        // estructurado; `ChunkMetrics`/`metrics.jsonl` se quedan en f32 sin cambios.
        tracing::debug!(
            chunk = metrics.chunk,
            decode_ms = metrics.decode_ms,
            whisper_ms = metrics.whisper_ms,
            persist_ms = metrics.persist_ms,
            total_ms = metrics.total_ms,
            text_len = metrics.text_len,
            avg_logprob = metrics.avg_logprob as f64,
            score = metrics.score as f64,
            no_speech_prob = metrics.no_speech_prob as f64,
            entropy = metrics.entropy as f64,
            segment_count = metrics.segment_count,
            "chunk procesado"
        );

        metrics_writer.append(metrics)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizar_texto_colapsa_espacios_y_saltos_de_linea() {
        assert_eq!(
            normalizar_texto("  Este es el canal\nde los estudiantes.  "),
            "Este es el canal de los estudiantes."
        );
    }

    /// Reproduce la secuencia real encontrada en `e2ce31cc-...` (chunks 233-235): el mismo texto
    /// tres veces seguidas debe llegar a `REPETICIONES_PARA_LOOP` exactamente en la tercera.
    #[test]
    fn tres_chunks_identicos_alcanzan_el_umbral_de_loop() {
        let texto = "Este es el canal de la reunión de los estudiantes.";
        let mut anterior = String::new();
        let mut contador = 0;

        contador = actualizar_contador_repeticion(&anterior, texto, contador);
        assert_eq!(contador, 1);
        anterior = texto.to_string();

        contador = actualizar_contador_repeticion(&anterior, texto, contador);
        assert_eq!(contador, 2);

        contador = actualizar_contador_repeticion(&anterior, texto, contador);
        assert_eq!(contador, 3);
        assert!(contador >= REPETICIONES_PARA_LOOP);
    }

    #[test]
    fn dos_chunks_identicos_no_alcanzan_el_umbral() {
        let texto = "por favor.";
        let mut contador = actualizar_contador_repeticion("", texto, 0);
        contador = actualizar_contador_repeticion(texto, texto, contador);
        assert_eq!(contador, 2);
        assert!(contador < REPETICIONES_PARA_LOOP);
    }

    #[test]
    fn texto_distinto_resetea_el_contador() {
        let contador = actualizar_contador_repeticion("¡Suscríbete!", "Gracias por venir.", 5);
        assert_eq!(contador, 1);
    }

    /// Un chunk vacío (silencio real) nunca cuenta como repetición, ni siquiera de sí mismo —
    /// evita que una racha de silencio genuino dispare el loop-breaker.
    #[test]
    fn texto_vacio_nunca_cuenta_como_repeticion() {
        assert_eq!(actualizar_contador_repeticion("", "", 0), 0);
        assert_eq!(actualizar_contador_repeticion("algo", "", 5), 0);
    }
}
