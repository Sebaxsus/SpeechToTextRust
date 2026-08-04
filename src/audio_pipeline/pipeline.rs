use std::time::Instant;

use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::decoder::StreamingDecoder;
use crate::audio_pipeline::jsonl_writer::JsonlWriter;
use crate::audio_pipeline::models::{ChunkMetrics, JobMetadata, TranscriptEntry};
use crate::audio_pipeline::whisper_runner::WhisperRunner;

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

    let mut runner = WhisperRunner::new("models/ggml-small-q5_1.bin")?;

    loop {
        let t_decode = Instant::now();
        let Some(chunk) = decoder.next_chunk()? else {
            break;
        };
        let decode_ms = t_decode.elapsed().as_millis() as u64;

        let t_whisper = Instant::now();
        let transcription = runner.transcribe_chunk(&chunk.samples)?;
        let whisper_ms = t_whisper.elapsed().as_millis() as u64;

        let text_len = transcription.text.chars().count();

        let t_persist = Instant::now();
        writer.append(TranscriptEntry {
            chunk: chunk.index,
            start: chunk.start_sec,
            end: chunk.end_sec,
            text: transcription.text,
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
