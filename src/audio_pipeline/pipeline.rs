use crate::audio_pipeline::checkpoint::CheckpointManager;
use crate::audio_pipeline::decoder::StreamingDecoder;
use crate::audio_pipeline::jsonl_writer::JsonlWriter;
use crate::audio_pipeline::models::{JobMetadata, TranscriptEntry};
use crate::audio_pipeline::whisper_runner::WhisperRunner;

pub fn run_pipeline(metadata: JobMetadata) -> anyhow::Result<()> {
    let mut decoder = StreamingDecoder::new(&metadata.audio_path)?;

    let mut writer = JsonlWriter::new(&metadata.transcript_path)?;

    let mut checkpoint = CheckpointManager::new(&metadata.checkpoint_path)?;

    let cp = checkpoint.load()?;

    // `last_chunk` es el último chunk que terminó de procesarse antes del corte (o 0 si no hay
    // checkpoint todavía) — el resume retoma en el siguiente. Sin checkpoint, `processed_seconds`
    // también es 0.0, así que `seek_seconds` hace early-return y `resume_from_chunk` se ignora.
    decoder.seek_seconds(cp.processed_seconds, cp.last_chunk + 1)?;

    let mut runner = WhisperRunner::new("models/ggml-small-q5_1.bin")?;

    while let Some(chunk) = decoder.next_chunk()? {
        let (text, avg_logprob) = runner.transcribe_chunk(&chunk.samples)?;

        writer.append(TranscriptEntry {
            chunk: chunk.index,
            start: chunk.start_sec,
            end: chunk.end_sec,
            text,
            avg_logprob,
        })?;

        checkpoint.save(chunk.index, chunk.end_sec)?;
    }

    Ok(())
}
