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

    decoder.seek_seconds(cp.processed_seconds)?;

    let runner = WhisperRunner::new("models/ggml-base-tdrz-q5_1.bin")?;

    while let Some(chunk) = decoder.next_chunk()? {
        let text = runner.transcribe_chunk(&chunk.samples)?;

        writer.append(TranscriptEntry {
            chunk: chunk.index,
            start: chunk.start_sec,
            end: chunk.end_sec,
            text,
        })?;

        checkpoint.save(chunk.index, chunk.end_sec)?;
    }

    Ok(())
}
