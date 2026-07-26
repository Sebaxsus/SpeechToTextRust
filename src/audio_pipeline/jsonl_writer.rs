use std::fs::File;
use std::io::BufWriter;

use crate::audio_pipeline::models::TranscriptEntry;

pub struct JsonlWriter {
    writer: BufWriter<File>,
}

impl JsonlWriter {
    pub fn new(transcript_path: &str) -> anyhow::Result<Self> {
        let file = File::create(transcript_path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    // TODO: append is not incremental/streaming-safe yet — must write one JSON line + flush
    // per call without buffering the whole transcript in memory (see CLAUDE.local.md: formato
    // de persistencia).
    pub fn append(&mut self, _entry: TranscriptEntry) -> anyhow::Result<()> {
        todo!("JsonlWriter::append is not implemented yet")
    }
}
