use std::fs::File;
use std::io::{BufWriter, Write};

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

    /// Escribe una línea JSON por entrada y flushea inmediatamente — nunca acumula el
    /// transcript completo en memoria (ver CLAUDE.local.md: formato de persistencia).
    pub fn append(&mut self, entry: TranscriptEntry) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.writer, &entry)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}
