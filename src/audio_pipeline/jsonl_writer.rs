use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use crate::audio_pipeline::models::TranscriptEntry;

pub struct JsonlWriter {
    writer: BufWriter<std::fs::File>,
}

impl JsonlWriter {
    /// Abre el archivo en modo *append* (creándolo si no existe), nunca lo trunca. Es crítico
    /// para el resume (`handlers::audio_handler::reanudar_job`): `run_pipeline` llama a
    /// `JsonlWriter::new` sin importar si el job es nuevo o se está retomando desde un
    /// `checkpoint.json` existente — si esto truncara el archivo, un resume borraría en
    /// silencio las líneas de los chunks ya transcritos antes del corte (bug real detectado y
    /// corregido durante la verificación manual del endpoint de resume, 2026-07-28).
    pub fn new(transcript_path: &str) -> anyhow::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(transcript_path)?;
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
