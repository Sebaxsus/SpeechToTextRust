use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;

use serde::Serialize;

/// Escritor JSONL genérico (una línea por `T::Serialize`, append-only, flush inmediato). Usado
/// tanto para `TranscriptEntry` (`transcript.jsonl`) como para `ChunkMetrics` (`metrics.jsonl`) —
/// mismo comportamiento, sin duplicar la lógica de apertura/escritura entre los dos archivos.
pub struct JsonlWriter<T> {
    writer: BufWriter<std::fs::File>,
    _marker: PhantomData<T>,
}

impl<T: Serialize> JsonlWriter<T> {
    /// Abre el archivo en modo *append* (creándolo si no existe), nunca lo trunca. Es crítico
    /// para el resume (`handlers::audio_handler::reanudar_job`): `run_pipeline` llama a
    /// `JsonlWriter::new` sin importar si el job es nuevo o se está retomando desde un
    /// `checkpoint.json` existente — si esto truncara el archivo, un resume borraría en
    /// silencio las líneas ya escritas antes del corte (bug real detectado y corregido en Fase 3,
    /// ver CLAUDE.local.md). Mismo criterio aplica a `metrics.jsonl`, aunque ese archivo es solo
    /// diagnóstico (no bloquea el resume si se pierde).
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            _marker: PhantomData,
        })
    }

    /// Escribe una línea JSON por entrada y flushea inmediatamente — nunca acumula el archivo
    /// completo en memoria (ver CLAUDE.local.md: formato de persistencia).
    pub fn append(&mut self, entry: T) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.writer, &entry)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}
