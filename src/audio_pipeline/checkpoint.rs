use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use crate::audio_pipeline::models::{Checkpoint, TranscriptEntry};
use crate::audio_pipeline::util::write_atomic;

pub struct CheckpointManager {
    checkpoint_path: PathBuf,
}

impl CheckpointManager {
    pub fn new(checkpoint_path: &str) -> anyhow::Result<Self> {
        Ok(Self {
            checkpoint_path: PathBuf::from(checkpoint_path),
        })
    }

    /// Si no existe checkpoint todavía (primera corrida del job), devuelve el default
    /// (last_chunk: 0, processed_seconds: 0.0) en vez de fallar.
    pub fn load(&mut self) -> anyhow::Result<Checkpoint> {
        if !self.checkpoint_path.exists() {
            return Ok(Checkpoint::default());
        }
        let contents = fs::read_to_string(&self.checkpoint_path)?;
        Ok(serde_json::from_str(&contents)?)
    }

    /// Como `load()`, pero si el archivo existe y no parsea (corrupto — hallazgo real 2026-08-30:
    /// un corte de luz a mitad del `write_atomic` de un chunk dejó `checkpoint.json` en 0 bytes,
    /// ver `docs/TODO.md`), reconstruye el checkpoint leyendo la última línea válida de
    /// `transcript.jsonl` en vez de fallar duro y perder el resume. `transcript.jsonl` es
    /// append-only y nunca se trunca (ver `JsonlWriter`/Fase 3), así que sigue siendo la fuente de
    /// verdad del progreso real incluso si `checkpoint.json` se perdió. Persiste el checkpoint
    /// recuperado antes de devolverlo, para que una corrida siguiente no tenga que recuperarlo de
    /// nuevo. Nunca falla por esto: sin `transcript.jsonl` o sin ninguna línea válida, recupera al
    /// default (0, 0.0) — equivalente a "nunca se procesó nada todavía".
    pub fn load_or_recover(&mut self, transcript_path: &str) -> anyhow::Result<Checkpoint> {
        match self.load() {
            Ok(cp) => Ok(cp),
            Err(e) => {
                let recuperado = ultimo_progreso_valido(transcript_path);
                tracing::warn!(
                    "checkpoint.json corrupto ({e}), reconstruido desde transcript.jsonl: \
                     last_chunk={} processed_seconds={}",
                    recuperado.last_chunk,
                    recuperado.processed_seconds
                );
                self.save(recuperado.last_chunk, recuperado.processed_seconds)?;
                Ok(recuperado)
            }
        }
    }

    pub fn save(&mut self, last_chunk: usize, processed_seconds: f32) -> anyhow::Result<()> {
        let checkpoint = Checkpoint {
            last_chunk,
            processed_seconds,
        };
        let json = serde_json::to_string_pretty(&checkpoint)?;
        write_atomic(&self.checkpoint_path, &json)?;
        Ok(())
    }
}

/// Última línea de `transcript.jsonl` que parsea como `TranscriptEntry` — tolera una última línea
/// torn/parcial (mismo tipo de corte de energía que puede dejar `checkpoint.json` corrupto)
/// descartándola en vez de fallar. Nunca falla: sin archivo o sin ninguna línea válida, devuelve
/// el default (0, 0.0).
fn ultimo_progreso_valido(transcript_path: &str) -> Checkpoint {
    let Ok(file) = fs::File::open(transcript_path) else {
        return Checkpoint::default();
    };

    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<TranscriptEntry>(&line).ok())
        .last()
        .map(|entry| Checkpoint {
            last_chunk: entry.chunk,
            processed_seconds: entry.end,
        })
        .unwrap_or_default()
}
