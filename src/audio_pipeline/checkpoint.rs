use std::fs;
use std::path::PathBuf;

use crate::audio_pipeline::models::Checkpoint;

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

    pub fn save(&mut self, last_chunk: usize, processed_seconds: f32) -> anyhow::Result<()> {
        let checkpoint = Checkpoint {
            last_chunk,
            processed_seconds,
        };
        let json = serde_json::to_string_pretty(&checkpoint)?;
        fs::write(&self.checkpoint_path, json)?;
        Ok(())
    }
}
