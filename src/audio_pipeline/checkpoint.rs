use crate::audio_pipeline::models::Checkpoint;

// TODO: implement crash-recoverable checkpoint persistence (see CLAUDE.local.md: checkpointing,
// persistencia incremental).
pub struct CheckpointManager;

impl CheckpointManager {
    pub fn new(_checkpoint_path: &str) -> anyhow::Result<Self> {
        todo!("CheckpointManager::new is not implemented yet")
    }

    pub fn load(&mut self) -> anyhow::Result<Checkpoint> {
        todo!("CheckpointManager::load is not implemented yet")
    }

    pub fn save(&mut self, _last_chunk: usize, _processed_seconds: f32) -> anyhow::Result<()> {
        todo!("CheckpointManager::save is not implemented yet")
    }
}
