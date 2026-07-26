use std::path::PathBuf;

use crate::audio_pipeline::models::JobMetadata;

// TODO: implement real job creation (job id, uploads/transcripts/checkpoint paths, metadata persistence).
pub fn create_job(_audio_path: PathBuf) -> anyhow::Result<JobMetadata> {
    todo!("create_job is not implemented yet")
}
