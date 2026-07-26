use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

use crate::audio_pipeline::models::{JobMetadata, JobStatus};

/// Creates a fresh job directory (`./jobs/{job_id}/`) and the associated metadata.
///
/// `extension` must already be a trusted, validated value (e.g. "mp3" or "mp4" derived from
/// sniffing magic bytes, never taken verbatim from user input) since it is used as part of a
/// disk path.
pub fn create_job(extension: &str) -> anyhow::Result<JobMetadata> {
    let job_id = Uuid::new_v4().to_string();
    let job_dir = std::path::Path::new("./jobs").join(&job_id);
    fs::create_dir_all(&job_dir)?;

    let audio_path = job_dir.join(format!("audio.{extension}"));
    let transcript_path = job_dir.join("transcript.jsonl");
    let checkpoint_path = job_dir.join("checkpoint.json");

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();

    let metadata = JobMetadata {
        job_id,
        audio_path: audio_path.to_string_lossy().into_owned(),
        transcript_path: transcript_path.to_string_lossy().into_owned(),
        checkpoint_path: checkpoint_path.to_string_lossy().into_owned(),
        status: JobStatus::Pending,
        created_at,
    };

    let job_json_path = job_dir.join("job.json");
    let job_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(job_json_path, job_json)?;

    Ok(metadata)
}
