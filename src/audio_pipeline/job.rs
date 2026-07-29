use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use uuid::Uuid;

use crate::audio_pipeline::models::{JobMetadata, JobStatus};
use crate::audio_pipeline::util::write_atomic;

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
    write_atomic(&job_json_path, &job_json)?;

    Ok(metadata)
}

/// Loads the metadata of an existing job — used to resume an interrupted run (see
/// `handlers::audio_handler::reanudar_job`) and by the read-only MCP tools (`list_audios`,
/// `get_audio_metadata`) that read `job.json`.
///
/// `job_id` may come straight from an HTTP path param or an MCP tool argument (untrusted): it is
/// validated as a well-formed UUID *before* being used to build a disk path, so a value like
/// `"../../whatever"` is rejected instead of escaping `./jobs/`. This is the same path-traversal
/// concern Fase 1 already avoids for the client-supplied filename (see CLAUDE.local.md) — here
/// the job_id genuinely has to become part of the path, so it gets validated instead of avoided.
pub fn load_job(job_id: &str) -> anyhow::Result<JobMetadata> {
    Uuid::parse_str(job_id).context("job_id no es un UUID válido")?;

    let job_json_path = std::path::Path::new("./jobs").join(job_id).join("job.json");
    let contents = fs::read_to_string(&job_json_path)
        .with_context(|| format!("no se encontró el job '{job_id}'"))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("job.json inválido para el job '{job_id}'"))
}
