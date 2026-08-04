use std::fs;
use std::sync::Mutex;

use anyhow::Context;
use uuid::Uuid;

use crate::audio_pipeline::models::{JobMetadata, JobStatus, SummaryStatus};
use crate::audio_pipeline::util::{now_epoch_string, write_atomic};

/// Serializa el read-modify-write de `update_job_metadata` frente a cualquier otra llamada
/// concurrente (mismo `job_id` o distinto). `write_atomic` (tmp + rename) ya evita que un lector
/// vea un `job.json` truncado/corrupto, pero por sí sola no evita un "lost update" si dos
/// llamadas a `update_job_metadata` para el mismo job corrieran de verdad en paralelo (la
/// segunda leería el estado previo a la escritura de la primera). Hoy esto no puede pasar porque
/// las 4 transiciones de `lanzar_procesamiento_job` ocurren todas mientras esa tarea sostiene el
/// único permiso de `heavy_compute_semaphore` — pero depender solo de esa coincidencia es frágil
/// frente a un futuro call site que escriba `job.json` sin pasar por el semáforo. Lock global (no
/// uno por `job_id`): a esta escala (4 escrituras por job, JSON de pocos cientos de bytes) un
/// `HashMap<String, Mutex<()>>` por job sería complejidad que no se justifica. `std::sync::Mutex`
/// (no `tokio::sync::Mutex`) porque la sección crítica es síncrona y brevísima, igual que el
/// resto de este módulo (I/O sync directo en contexto async, sin `spawn_blocking`).
static JOB_METADATA_LOCK: Mutex<()> = Mutex::new(());

/// Creates a fresh job directory (`./jobs/{job_id}/`) and the associated metadata.
///
/// `extension` must already be a trusted, validated value (e.g. "mp3" or "mp4" derived from
/// sniffing magic bytes, never taken verbatim from user input) since it is used as part of a
/// disk path.
pub fn create_job(extension: &str) -> anyhow::Result<JobMetadata> {
    let job_id = Uuid::new_v4().to_string();
    let job_dir = crate::config::get().paths.jobs_dir.join(&job_id);
    fs::create_dir_all(&job_dir)?;

    let audio_path = job_dir.join(format!("audio.{extension}"));
    let transcript_path = job_dir.join("transcript.jsonl");
    let checkpoint_path = job_dir.join("checkpoint.json");

    let metadata = JobMetadata {
        job_id,
        audio_path: audio_path.to_string_lossy().into_owned(),
        transcript_path: transcript_path.to_string_lossy().into_owned(),
        checkpoint_path: checkpoint_path.to_string_lossy().into_owned(),
        status: JobStatus::Pending,
        created_at: now_epoch_string(),
        transcript_ready: false,
        processing_started_at: None,
        transcript_ready_at: None,
        completed_at: None,
        summary_status: SummaryStatus::default(),
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

    let job_json_path = crate::config::get()
        .paths
        .jobs_dir
        .join(job_id)
        .join("job.json");
    let contents = fs::read_to_string(&job_json_path)
        .with_context(|| format!("no se encontró el job '{job_id}'"))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("job.json inválido para el job '{job_id}'"))
}

/// Helper centralizado para mutar `job.json`: carga el estado actual (vía `load_job`, que ya
/// valida el UUID y la existencia del job), aplica `mutate` en memoria, y reescribe el archivo
/// completo de forma atómica. Único punto de escritura para transiciones de estado (`status`,
/// `transcript_ready`, etc.) — evita que cada call site reconstruya `JobMetadata` a mano y pise
/// campos como `audio_path`/`created_at` con datos obsoletos.
///
/// Serializado por `JOB_METADATA_LOCK` (ver arriba) para que el read-modify-write completo sea
/// atómico frente a otras llamadas concurrentes, no solo la escritura final en disco.
pub fn update_job_metadata<F>(job_id: &str, mutate: F) -> anyhow::Result<()>
where
    F: FnOnce(&mut JobMetadata),
{
    let _guard = JOB_METADATA_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let mut metadata = load_job(job_id)?;
    mutate(&mut metadata);

    let job_json_path = crate::config::get()
        .paths
        .jobs_dir
        .join(job_id)
        .join("job.json");
    let job_json = serde_json::to_string_pretty(&metadata)?;
    write_atomic(&job_json_path, &job_json)
}

/// Enumera `./jobs/*` y carga cada `job.json` vía `load_job` — un directorio con `job.json`
/// ilegible se saltea (log en stderr) en vez de abortar el listado completo. Usada tanto por
/// `GET /api/jobs` (`handlers::jobs_handler`) como por la tool MCP `list_audios`
/// (`mcp::listar_jobs`), que delega acá para no duplicar el `read_dir` + `load_job`.
pub fn list_jobs() -> anyhow::Result<Vec<JobMetadata>> {
    let mut jobs = Vec::new();
    for entrada in fs::read_dir(&crate::config::get().paths.jobs_dir)? {
        let entrada = entrada?;
        if !entrada.file_type()?.is_dir() {
            continue;
        }
        let job_id = entrada.file_name().to_string_lossy().into_owned();
        match load_job(&job_id) {
            Ok(metadata) => jobs.push(metadata),
            Err(e) => tracing::warn!("No se pudo leer el job '{job_id}': {e}"),
        }
    }
    Ok(jobs)
}

/// Lock exclusivo para tests que crean/cuentan entradas en el `./jobs` real (compartido con
/// `router::tests`, ver ahí) — `cargo test` corre los tests de un mismo binario en paralelo
/// dentro del mismo proceso, así que sin esto un test que cuenta entradas de `./jobs` antes/
/// después de una operación puede ver una entrada creada por otro test corriendo al mismo tiempo.
/// Solo existe en builds de test (`#[cfg(test)]`), no es código de producción.
///
/// `tokio::sync::Mutex` (no `std::sync::Mutex`, a diferencia de `JOB_METADATA_LOCK` arriba):
/// estos tests sostienen el guard mientras esperan una request HTTP simulada (`.await`), y
/// `clippy::await_holding_lock` marca correctamente que un `std::sync::MutexGuard` sostenido a
/// través de un `.await` es frágil (depende de que el runtime de test siga siendo
/// single-threaded). El Mutex async no tiene ese problema.
#[cfg(test)]
pub(crate) static JOBS_DIR_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// `update_job_metadata` no tiene orden garantizado entre llamadas concurrentes para el mismo
    /// `job_id` — lo que `JOB_METADATA_LOCK` garantiza es que el read-modify-write de cada llamada
    /// nunca se entrelaza con el de otra, así que `job.json` nunca queda en un estado corrupto o
    /// con un campo a medio escribir. Este test dispara varias mutaciones concurrentes de verdad
    /// (`tokio::spawn`, no secuenciales) y confirma que `load_job` sigue devolviendo un archivo
    /// válido al final.
    #[tokio::test]
    async fn update_job_metadata_concurrente_no_corrompe_job_json() {
        let _guard = JOBS_DIR_TEST_LOCK.lock().await;

        let metadata = create_job("wav").expect("no se pudo crear el job de prueba");
        let job_id = metadata.job_id.clone();

        let mut handles = Vec::new();
        for i in 0..20u32 {
            let job_id = job_id.clone();
            handles.push(tokio::spawn(async move {
                update_job_metadata(&job_id, move |m| {
                    m.status = if i % 2 == 0 {
                        JobStatus::Processing
                    } else {
                        JobStatus::Failed
                    };
                    m.transcript_ready = i % 3 == 0;
                })
            }));
        }

        for handle in handles {
            handle
                .await
                .expect("panic en la tarea")
                .expect("update_job_metadata falló");
        }

        let recargado =
            load_job(&job_id).expect("job.json quedó ilegible tras escrituras concurrentes");
        assert_eq!(recargado.job_id, job_id);

        let _ = std::fs::remove_dir_all(format!("./jobs/{job_id}"));
    }
}
