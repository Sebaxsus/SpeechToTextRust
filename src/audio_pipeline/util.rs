use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Escribe `contents` en `path` de forma atómica: primero a un archivo temporal en el mismo
/// directorio, `fsync` de esos datos, y recién después un `rename` (atómico dentro del mismo
/// filesystem, tanto en Windows como en Linux). El `fsync` antes del `rename` es la parte que
/// antes faltaba (hallazgo real, 2026-08-30: el corte de luz que interrumpió el job
/// `fa5df6b2-...` dejó `checkpoint.json` en 0 bytes — ver `docs/TODO.md`) — `rename` es atómico a
/// nivel de *metadata* del directorio, pero sin `fsync` los bytes del `.tmp` pueden seguir en el
/// write cache del SO/disco cuando el `rename` ya se confirmó; un corte de luz justo ahí deja el
/// archivo final existente pero vacío/con datos viejos, no el JSON completo. `File::sync_all`
/// fuerza esos bytes a disco antes de que el rename pueda confirmarse, así que ese escenario deja
/// de ser posible.
pub(crate) fn write_atomic(path: &Path, contents: &str) -> anyhow::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = std::path::PathBuf::from(tmp);

    let mut file = File::create(&tmp_path)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    drop(file);

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Epoch-segundos como string — mismo formato que `JobMetadata::created_at`. Extraído del cálculo
/// que antes vivía inline en `job::create_job` para reusarlo también en los timestamps de fase
/// nuevos (`processing_started_at`/`transcript_ready_at`/`completed_at`, ver
/// `handlers::audio_handler::lanzar_procesamiento_job`).
pub(crate) fn now_epoch_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}
