use std::path::Path;

/// Escribe `contents` en `path` de forma atómica: primero a un archivo temporal en el mismo
/// directorio, después un `rename` (atómico dentro del mismo filesystem, tanto en Windows como en
/// Linux). Sin esto, un crash a mitad de un `fs::write` directo deja el JSON truncado/corrupto —
/// justo el escenario de "crash a mitad de camino" que `checkpoint.json`/`job.json` existen para
/// tolerar (ver CLAUDE.local.md: checkpointing y crash recovery).
pub(crate) fn write_atomic(path: &Path, contents: &str) -> anyhow::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = std::path::PathBuf::from(tmp);

    std::fs::write(&tmp_path, contents)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}
