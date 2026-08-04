use crate::audio_pipeline::embeddings::run_embedding_phase;
use crate::audio_pipeline::job::{create_job, load_job, update_job_metadata};
use crate::audio_pipeline::models::{JobMetadata, JobStatus, SummaryStatus};
use crate::audio_pipeline::pipeline::run_pipeline;
use crate::audio_pipeline::util::{now_epoch_string, write_atomic};
use crate::router::UPLOAD_BODY_LIMIT_BYTES;
use crate::state::SharedState;
use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

/// Acepta solo `http`/`https` — chequeo mínimo a propósito (no un parseo completo de URL con la
/// crate `url`), suficiente para descartar valores claramente inválidos antes de guardarlos; el
/// riesgo de SSRF de una URL "válida" mal intencionada queda como riesgo aceptado y documentado
/// (ver `JobMetadata::callback_url`), no algo que este chequeo pretenda cerrar del todo.
fn callback_url_es_valida(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Recarga `job.json` (mismo criterio que `lanzar_generacion_resumen`: siempre ver el
/// `callback_url` más reciente, no una copia de `JobMetadata` que pudo quedar desactualizada
/// desde que se creó el job) y dispara el webhook si el job tiene uno configurado. No-op
/// silencioso (solo un log) si el job no tiene `callback_url` o si `load_job` falla.
fn disparar_webhook_si_corresponde(audio_id: &str, status: JobStatus) {
    match load_job(audio_id) {
        Ok(metadata) => {
            if let Some(url) = metadata.callback_url {
                notificar_callback_url(url, audio_id.to_string(), status);
            }
        }
        Err(e) => {
            tracing::warn!("No se pudo recargar el job '{audio_id}' para el webhook: {e}");
        }
    }
}

/// Dispara un POST best-effort (`{"job_id", "status"}`) a `callback_url` cuando un job termina
/// (`Completed`/`Failed`) — ver `JobMetadata::callback_url`. `tokio::spawn` aparte: nunca bloquea
/// ni puede fallar la transición de estado que ya se persistió en `job.json` antes de llamar acá.
/// Un único intento, sin cola de reintentos (mismo criterio ya aceptado en el proyecto para
/// fallos de Fase 4/resumen — ver `docs/TODO.md`), con timeout corto para no dejar la tarea
/// colgada si la URL no responde.
fn notificar_callback_url(callback_url: String, job_id: String, status: JobStatus) {
    #[derive(serde::Serialize)]
    struct WebhookPayload {
        job_id: String,
        status: JobStatus,
    }

    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                tracing::warn!("No se pudo construir el cliente HTTP del webhook: {e}");
                return;
            }
        };

        let payload = WebhookPayload {
            job_id: job_id.clone(),
            status,
        };

        match client.post(&callback_url).json(&payload).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!(target: "lifecycle", job_id = %job_id, "Webhook notificado en {callback_url}");
            }
            Ok(resp) => {
                tracing::warn!(
                    "Webhook a {callback_url} respondió {} para el job '{job_id}'",
                    resp.status()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "No se pudo notificar el webhook {callback_url} para el job '{job_id}': {e}"
                );
            }
        }
    });
}

/// Mide la duración total del audio con `ffprobe` (mismo binario que ya requiere el fallback de
/// `decoder.rs`, ver `CLAUDE.local.md`) — `tokio::process::Command` async, no la variante sync
/// que usa `decoder.rs`, porque acá corre dentro de un handler de Axum, no de un
/// `spawn_blocking`. `None` en cualquier falla (binario ausente, salida inesperada): nunca hace
/// fallar el upload por esto, solo deja `duration_seconds` sin poblar.
async fn probe_duration_seconds(audio_path: &str) -> Option<f32> {
    let output = tokio::process::Command::new(&crate::config::get().paths.ffprobe_bin)
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            audio_path,
        ])
        .output()
        .await;

    let output = match output {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            tracing::warn!(
                "ffprobe no pudo leer la duración de {audio_path}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return None;
        }
        Err(e) => {
            tracing::warn!("no se pudo ejecutar ffprobe para {audio_path}: {e}");
            return None;
        }
    };

    let json: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!("salida de ffprobe ilegible para {audio_path}: {e}");
            return None;
        }
    };

    json["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f32>().ok())
}

/// Detecta el contenedor real a partir de los primeros bytes del archivo (magic bytes), en vez
/// de confiar en la extensión declarada por el cliente. Devuelve la extensión interna ("mp3" /
/// "mp4" / "wav") que se usa después para el nombre de archivo en disco, o `None` si no es un
/// formato soportado.
///
/// mp4 y m4a son el mismo contenedor ISO-BMFF (caja `ftyp`), así que ambos caen en "mp4". El wav
/// no necesita ningún cambio en `StreamingDecoder`: Symphonia lo decodifica nativo (feature `all`
/// ya habilitado), sin pasar por el fallback de ffmpeg.
fn sniff_audio_extension(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return Some("mp3");
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return Some("mp3");
    }
    if bytes.len() >= 8 && &bytes[4..8] == b"ftyp" {
        return Some("mp4");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some("wav");
    }
    None
}

pub async fn recibir_y_procesar_audio(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    tracing::info!("Recibiendo petición en la app: {}", state.nombre_app);

    let Ok(Some(mut campo)) = multipart.next_field().await else {
        return (
            StatusCode::BAD_REQUEST,
            "Error al leer el archivo multipart",
        )
            .into_response();
    };

    // `None` si el cliente no mandó ningún `filename` en el multipart — se persiste tal cual
    // (ver `JobMetadata::original_filename`), en vez de rellenarlo con un placeholder que
    // terminaría guardado como si fuera un nombre real.
    let nombre_archivo = campo.file_name().map(str::to_string);

    // Sniff del primer chunk ANTES de crear cualquier archivo o directorio en disco.
    let primer_chunk = match campo.chunk().await {
        Ok(Some(chunk)) => chunk,
        _ => {
            return (StatusCode::BAD_REQUEST, "Archivo vacío o inválido").into_response();
        }
    };

    let Some(extension) = sniff_audio_extension(&primer_chunk) else {
        return (
            StatusCode::BAD_REQUEST,
            "Formato no soportado: solo se aceptan mp3, mp4/m4a y wav",
        )
            .into_response();
    };

    let metadata = match create_job(extension, nombre_archivo.clone()) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("Error creando el job: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response();
        }
    };

    let mut archivo_local = match tokio::fs::File::create(&metadata.audio_path).await {
        Ok(file) => file,
        Err(e) => {
            tracing::error!("Error al crear el archivo: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error interno del servidor",
            )
                .into_response();
        }
    };

    // Streaming directo al SSD para evitar Thrashing de la RAM.
    if let Err(e) = archivo_local.write_all(&primer_chunk).await {
        tracing::error!("Error escribiendo en disco: {}", e);
        let _ =
            tokio::fs::remove_dir_all(crate::config::get().paths.jobs_dir.join(&metadata.job_id))
                .await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error al guardar el archivo",
        )
            .into_response();
    }

    loop {
        let chunk = match campo.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(e) => {
                // Un error acá (body sobre el límite de tamaño, conexión cortada a mitad de la
                // subida) NO es "fin del stream" — el `while let Ok(Some(chunk))` anterior lo
                // trataba como tal, escribiendo un archivo truncado en disco y respondiendo 202
                // igual. Ese archivo corrupto recién fallaba horas después, al decodificarlo en
                // Fase 2, sin ningún indicio de que la subida nunca se completó.
                tracing::error!("Error leyendo el archivo del multipart: {e}");
                let _ = tokio::fs::remove_dir_all(
                    crate::config::get().paths.jobs_dir.join(&metadata.job_id),
                )
                .await;
                return (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Error leyendo el archivo subido (¿se cortó la conexión, o el archivo \
                         supera el límite de {} MiB?)",
                        UPLOAD_BODY_LIMIT_BYTES / 1024 / 1024
                    ),
                )
                    .into_response();
            }
        };

        if let Err(e) = archivo_local.write_all(&chunk).await {
            tracing::error!("Error escribiendo en disco: {}", e);
            let _ = tokio::fs::remove_dir_all(
                crate::config::get().paths.jobs_dir.join(&metadata.job_id),
            )
            .await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error al guardar el archivo",
            )
                .into_response();
        }
    }

    // `campo` ya está completamente drenado (el loop de arriba llegó a `Ok(None)`), pero seguía
    // vivo como binding — soltarlo explícitamente antes de pedir el siguiente campo es necesario:
    // `Multipart::next_field()` devuelve un error de parseo si se llama mientras el `Field`
    // anterior no se dropeó todavía, aunque ya no le quede ningún byte por leer (verificado en
    // este proyecto: sin el `drop` explícito, `next_field()` fallaba con "Error parsing
    // `multipart/form-data` request" pese a que el campo del archivo ya estaba agotado).
    drop(campo);

    let job_id = metadata.job_id.clone();

    tracing::info!(
        target: "lifecycle",
        "Archivo {} guardado exitosamente en el SSD (job {}).",
        nombre_archivo.as_deref().unwrap_or("(sin nombre)"),
        job_id
    );

    // Campos opcionales del multipart DESPUÉS del archivo (el primer campo siempre es el audio,
    // sin importar su nombre — ver comentario de más arriba). Hoy solo se busca `callback_url`;
    // cualquier otro campo/nombre se ignora. Best-effort: una URL inválida o un campo ilegible se
    // loguean, nunca hacen fallar el upload (el archivo ya está guardado en disco a esta altura).
    while let Ok(Some(campo_extra)) = multipart.next_field().await {
        if campo_extra.name() != Some("callback_url") {
            continue;
        }
        match campo_extra.text().await {
            Ok(texto) if callback_url_es_valida(texto.trim()) => {
                let url = texto.trim().to_string();
                if let Err(e) = update_job_metadata(&job_id, |m| m.callback_url = Some(url)) {
                    tracing::error!("No se pudo guardar callback_url del job '{job_id}': {e}");
                }
            }
            Ok(texto) => {
                tracing::warn!("callback_url inválida ignorada para el job '{job_id}': {texto}");
            }
            Err(e) => {
                tracing::warn!("No se pudo leer el campo callback_url del job '{job_id}': {e}");
            }
        }
    }

    // Corto y sincrónico con la respuesta HTTP a propósito (a diferencia de Whisper/embeddings,
    // que corren en background): el cliente necesita `duration_seconds` disponible desde el
    // primer `GET /api/jobs` después del 202, no en un estado intermedio "todavía desconocida".
    // Best-effort: si ffprobe falla, se loguea y el job sigue con `duration_seconds: None` en vez
    // de fallar el upload por esto.
    if let Some(duration) = probe_duration_seconds(&metadata.audio_path).await
        && let Err(e) = update_job_metadata(&job_id, |m| m.duration_seconds = Some(duration))
    {
        tracing::error!("No se pudo guardar duration_seconds del job '{job_id}': {e}");
    }

    lanzar_procesamiento_job(state, metadata, false);

    (
        StatusCode::ACCEPTED,
        axum::Json(serde_json::json!({
            "job_id": job_id,
            "status": "processing",
        })),
    )
        .into_response()
}

/// Corre Fase 2/3 (Whisper) y, si termina bien, encadena Fase 4 (embeddings) — la misma cadena
/// tanto para un job recién creado (`recibir_y_procesar_audio`) como para uno reanudado
/// (`reanudar_job`). No hace falta distinguir "nuevo" de "reanudado" para la lógica del pipeline:
/// `run_pipeline` ya es resume-safe por sí mismo (lee `checkpoint.json` y retoma donde cortó — ver
/// `pipeline.rs`), y `run_embedding_phase` es idempotente (point ID determinístico), así que
/// reintentar Fase 4 sobre un transcript ya embebido tampoco duplica vectores. `es_resume` solo
/// existe para que el log de `lifecycle` diga "transcribiendo" o "resumiendo" según corresponda.
fn lanzar_procesamiento_job(state: SharedState, metadata: JobMetadata, es_resume: bool) {
    tokio::spawn(async move {
        let permit = state
            .heavy_compute_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        let _permit = permit;

        let audio_id = metadata.job_id.clone();
        let transcript_path = metadata.transcript_path.clone();

        if es_resume {
            tracing::info!(target: "lifecycle", job_id = %audio_id, "Resumiendo la transcripción del job");
        } else {
            tracing::info!(target: "lifecycle", job_id = %audio_id, "Transcribiendo el job");
        }

        // "Processing" se marca acá, recién con el permiso ya adquirido: significa "corriendo
        // activamente", no "encolado". Mientras un job espera detrás de otro job pesado en
        // heavy_compute_semaphore, su status queda en Pending (decisión explícita, ver
        // docs/TODO.md). Un fallo actualizando job.json se loguea pero nunca aborta el pipeline
        // real — es bookkeeping best-effort sobre trabajo que ya va a correr de todas formas.
        if let Err(e) = update_job_metadata(&audio_id, |m| {
            m.status = JobStatus::Processing;
            m.processing_started_at = Some(now_epoch_string());
        }) {
            tracing::error!("No se pudo marcar el job '{audio_id}' como Processing: {e}");
        }

        // Fase 2/3 (Whisper, CPU-bound) corre entera dentro de spawn_blocking y termina de
        // liberar WhisperRunner antes de que Fase 4 (embeddings, I/O-bound) arranque — nunca
        // se solapan Whisper y Ollama (ver CLAUDE.local.md: Concurrencia).
        match tokio::task::spawn_blocking(move || run_pipeline(metadata)).await {
            Ok(Err(e)) => {
                tracing::error!("Error en el pipeline: {}", e);
                if let Err(e) = update_job_metadata(&audio_id, |m| m.status = JobStatus::Failed) {
                    tracing::error!("No se pudo marcar el job '{audio_id}' como Failed: {e}");
                }
                disparar_webhook_si_corresponde(&audio_id, JobStatus::Failed);
            }
            Err(e) => {
                tracing::error!("Error en el pipeline (join error): {}", e);
                if let Err(e) = update_job_metadata(&audio_id, |m| m.status = JobStatus::Failed) {
                    tracing::error!("No se pudo marcar el job '{audio_id}' como Failed: {e}");
                }
                disparar_webhook_si_corresponde(&audio_id, JobStatus::Failed);
            }
            Ok(Ok(())) => {
                // transcript_ready refleja que Fase 2/3 terminó bien, independientemente de si
                // Fase 4 (embeddings, abajo) todavía está en curso o falla — un cliente puede
                // pedir el transcript aunque status todavía no sea Completed.
                if let Err(e) = update_job_metadata(&audio_id, |m| {
                    m.transcript_ready = true;
                    m.transcript_ready_at = Some(now_epoch_string());
                }) {
                    tracing::error!(
                        "No se pudo marcar transcript_ready para el job '{audio_id}': {e}"
                    );
                }

                tracing::info!(target: "lifecycle", job_id = %audio_id, "Generando embeddings del job");
                match run_embedding_phase(
                    &state.ollama,
                    &state.qdrant,
                    &audio_id,
                    &transcript_path,
                    &state.config.rag,
                )
                .await
                {
                    Ok(()) => {
                        if let Err(e) = update_job_metadata(&audio_id, |m| {
                            m.status = JobStatus::Completed;
                            m.completed_at = Some(now_epoch_string());
                        }) {
                            tracing::error!(
                                "No se pudo marcar el job '{audio_id}' como Completed: {e}"
                            );
                        }
                        disparar_webhook_si_corresponde(&audio_id, JobStatus::Completed);
                        lanzar_generacion_resumen(state.clone(), audio_id);
                    }
                    Err(e) => {
                        tracing::error!("Error en la fase de embeddings: {}", e);
                        if let Err(e) =
                            update_job_metadata(&audio_id, |m| m.status = JobStatus::Failed)
                        {
                            tracing::error!(
                                "No se pudo marcar el job '{audio_id}' como Failed: {e}"
                            );
                        }
                        disparar_webhook_si_corresponde(&audio_id, JobStatus::Failed);
                    }
                }
            }
        }
    });
}

/// Genera el resumen del audio (ver `rag::summary::generate_summary`) como tarea independiente,
/// disparada después de que `lanzar_procesamiento_job` llega a `JobStatus::Completed` — nunca
/// anidada dentro de esa tarea, para no demorar su transición a `Completed` ni la liberación de
/// su permiso de `heavy_compute_semaphore`.
///
/// Recarga `job.json` vía `load_job` en vez de recibir un `JobMetadata` ya en memoria: así
/// siempre ve el `summary_status` más reciente (relevante si esto se dispara de nuevo tras un
/// resume) en vez de una copia que pudo quedar desactualizada.
///
/// Adquiere su PROPIO permiso del mismo `heavy_compute_semaphore` que usan Whisper y las tools de
/// RAG (no un semáforo nuevo) — coherente con la decisión ya tomada en el proyecto de nunca
/// correr dos cargas pesadas a la vez, sea cual sea la combinación (ver docs/TODO.md). Sin
/// reintento automático si falla — mismo criterio ya aceptado para `run_embedding_phase`.
fn lanzar_generacion_resumen(state: SharedState, audio_id: String) {
    tokio::spawn(async move {
        let metadata = match load_job(&audio_id) {
            Ok(metadata) => metadata,
            Err(e) => {
                tracing::error!("No se pudo recargar el job '{audio_id}' para el resumen: {e}");
                return;
            }
        };

        if metadata.summary_status == SummaryStatus::Ready {
            return; // ya generado (ej. resume de un job ya completado) — no regenerar.
        }

        let _permit = state
            .heavy_compute_semaphore
            .clone()
            .acquire_owned()
            .await
            .unwrap();

        if let Err(e) =
            update_job_metadata(&audio_id, |m| m.summary_status = SummaryStatus::Generating)
        {
            tracing::error!("No se pudo marcar el job '{audio_id}' como Generating (resumen): {e}");
        }

        tracing::info!(target: "lifecycle", job_id = %audio_id, "Generando el resumen del job");
        match crate::rag::generate_summary(
            &state.ollama,
            &metadata.transcript_path,
            &state.config.rag,
        )
        .await
        {
            Ok(texto) => {
                let persistido = write_atomic(&metadata.summary_path(), &texto);
                let nuevo_status = if persistido.is_ok() {
                    tracing::info!(target: "lifecycle", job_id = %audio_id, "Resumen del job listo");
                    SummaryStatus::Ready
                } else {
                    if let Err(e) = persistido {
                        tracing::error!(
                            "No se pudo persistir el resumen del job '{audio_id}': {e}"
                        );
                    }
                    SummaryStatus::Failed
                };
                if let Err(e) = update_job_metadata(&audio_id, |m| m.summary_status = nuevo_status)
                {
                    tracing::error!(
                        "No se pudo actualizar summary_status del job '{audio_id}': {e}"
                    );
                }
            }
            Err(e) => {
                tracing::error!("Error generando el resumen del job '{audio_id}': {e}");
                if let Err(e) =
                    update_job_metadata(&audio_id, |m| m.summary_status = SummaryStatus::Failed)
                {
                    tracing::error!(
                        "No se pudo marcar el job '{audio_id}' como Failed (resumen): {e}"
                    );
                }
            }
        }
    });
}

/// Body JSON opcional de `POST /api/jobs/{job_id}/resume` — a diferencia del multipart de
/// upload, acá se lee como JSON simple porque el endpoint no maneja ningún archivo. Sin body (el
/// caso más común: reanudar sin cambiar nada) sigue siendo válido — ver `reanudar_job`.
#[derive(Debug, Deserialize, Default)]
struct ResumeBody {
    #[serde(default)]
    callback_url: Option<String>,
}

/// `POST /api/jobs/{job_id}/resume` — reanuda un job existente cuyo procesamiento se cortó a
/// mitad de camino (proceso matado, crash, etc. — ver docs/TODO.md, Fase 3). `run_pipeline` ya
/// era resume-safe a nivel de función; lo que faltaba era este entry point HTTP.
///
/// Deliberadamente NO es una tool de MCP (ver CLAUDE.local.md: "MCP de solo lectura") — el
/// servidor MCP nunca dispara procesamiento, ni siquiera para "solo continuar" algo ya pedido,
/// porque esa es la línea que hace aceptable exponerlo en LAN.
pub async fn reanudar_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    let metadata = match load_job(&job_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::error!("No se pudo reanudar el job '{job_id}': {e}");
            return (StatusCode::NOT_FOUND, "Job no encontrado").into_response();
        }
    };

    // `Bytes` en vez de `Json<Option<ResumeBody>>`: el caso normal (reanudar sin body) no trae
    // ningún JSON válido, y el extractor `Json` de axum rechaza un body vacío en vez de tratarlo
    // como "no mandaron nada". Un body presente pero mal formado se ignora (se loguea) en vez de
    // devolver 400 — el resume en sí es válido igual, con o sin ese dato.
    if !body.is_empty() {
        match serde_json::from_slice::<ResumeBody>(&body) {
            Ok(ResumeBody {
                callback_url: Some(url),
            }) if callback_url_es_valida(&url) => {
                if let Err(e) = update_job_metadata(&job_id, |m| m.callback_url = Some(url)) {
                    tracing::error!("No se pudo actualizar callback_url del job '{job_id}': {e}");
                }
            }
            Ok(ResumeBody {
                callback_url: Some(url),
            }) => {
                tracing::warn!("callback_url inválida ignorada para el job '{job_id}': {url}");
            }
            Ok(ResumeBody { callback_url: None }) => {}
            Err(e) => {
                tracing::warn!("Body de resume ilegible para el job '{job_id}', se ignora: {e}");
            }
        }
    }

    lanzar_procesamiento_job(state, metadata, true);

    (
        StatusCode::ACCEPTED,
        axum::Json(serde_json::json!({
            "job_id": job_id,
            "status": "processing",
        })),
    )
        .into_response()
}

#[cfg(test)]
mod sniff_tests {
    use super::sniff_audio_extension;

    #[test]
    fn sniff_reconoce_mp3_mp4_y_wav() {
        assert_eq!(sniff_audio_extension(b"ID3\x03\x00\x00\x00"), Some("mp3"));
        assert_eq!(
            sniff_audio_extension(&[0xFF, 0xFB, 0x90, 0x00]),
            Some("mp3")
        );
        assert_eq!(
            sniff_audio_extension(b"\x00\x00\x00\x18ftypmp42"),
            Some("mp4")
        );
        assert_eq!(
            sniff_audio_extension(b"RIFF\x24\x08\x00\x00WAVEfmt "),
            Some("wav")
        );
        assert_eq!(sniff_audio_extension(b"esto no es audio"), None);
    }
}
