//! Resumen por audio vía Ollama, map-reduce por lotes (ver CLAUDE.local.md: "Resumen por audio").
//!
//! Lee `transcript.jsonl` directo (streaming, sin pasar por Qdrant/embeddings — ya tiene todos
//! los chunks del audio en orden), agrupa en lotes acotados por cantidad de chunks y por
//! caracteres, resume cada lote, y si hubo más de un lote consolida los resúmenes parciales en
//! uno solo. Pensado para correr como tarea secundaria después de que un job llega a
//! `JobStatus::Completed` (ver `handlers::audio_handler::lanzar_generacion_resumen`), nunca
//! bloqueando el pipeline principal.
//!
//! Decisión de diseño (2026-08-01, ver CLAUDE.local.md): se descartó tanto una sola llamada a
//! contexto máximo (32768 tokens — riesgo real de quedarse sin VRAM en una GPU de 6GB) como un
//! map-reduce sin agrupar (~600 llamadas sueltas para un audio de 5h) — los lotes moderados
//! (contexto ~8192 tokens) evitan ambos extremos.

use std::io::{BufRead, BufReader};

use ollama_rs::Ollama;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::parameters::KeepAlive;
use ollama_rs::models::ModelOptions;

use super::generation::GENERATION_MODEL;
use crate::audio_pipeline::models::TranscriptEntry;

/// Un lote se cierra al llegar a lo que ocurra primero — cantidad de chunks o caracteres. El
/// límite de caracteres es una red de seguridad si algunos chunks son inusualmente largos; en la
/// práctica, para un audio de 5h (~600 chunks de 30s), el límite de chunks es el que normalmente
/// cierra cada lote, dando ~10-15 lotes.
const BATCH_MAX_CHUNKS: usize = 50;
const BATCH_MAX_CHARS: usize = 12_000;
/// Contexto moderado — muy por debajo del techo de 32768 tokens del modelo, con margen cómodo en
/// una GPU de 6GB de VRAM (ver CLAUDE.local.md).
const BATCH_NUM_CTX: u64 = 8192;
const BATCH_NUM_PREDICT: i32 = 300;
/// La consolidación final recibe resúmenes de lote (mucho más cortos que el transcript crudo),
/// así que el mismo `BATCH_NUM_CTX` alcanza de sobra — se le da más presupuesto de salida porque
/// es el resumen que finalmente se persiste.
const CONSOLIDATION_NUM_PREDICT: i32 = 500;

const NO_SPEECH_MESSAGE: &str = "No se detectó contenido hablado para resumir.";

/// Agrupa las entradas (ya filtradas de texto vacío) en lotes de texto plano concatenado.
fn build_batches(entries: Vec<TranscriptEntry>) -> Vec<String> {
    let mut batches = Vec::new();
    let mut current = String::new();
    let mut current_chunks = 0usize;

    for entry in entries {
        let text = entry.text.trim();
        if text.is_empty() {
            continue;
        }

        let excede_limite =
            current_chunks >= BATCH_MAX_CHUNKS || current.len() + text.len() > BATCH_MAX_CHARS;
        if excede_limite && !current.is_empty() {
            batches.push(std::mem::take(&mut current));
            current_chunks = 0;
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(text);
        current_chunks += 1;
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

/// Resume un lote. `is_last_call` marca si esta es la última llamada a Ollama de todo el job de
/// resumen (solo ocurre cuando hay un único lote y no hace falta consolidación) — en ese caso se
/// libera el modelo al terminar (ver CLAUDE.local.md: Ollama, modelos no residentes).
async fn summarize_batch(
    ollama: &Ollama,
    batch_text: &str,
    is_last_call: bool,
) -> anyhow::Result<String> {
    let prompt = format!(
        "Resumí brevemente, en español, el contenido de este fragmento de una reunión grabada \
         por teléfono (puede tener habla superpuesta y ruido de fondo). Enfocate en decisiones, \
         temas tratados y datos concretos (números, fechas, nombres). No inventes información \
         que no esté en el texto.\n\nFragmento:\n{batch_text}"
    );

    let mut request = ChatMessageRequest::new(
        GENERATION_MODEL.to_string(),
        vec![ChatMessage::user(prompt)],
    )
    .options(
        ModelOptions::default()
            .num_ctx(BATCH_NUM_CTX)
            .num_predict(BATCH_NUM_PREDICT),
    );
    if is_last_call {
        request = request.keep_alive(KeepAlive::UnloadOnCompletion);
    }

    let response = ollama.send_chat_messages(request).await?;
    Ok(response.message.content)
}

/// Consolida los resúmenes de lote en uno solo — siempre la última llamada del job (libera el
/// modelo al terminar).
async fn consolidate_summaries(
    ollama: &Ollama,
    batch_summaries: &[String],
) -> anyhow::Result<String> {
    let joined = batch_summaries.join("\n\n");
    let prompt = format!(
        "A continuación tenés varios resúmenes parciales de una misma reunión, en orden \
         cronológico. Consolidalos en un solo resumen coherente y breve (2-4 párrafos), en \
         español, sin repetir información entre secciones.\n\nResúmenes parciales:\n{joined}"
    );

    let request = ChatMessageRequest::new(
        GENERATION_MODEL.to_string(),
        vec![ChatMessage::user(prompt)],
    )
    .options(
        ModelOptions::default()
            .num_ctx(BATCH_NUM_CTX)
            .num_predict(CONSOLIDATION_NUM_PREDICT),
    )
    .keep_alive(KeepAlive::UnloadOnCompletion);

    let response = ollama.send_chat_messages(request).await?;
    Ok(response.message.content)
}

/// Genera el resumen de un audio a partir de su `transcript.jsonl` — ver el módulo para el
/// diseño de map-reduce por lotes. No toca Qdrant/embeddings en absoluto.
pub async fn generate_summary(ollama: &Ollama, transcript_path: &str) -> anyhow::Result<String> {
    let file = std::fs::File::open(transcript_path)?;
    let mut entries = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        entries.push(serde_json::from_str::<TranscriptEntry>(&line)?);
    }

    let batches = build_batches(entries);

    if batches.is_empty() {
        return Ok(NO_SPEECH_MESSAGE.to_string());
    }

    if batches.len() == 1 {
        return summarize_batch(ollama, &batches[0], true).await;
    }

    let mut batch_summaries = Vec::with_capacity(batches.len());
    for batch in &batches {
        batch_summaries.push(summarize_batch(ollama, batch, false).await?);
    }

    consolidate_summaries(ollama, &batch_summaries).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(chunk: usize, text: &str) -> TranscriptEntry {
        TranscriptEntry {
            chunk,
            start: chunk as f32 * 30.0,
            end: (chunk + 1) as f32 * 30.0,
            text: text.to_string(),
            avg_logprob: -0.2,
        }
    }

    #[test]
    fn build_batches_saltea_texto_vacio_y_respeta_limite_de_chunks() {
        let mut entries: Vec<TranscriptEntry> = (0..BATCH_MAX_CHUNKS + 5)
            .map(|i| entry(i, "palabra"))
            .collect();
        entries.push(entry(BATCH_MAX_CHUNKS + 5, "")); // silencio, debe saltearse

        let batches = build_batches(entries);

        assert_eq!(
            batches.len(),
            2,
            "debería cerrar el primer lote a los 50 chunks"
        );
        assert_eq!(
            batches[0].split_whitespace().count(),
            BATCH_MAX_CHUNKS,
            "el primer lote debe tener exactamente BATCH_MAX_CHUNKS palabras (una por chunk)"
        );
    }

    #[test]
    fn build_batches_sin_entradas_con_texto_da_lista_vacia() {
        let entries = vec![entry(0, ""), entry(1, "   ")];
        assert!(build_batches(entries).is_empty());
    }

    #[tokio::test]
    async fn generate_summary_sobre_transcript_vacio_no_llama_a_ollama() {
        let dir = std::env::temp_dir().join(format!("summary-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let transcript_path = dir.join("transcript.jsonl");
        std::fs::write(
            &transcript_path,
            "{\"chunk\":0,\"start\":0.0,\"end\":30.0,\"text\":\"\",\"avg_logprob\":0.0}\n",
        )
        .unwrap();

        // `Ollama::default()` no se usa realmente si el transcript está vacío — la función debe
        // devolver el mensaje fijo sin intentar ninguna llamada de red.
        let ollama = Ollama::default();
        let resultado = generate_summary(&ollama, transcript_path.to_str().unwrap())
            .await
            .expect("no debería fallar sobre un transcript vacío");

        assert_eq!(resultado, NO_SPEECH_MESSAGE);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Prueba de punta a punta contra Ollama real: siembra un transcript sintético con más
    /// chunks que `BATCH_MAX_CHUNKS` (fuerza al menos 2 lotes + consolidación) y confirma que el
    /// resumen final menciona contenido de más de un lote, no solo el primero.
    ///
    /// Ignorado por defecto: requiere `qcwind/qwen2.5-7b-instruct-Q4_K_M` real en Ollama.
    /// Corrida manual: `cargo test generate_summary_multi_lote -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requiere Ollama (qcwind/qwen2.5-7b-instruct-Q4_K_M) real"]
    async fn generate_summary_multi_lote_contra_ollama_real() {
        let ollama = Ollama::default();

        let dir = std::env::temp_dir().join(format!("summary-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let transcript_path = dir.join("transcript.jsonl");

        let mut jsonl = String::new();
        for i in 0..BATCH_MAX_CHUNKS {
            jsonl.push_str(&format!(
                "{{\"chunk\":{i},\"start\":{s},\"end\":{e},\"text\":\"Estamos hablando del presupuesto de marketing.\",\"avg_logprob\":-0.2}}\n",
                s = i as f32 * 30.0,
                e = (i + 1) as f32 * 30.0,
            ));
        }
        for i in BATCH_MAX_CHUNKS..BATCH_MAX_CHUNKS + 5 {
            jsonl.push_str(&format!(
                "{{\"chunk\":{i},\"start\":{s},\"end\":{e},\"text\":\"Ahora pasamos a coordinar la fecha de la próxima reunión con el arquitecto Julián.\",\"avg_logprob\":-0.2}}\n",
                s = i as f32 * 30.0,
                e = (i + 1) as f32 * 30.0,
            ));
        }
        std::fs::write(&transcript_path, &jsonl).unwrap();

        let resumen = generate_summary(&ollama, transcript_path.to_str().unwrap())
            .await
            .expect("generate_summary falló contra Ollama real");

        let _ = std::fs::remove_dir_all(&dir);

        println!("Resumen consolidado: {resumen}");
        let resumen_lower = resumen.to_lowercase();
        assert!(
            resumen_lower.contains("presupuesto") || resumen_lower.contains("marketing"),
            "el resumen no menciona contenido del primer lote: {resumen}"
        );
        assert!(
            resumen_lower.contains("julián") || resumen_lower.contains("reunión"),
            "el resumen no menciona contenido del segundo lote: {resumen}"
        );
    }
}
