use std::io::{BufRead, BufReader};

use ollama_rs::Ollama;
use ollama_rs::generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest};
use ollama_rs::generation::parameters::KeepAlive;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType, PointStruct,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use uuid::Uuid;

use crate::audio_pipeline::models::TranscriptEntry;

/// Colección única para todo el corpus (ver CLAUDE.local.md: Qdrant — topología). No una
/// colección por audio: a la escala de este proyecto (~600 vectores por audio de 5h) el overhead
/// fijo de HNSW/segmentos/WAL por colección no se justifica dividiendo.
///
/// `pub(crate)` (junto con `EMBEDDING_MODEL` y `deterministic_point_id` abajo) para que
/// `crate::rag::retrieval` (Fase 5) los reutilice en vez de duplicar el nombre de colección, el
/// modelo de embeddings o el esquema de point ID — una divergencia ahí haría que la Fase 5
/// buscara silenciosamente en la colección equivocada o recalculara IDs que no matchean.
pub(crate) const COLLECTION_NAME: &str = "transcripts";
/// `bge-m3` — dense, 1024 dims, soporte multilingüe real (español incluido).
pub(crate) const EMBEDDING_MODEL: &str = "bge-m3";
const EMBEDDING_DIM: u64 = 1024;

/// Crea la colección `transcripts` si no existe todavía: `Cosine`, vectores+payload `on_disk`
/// (el corpus crece indefinidamente y no debe forzarse entero a RAM), sin quantization
/// (accuracy es la prioridad #1, no vale el ahorro de RAM a este volumen), e índice de payload
/// sobre `audio_id` para que filtrar por audio dentro de la colección única sea casi gratis.
pub async fn ensure_collection(qdrant: &Qdrant) -> anyhow::Result<()> {
    if qdrant.collection_exists(COLLECTION_NAME).await? {
        return Ok(());
    }

    qdrant
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION_NAME)
                .vectors_config(
                    VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine).on_disk(true),
                )
                .on_disk_payload(true),
        )
        .await?;

    qdrant
        .create_field_index(CreateFieldIndexCollectionBuilder::new(
            COLLECTION_NAME,
            "audio_id",
            FieldType::Keyword,
        ))
        .await?;

    Ok(())
}

/// Point ID determinístico (`Uuid::new_v5` sobre `audio_id:chunk_id`, no UUID random) para que un
/// reintento de esta fase tras un crash haga upsert idempotente en vez de duplicar puntos — ver
/// CLAUDE.local.md, no hace falta un checkpoint propio para Fase 4.
pub(crate) fn deterministic_point_id(audio_id: &str, chunk_id: usize) -> String {
    let name = format!("{audio_id}:{chunk_id}");
    Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes()).to_string()
}

/// Lee `transcript.jsonl` línea por línea (streaming — nunca carga el archivo completo en
/// memoria de una vez, aunque para un solo job el texto son pocos KB), genera un embedding por
/// línea con Ollama y hace upsert en Qdrant con point ID determinístico. Debe correr como tarea
/// async normal (nunca `spawn_blocking`) **después** de que `run_pipeline` (Fase 2/3, síncrono y
/// CPU-bound dentro de `spawn_blocking`) haya terminado y `WhisperRunner` ya se haya dropeado —
/// nunca solapar Whisper con la llamada a Ollama (ver CLAUDE.local.md: Concurrencia).
pub async fn run_embedding_phase(
    ollama: &Ollama,
    qdrant: &Qdrant,
    audio_id: &str,
    transcript_path: &str,
) -> anyhow::Result<()> {
    ensure_collection(qdrant).await?;

    let file = std::fs::File::open(transcript_path)?;
    let mut lines = BufReader::new(file).lines().peekable();

    while let Some(line) = lines.next() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let entry: TranscriptEntry = serde_json::from_str(&line)?;
        if entry.text.trim().is_empty() {
            // Chunk de silencio/no-speech — no vale la pena embeberlo.
            continue;
        }

        let is_last_line = lines.peek().is_none();
        let mut request = GenerateEmbeddingsRequest::new(
            EMBEDDING_MODEL.to_string(),
            EmbeddingsInput::Single(entry.text.clone()),
        );
        if is_last_line {
            // Libera el modelo de Ollama al terminar el job — nunca queda residente entre jobs
            // (ver CLAUDE.local.md: Ollama). Las llamadas intermedias dejan el keep_alive por
            // default (Ollama gestiona su propio timeout corto).
            request = request.keep_alive(KeepAlive::UnloadOnCompletion);
        }

        let response = ollama.generate_embeddings(request).await?;
        let embedding = response.embeddings.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!(
                "Ollama no devolvió ningún embedding para el chunk {}",
                entry.chunk
            )
        })?;
        tracing::debug!(
            audio_id,
            chunk = entry.chunk,
            embedding_dim = embedding.len(),
            "Ollama devolvió el embedding del chunk"
        );

        let point_id = deterministic_point_id(audio_id, entry.chunk);
        let payload: Payload = serde_json::json!({
            "audio_id": audio_id,
            "chunk_id": entry.chunk,
            "start": entry.start,
            "end": entry.end,
            "text": entry.text,
            "speaker": "unknown",
            "avg_logprob": entry.avg_logprob,
        })
        .try_into()?;

        let point = PointStruct::new(point_id.clone(), embedding, payload);
        qdrant
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, vec![point]))
            .await?;
        tracing::debug!(
            audio_id,
            chunk = entry.chunk,
            point_id,
            "Qdrant confirmó el upsert del punto"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::{Condition, CountPointsBuilder, DeletePointsBuilder, Filter};

    /// Prueba de punta a punta contra un Ollama y un Qdrant reales (no mockeados) — necesita
    /// `bge-m3` ya instalado en Ollama (`ollama pull bge-m3`) y un Qdrant escuchando en
    /// `localhost:6334` (ver README/`docs/TODO.md` para levantarlo con Docker). Ignorado por
    /// defecto porque, a diferencia del resto de la suite, depende de servicios externos vivos.
    ///
    /// Corrida manual: `cargo test embeddings -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requiere Ollama (bge-m3) y Qdrant reales corriendo en localhost"]
    async fn run_embedding_phase_contra_servicios_reales() {
        let ollama = Ollama::default();
        let qdrant = Qdrant::from_url("http://localhost:6334")
            .build()
            .expect("no se pudo construir el cliente de Qdrant");

        // audio_id único por corrida para no chocar con datos reales ya en la colección, y para
        // poder limpiar exactamente lo que este test escribió al final.
        let audio_id = format!("test-embeddings-{}", uuid::Uuid::new_v4());

        let transcript_path = std::env::temp_dir().join(format!("{audio_id}.jsonl"));
        let jsonl = concat!(
            "{\"chunk\":0,\"start\":0.0,\"end\":30.0,\"text\":\"Hola, esto es una prueba de la fase de embeddings.\",\"avg_logprob\":-0.2}\n",
            "{\"chunk\":1,\"start\":30.0,\"end\":60.0,\"text\":\"\",\"avg_logprob\":0.0}\n",
            "{\"chunk\":2,\"start\":60.0,\"end\":90.0,\"text\":\"Segunda línea con contenido real para embeber.\",\"avg_logprob\":-0.15}\n",
        );
        std::fs::write(&transcript_path, jsonl).expect("no se pudo escribir el jsonl de prueba");

        let result = run_embedding_phase(
            &ollama,
            &qdrant,
            &audio_id,
            transcript_path.to_str().unwrap(),
        )
        .await;

        let _ = std::fs::remove_file(&transcript_path);

        result.expect("run_embedding_phase falló contra los servicios reales");

        let filter = Filter::must([Condition::matches("audio_id", audio_id.clone())]);

        let count = qdrant
            .count(
                CountPointsBuilder::new(COLLECTION_NAME)
                    .filter(filter.clone())
                    .exact(true),
            )
            .await
            .expect("no se pudo contar los puntos insertados")
            .result
            .expect("respuesta de count sin resultado")
            .count;

        // Limpieza: siempre borrar los puntos de este audio_id de prueba, incluso si la
        // aserción de abajo falla, para no ensuciar la colección real entre corridas.
        let cleanup = qdrant
            .delete_points(DeletePointsBuilder::new(COLLECTION_NAME).points(filter))
            .await;

        assert_eq!(
            count, 2,
            "se esperaban 2 puntos (el chunk de texto vacío debe saltarse)"
        );
        cleanup.expect("no se pudieron limpiar los puntos de prueba");
    }
}
