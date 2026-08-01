use ollama_rs::Ollama;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::parameters::KeepAlive;
use ollama_rs::models::ModelOptions;
use qdrant_client::Qdrant;

use super::retrieval::{ChunkHit, SearchScope, search};
use crate::rag::LOW_CONFIDENCE_THOLD;

/// Generación por defecto para ambos `SearchScope` (ver CLAUDE.local.md: "Generación — Ollama
/// por defecto, Gemini Nano opcional") — la prioridad #1 del proyecto es accuracy, no descargar
/// cómputo a un modelo on-device más liviano.
pub(crate) const GENERATION_MODEL: &str = "qcwind/qwen2.5-7b-instruct-Q4_K_M:latest";
/// Top-k dentro del rango 5-8 fijado en CLAUDE.local.md.
const TOP_K: u64 = 6;
/// El default de Ollama (2048) es fácil de exceder incluso con un contexto ensamblado modesto
/// (`TOP_K=6` hits + vecinos) — hallazgo real (2026-08-01, ver CLAUDE.local.md): `rag_answer`
/// nunca seteaba `num_ctx`, con riesgo de truncamiento silencioso del contexto. El modelo
/// (`qcwind/qwen2.5-7b-instruct-Q4_K_M`) soporta hasta 32768 tokens; 4096 da margen real sin
/// acercarse a ese techo.
const RAG_ANSWER_NUM_CTX: u64 = 4096;

fn assemble_context(hits: &[ChunkHit]) -> String {
    let mut context = String::new();
    for hit in hits {
        let aviso = if hit.chunk.avg_logprob < LOW_CONFIDENCE_THOLD {
            " [BAJA CONFIANZA — posible ruido de fondo o cross-talk, ver CLAUDE.local.md]"
        } else {
            ""
        };
        context.push_str(&format!(
            "[audio {} | chunk {} | {:.1}s-{:.1}s{}]\n{}\n\n",
            hit.chunk.audio_id,
            hit.chunk.chunk_id,
            hit.chunk.start,
            hit.chunk.end,
            aviso,
            hit.chunk.text,
        ));
    }
    context
}

/// Retrieval (ver `retrieval::search`) + generación server-side vía Ollama — el default y único
/// camino de generación implementado en el binario de Rust; Gemini Nano client-side (ver
/// CLAUDE.local.md) se invoca desde el browser, nunca desde acá.
///
/// `keep_alive` se fuerza a `UnloadOnCompletion`: en el flujo request/response simple del MVP de
/// RAG no vale la pena mantener un modelo de 7-8B residente entre preguntas sueltas (ver
/// CLAUDE.local.md: Ollama — "los modelos NO deben permanecer cargados permanentemente").
pub async fn rag_answer(
    ollama: &Ollama,
    qdrant: &Qdrant,
    question: &str,
    scope: &SearchScope,
) -> anyhow::Result<String> {
    let hits = search(ollama, qdrant, question, scope, TOP_K).await?;

    if hits.is_empty() {
        return Ok(
            "No se encontró contexto relevante en la transcripción para responder esta pregunta."
                .to_string(),
        );
    }

    let context = assemble_context(&hits);

    let system_prompt = format!(
        "Sos un asistente que responde preguntas sobre transcripciones de reuniones grabadas \
         con teléfono (audio con cross-talk y ruido de fondo real). Respondé ÚNICAMENTE en base \
         al contexto de abajo, en español. Si el contexto no alcanza para responder, decilo \
         explícitamente en vez de inventar. Si citás un tramo marcado como BAJA CONFIANZA, \
         aclará que esa parte de la transcripción puede no ser exacta.\n\nContexto:\n{context}"
    );

    let messages = vec![
        ChatMessage::system(system_prompt),
        ChatMessage::user(question.to_string()),
    ];

    let request = ChatMessageRequest::new(GENERATION_MODEL.to_string(), messages)
        .options(ModelOptions::default().num_ctx(RAG_ANSWER_NUM_CTX))
        .keep_alive(KeepAlive::UnloadOnCompletion);

    let response = ollama.send_chat_messages(request).await?;

    Ok(response.message.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_pipeline::embeddings::{COLLECTION_NAME, run_embedding_phase};
    use qdrant_client::qdrant::{Condition, DeletePointsBuilder, Filter};

    /// Prueba de punta a punta contra Ollama y Qdrant reales: siembra una transcripción
    /// sintética vía `run_embedding_phase` (Fase 4, ya probado en `embeddings.rs`), corre
    /// `rag_answer` (Fase 5) sobre ese audio_id, y limpia los puntos de prueba al terminar. Es el
    /// primer test que ejercita retrieval + generación juntos, no solo aislados.
    ///
    /// Ignorado por defecto: requiere `bge-m3` y `qcwind/qwen2.5-7b-instruct-Q4_K_M` instalados
    /// en Ollama, y Qdrant escuchando en `localhost:6334`. Corrida manual:
    /// `cargo test rag_answer -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requiere Ollama (bge-m3 + qcwind/qwen2.5-7b-instruct-Q4_K_M) y Qdrant reales"]
    async fn rag_answer_contra_servicios_reales() {
        let ollama = Ollama::default();
        let qdrant = Qdrant::from_url("http://localhost:6334")
            .build()
            .expect("no se pudo construir el cliente de Qdrant");

        let audio_id = format!("test-rag-{}", uuid::Uuid::new_v4());
        let transcript_path = std::env::temp_dir().join(format!("{audio_id}.jsonl"));
        let jsonl = concat!(
            "{\"chunk\":0,\"start\":0.0,\"end\":30.0,\"text\":\"El presupuesto del proyecto para el próximo trimestre es de cinco mil dólares.\",\"avg_logprob\":-0.1}\n",
            "{\"chunk\":1,\"start\":30.0,\"end\":60.0,\"text\":\"Ese presupuesto se va a repartir entre marketing y desarrollo.\",\"avg_logprob\":-0.1}\n",
            "{\"chunk\":2,\"start\":60.0,\"end\":90.0,\"text\":\"Cambiando de tema, mañana hay reunión con el cliente a las diez.\",\"avg_logprob\":-0.1}\n",
        );
        std::fs::write(&transcript_path, jsonl).expect("no se pudo escribir el jsonl de prueba");

        let seed_result = run_embedding_phase(
            &ollama,
            &qdrant,
            &audio_id,
            transcript_path.to_str().unwrap(),
        )
        .await;
        let _ = std::fs::remove_file(&transcript_path);
        seed_result.expect("no se pudo sembrar los embeddings de prueba");

        let respuesta = rag_answer(
            &ollama,
            &qdrant,
            "¿Cuál es el presupuesto del proyecto?",
            &SearchScope::Audio(audio_id.clone()),
        )
        .await;

        // Limpieza: siempre borrar los puntos de este audio_id de prueba, incluso si la
        // aserción de abajo falla.
        let filter = Filter::must([Condition::matches("audio_id", audio_id.clone())]);
        let cleanup = qdrant
            .delete_points(DeletePointsBuilder::new(COLLECTION_NAME).points(filter))
            .await;

        let respuesta = respuesta.expect("rag_answer falló contra los servicios reales");
        println!("Respuesta del RAG: {respuesta}");
        cleanup.expect("no se pudieron limpiar los puntos de prueba");

        let respuesta_lower = respuesta.to_lowercase();
        assert!(
            respuesta_lower.contains("5000")
                || respuesta_lower.contains("5.000")
                || respuesta_lower.contains("cinco mil"),
            "la respuesta no parece mencionar el presupuesto recuperado del contexto: {respuesta}"
        );
    }
}
