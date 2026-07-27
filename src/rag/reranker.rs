use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::{PaddingParams, Tokenizer};

use super::retrieval::ChunkHit;

/// Cross-encoder candidato fijado en CLAUDE.local.md — XLM-RoBERTa-large con una cabeza de
/// clasificación de una sola salida (relevance score), multilingüe (incluye español). Mismo
/// repo/arquitectura que usa el ejemplo oficial de `candle` (`candle-examples/examples/xlm-roberta`,
/// tarea `Reranker`), que es la referencia sobre la que se basó esta implementación.
const RERANKER_MODEL_ID: &str = "BAAI/bge-reranker-v2-m3";
const RERANKER_REVISION: &str = "main";

/// Reordena `hits` por relevancia real query↔passage con el cross-encoder — SOLO tiene sentido
/// para `SearchScope::AllCorpus` (ver CLAUDE.local.md: con un solo audio el ranking por coseno
/// del bi-encoder ya es casi exhaustivo). Es CPU-bound y debe correr dentro de
/// `tokio::task::spawn_blocking`, igual que Whisper — nunca directo en una tarea async.
///
/// **Pesos en f32, no f16** (a diferencia del ejemplo de referencia, que usa f16 pensando en
/// GPU): este proyecto es CPU-only y la prioridad #1 es accuracy, no velocidad — no vale la pena
/// arriesgar precisión del cross-encoder para ahorrar la mitad de la RAM transitoria de un modelo
/// que de por sí no queda residente (ver más abajo).
///
/// Carga el modelo de forma transiente en cada llamada: pesos mmapeados vía
/// `VarBuilder::from_mmaped_safetensors` (el OS cachea las páginas entre llamadas, no se duplica
/// en RAM física cada vez) y se sueltan al retornar — nunca queda un handle persistente en
/// `AppState`, misma filosofía que `WhisperRunner`/Ollama (cargar, procesar, soltar).
///
/// **Primera ejecución**: descarga los pesos/tokenizer/config de Hugging Face Hub (~2.2GB en
/// f32) y los cachea localmente vía `hf-hub` (respeta `HF_HOME`, default
/// `~/.cache/huggingface/hub`). Es la única llamada de red de este módulo — documentada como
/// excepción deliberada igual que el primer `ollama pull` o el fallback a `ffmpeg`/`ffprobe` (ver
/// CLAUDE.local.md): corridas siguientes son 100% offline, el archivo ya está en disco.
fn rerank_blocking(query: &str, mut hits: Vec<ChunkHit>) -> anyhow::Result<Vec<ChunkHit>> {
    if hits.len() <= 1 {
        return Ok(hits);
    }

    let device = Device::Cpu;
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        RERANKER_MODEL_ID.to_string(),
        RepoType::Model,
        RERANKER_REVISION.to_string(),
    ));

    let tokenizer_path = repo.get("tokenizer.json")?;
    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        }))
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;

    // `unsafe`: `from_mmaped_safetensors` mapea el archivo directo a memoria sin validar que
    // nadie más lo modifique mientras está mapeado — mismo contrato que usa el ejemplo oficial
    // de candle, el archivo viene de la cache local de `hf-hub` y no se escribe concurrentemente.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = XLMRobertaForSequenceClassification::new(1, &config, vb)?;

    let pairs: Vec<(String, String)> = hits
        .iter()
        .map(|hit| (query.to_string(), hit.chunk.text.clone()))
        .collect();

    let encodings = tokenizer
        .encode_batch(pairs, true)
        .map_err(anyhow::Error::msg)?;

    let input_ids = encodings
        .iter()
        .map(|enc| Tensor::new(enc.get_ids(), &device))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let input_ids = Tensor::stack(&input_ids, 0)?;

    let attention_mask = encodings
        .iter()
        .map(|enc| Tensor::new(enc.get_attention_mask(), &device))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    let token_type_ids = Tensor::zeros(input_ids.dims(), input_ids.dtype(), &device)?;

    let logits = model.forward(&input_ids, &attention_mask, &token_type_ids)?;
    let scores = candle_nn::ops::sigmoid(&logits)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    for (hit, score) in hits.iter_mut().zip(scores) {
        // El score del cross-encoder reemplaza el coseno del bi-encoder que traía el hit — a
        // partir de acá `ChunkHit::score` significa "relevancia según el reranker", no
        // similitud vectorial (ver doc de `ChunkHit` en `retrieval.rs`).
        hit.score = Some(score);
    }
    hits.sort_by(|a, b| {
        b.score
            .unwrap_or(f32::MIN)
            .total_cmp(&a.score.unwrap_or(f32::MIN))
    });

    Ok(hits)
}

/// Wrapper async: mueve el trabajo CPU-bound de `rerank_blocking` a `spawn_blocking` (ver
/// CLAUDE.local.md: Whisper/cómputo pesado nunca en una tarea async directa).
///
/// **Degrada sin romper la query**: si el reranker falla por cualquier motivo (sin red en el
/// primer uso, modelo corrupto en cache, error del framework), se loguea el error y se devuelve
/// el orden original del bi-encoder en vez de propagar el fallo — la prioridad de este paso es
/// afinar el ranking, no bloquear una respuesta que igual tiene contexto razonable sin él (ver
/// CLAUDE.local.md: tolerancia a fallos).
pub async fn rerank(query: &str, hits: Vec<ChunkHit>) -> Vec<ChunkHit> {
    let fallback = hits.clone();
    let query = query.to_string();

    match tokio::task::spawn_blocking(move || rerank_blocking(&query, hits)).await {
        Ok(Ok(reranked)) => reranked,
        Ok(Err(e)) => {
            eprintln!("Reranker falló, se mantiene el orden del bi-encoder: {e}");
            fallback
        }
        Err(join_err) => {
            eprintln!(
                "Reranker: spawn_blocking falló ({join_err}), se mantiene el orden del bi-encoder"
            );
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::retrieval::ChunkPayload;

    fn hit(text: &str) -> ChunkHit {
        ChunkHit {
            chunk: ChunkPayload {
                audio_id: "test-reranker".to_string(),
                chunk_id: 0,
                start: 0.0,
                end: 30.0,
                text: text.to_string(),
                speaker: "unknown".to_string(),
                avg_logprob: -0.1,
            },
            // Score de bi-encoder arbitrario y deliberadamente "al revés" del orden esperado —
            // si el test pasara con este score sin que el reranker corriera, no probaría nada.
            score: Some(0.1),
        }
    }

    /// No depende de Qdrant/Ollama — `rerank_blocking` es una función pura sobre `ChunkHit` en
    /// memoria, se puede probar el cross-encoder aislado del resto de la infraestructura. Sí
    /// depende de red en la primera corrida (descarga `BAAI/bge-reranker-v2-m3` de Hugging Face
    /// Hub, ~2.2GB en f32) y de que el resultado quede cacheado localmente para corridas
    /// siguientes. Ignorado por defecto por el costo/tiempo de esa descarga + inferencia, no
    /// por depender de servicios externos vivos como el resto de los tests `#[ignore]`.
    ///
    /// Corrida manual: `cargo test rerank_reordena -- --ignored --nocapture`
    #[test]
    #[ignore = "descarga BAAI/bge-reranker-v2-m3 de Hugging Face Hub (~2.2GB) la primera vez"]
    fn rerank_reordena_hits_por_relevancia_real() {
        let hits = vec![
            hit("El clima en Madrid es soleado con veinticinco grados."),
            hit("El presupuesto del proyecto es de cinco mil dólares para el próximo trimestre."),
            hit("Mañana hay una reunión con el equipo de marketing a las diez."),
        ];

        let reranked = rerank_blocking("¿Cuál es el presupuesto del proyecto?", hits)
            .expect("el reranker falló contra el modelo real");

        assert_eq!(
            reranked[0].chunk.text,
            "El presupuesto del proyecto es de cinco mil dólares para el próximo trimestre.",
            "el chunk realmente relevante para la query debería quedar primero tras el rerank, \
             no el que traía el score de bi-encoder más alto"
        );
    }
}
