use std::collections::HashMap;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Frase de sesgo de idioma/registro pasada a `set_initial_prompt` en cada chunk — genérica (no
/// vocabulario de un dominio específico como construcción) para que sirva igual en audios futuros
/// de cualquier tema. Agregada 2026-07-31 como mitigación de alucinaciones tipo "(Portuguesa)"
/// (ver CLAUDE.local.md).
const INITIAL_PROMPT_ES: &str =
    "Transcripción de una reunión de trabajo en español, conversación natural.";

/// Resultado de transcribir un chunk: texto + métricas de calidad/confianza derivadas de los
/// segmentos y tokens que devuelve whisper-rs. No se persiste tal cual — `pipeline.rs` lo separa
/// en `TranscriptEntry` (payload de Qdrant, sin tocar) y `ChunkMetrics` (diagnóstico, nuevo).
pub(crate) struct ChunkTranscription {
    pub text: String,
    pub avg_logprob: f32,
    /// Máximo de `no_speech_probability()` entre los segmentos del chunk — un solo segmento con
    /// alta probabilidad de "no es habla" es la señal que importa, no el promedio.
    pub no_speech_prob: f32,
    /// Entropía de Shannon sobre la frecuencia de `token_id()` de todos los tokens del chunk —
    /// misma fórmula que usa whisper.cpp internamente para `entropy_thold` (ver
    /// `whisper.cpp:6584-6605` en el vendor de `whisper-rs-sys`), pero recalculada acá sobre el
    /// chunk ya finalizado (no sobre la ventana interna de 32 tokens de un intento de decode en
    /// curso, que whisper-rs no expone). Baja entropía = tokens repetitivos (señal de un loop de
    /// alucinación tipo "(Portuguesa) (Portuguesa)...").
    pub entropy: f32,
    /// Segmentos efectivamente leídos (algunos índices pueden devolver `None`, ver el loop).
    pub segment_count: usize,
}

/// Wrapper de whisper-rs. Mantiene un único `WhisperState` vivo durante todo el job (no uno por
/// chunk) para que `set_no_context(false)` realmente aporte continuidad de contexto entre los
/// chunks de una reunión larga, en vez de que cada chunk se transcriba en el vacío.
///
/// Siempre se invoca desde dentro de `tokio::task::spawn_blocking` (ver
/// `audio_pipeline::pipeline::run_pipeline` y `handlers::audio_handler`) porque `full()` es
/// trabajo de CPU síncrono y bloquearía el runtime async si corriera en `tokio::spawn` directo.
pub struct WhisperRunner {
    state: WhisperState,
}

impl WhisperRunner {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())?;
        let state = ctx.create_state()?;
        Ok(Self { state })
    }

    /// Config de `FullParams` fijada en `CLAUDE.local.md` — no modificar sin justificar el
    /// impacto en precisión.
    fn build_params() -> FullParams<'static, 'static> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("es"));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_no_context(false);
        params.set_entropy_thold(2.3);
        params.set_suppress_blank(true);
        // Suprime un conjunto fijo de tokens de vocabulario (símbolos/puntuación sueltos, entre
        // ellos literalmente "(" y ")" — ver whisper.cpp:6095-6100 vendorizado en
        // whisper-rs-sys). No es supresión de ruido de audio: es el fix quirúrgico contra el
        // loop de alucinación "(Portuguesa) (Portuguesa)..." documentado en CLAUDE.local.md
        // (2026-07-31) — bloquea el token que abre esas frases entre paréntesis.
        params.set_suppress_nst(true);
        // Sesga idioma/registro antes de cada chunk (ver CLAUDE.local.md). Nota: `set_initial_prompt`
        // filtra la CString internamente (`into_raw()` sin `Drop` en whisper-rs 0.16.0) — leak de
        // ~100 bytes por llamada, negligible frente al límite de 16GB en un job de horas.
        params.set_initial_prompt(INITIAL_PROMPT_ES);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.1);
        params.set_logprob_thold(-0.8);
        params.set_no_speech_thold(0.5);
        params.set_single_segment(false);
        params.set_token_timestamps(false);
        params.set_split_on_word(true);
        params.set_n_threads(8);
        params
    }

    /// Transcribe un chunk y devuelve texto + métricas de calidad (ver `ChunkTranscription`).
    /// `avg_logprob` es la media de `ln(token_probability())` sobre todos los tokens del chunk —
    /// confianza para el payload de Qdrant (ver CLAUDE.local.md, campo `avg_logprob`). Un chunk
    /// sin tokens (silencio puro) devuelve `avg_logprob: 0.0`, `entropy: 0.0`.
    pub(crate) fn transcribe_chunk(
        &mut self,
        samples: &[f32],
    ) -> anyhow::Result<ChunkTranscription> {
        let params = Self::build_params();
        self.state.full(params, samples)?;

        let mut text = String::new();
        let mut logprob_sum = 0.0f64;
        let mut token_count = 0u32;
        let mut no_speech_prob = 0.0f32;
        let mut segment_count = 0usize;
        let mut token_id_counts: HashMap<i32, u32> = HashMap::new();

        for i in 0..self.state.full_n_segments() {
            let Some(segment) = self.state.get_segment(i) else {
                continue;
            };
            segment_count += 1;
            text.push_str(&segment.to_str_lossy()?);
            no_speech_prob = no_speech_prob.max(segment.no_speech_probability());

            for j in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(j) else {
                    continue;
                };
                logprob_sum += (token.token_probability() as f64).ln();
                token_count += 1;
                *token_id_counts.entry(token.token_id()).or_insert(0) += 1;
            }
        }

        let avg_logprob = if token_count > 0 {
            (logprob_sum / token_count as f64) as f32
        } else {
            0.0
        };

        // Entropía de Shannon sobre la frecuencia de token_id — ver doc de `ChunkTranscription`.
        let entropy = if token_count > 0 {
            let total = token_count as f64;
            -token_id_counts
                .values()
                .map(|&count| {
                    let p = count as f64 / total;
                    p * p.ln()
                })
                .sum::<f64>() as f32
        } else {
            0.0
        };

        Ok(ChunkTranscription {
            text,
            avg_logprob,
            no_speech_prob,
            entropy,
            segment_count,
        })
    }
}
