use std::collections::HashMap;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::config::WhisperConfig;

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
    /// Leído una sola vez del entorno al construir el runner (ver `config::Config`), nunca
    /// releído por chunk — `build_params` se llama una vez por cada `transcribe_chunk`, así que
    /// cachear esto acá evita volver a parsear variables de entorno en el loop caliente del
    /// pipeline (ver `audio_pipeline::pipeline::run_pipeline`).
    tuning: WhisperConfig,
    /// `cargo run -- --greedy` (ver `WHISPER_TUNING_LOG.md`, 2026-08-07) — mismo patrón que
    /// `--log` en `main.rs`: flag de CLI leída directamente vía `std::env::args()`, sin pasar por
    /// `Config`/`.env` (ese módulo está documentado como "cargado desde variables de entorno", no
    /// de argv). Leída una sola vez por job (acá, no por chunk) y cacheada, igual que `tuning`.
    ///
    /// Default `false` = `BeamSearch` (`beam_size=5`): la verificación empírica de larga duración
    /// del 2026-08-07 (audio real de 49 min, HWiNFO en ambos modos) mostró que el costo de
    /// CPU/térmico sostenido de BeamSearch es prácticamente ruido de medición frente a Greedy en
    /// este hardware — el argumento térmico que antes lo mantenía opt-in ya no aplica — y la
    /// comparación de precisión (`docs/Benchmark.md`) mostró señales a favor de BeamSearch. Con
    /// `true` (`--greedy`) se usa `Greedy { best_of: tuning.greedy_best_of }`, más rápido pero sin
    /// esa ganancia de precisión.
    use_greedy: bool,
}

impl WhisperRunner {
    pub fn new(model_path: &str, tuning: WhisperConfig) -> anyhow::Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())?;
        let state = ctx.create_state()?;
        let use_greedy = std::env::args().any(|arg| arg == "--greedy");
        if use_greedy {
            tracing::info!(
                target: "lifecycle",
                "cargo run -- --greedy activo: usando Greedy (best_of={}) en vez de BeamSearch \
                 (default) — más rápido, pero ver WHISPER_TUNING_LOG.md (2026-08-07) para la \
                 evidencia de precisión a favor de BeamSearch.",
                tuning.greedy_best_of
            );
        }
        Ok(Self {
            state,
            tuning,
            use_greedy,
        })
    }

    /// Config de `FullParams` — valores por defecto fijados/documentados en `CLAUDE.local.md`,
    /// overridable vía `.env` (ver `config::WhisperConfig`) — no modificar sin justificar el
    /// impacto en precisión.
    ///
    /// `force_fresh_context`: override puntual de `no_context` para un solo chunk, usado por el
    /// loop-breaker de `audio_pipeline::pipeline::run_pipeline` (ver `WHISPER_TUNING_LOG.md`,
    /// hallazgo 2026-08-06) — cuando el pipeline detecta 3+ chunks consecutivos con texto
    /// idéntico (loop de alucinación tipo "Este es el canal de..."), fuerza `no_context=true` en
    /// el chunk siguiente para que Whisper no reciba el texto alucinado como "initial prompt" del
    /// propio `WhisperState` persistente y pueda decodificar ese chunk sin ese sesgo. Normalmente
    /// es `false` (comportamiento sin cambios: `no_context=false` de siempre).
    ///
    /// `use_greedy`: ver doc de `WhisperRunner::use_greedy` — con `false` (default) usa
    /// `BeamSearch { beam_size: 5, patience: -1.0 }` (defaults documentados por whisper-rs/
    /// whisper.cpp — `patience` no está implementado aún en whisper.cpp); con `true`
    /// (`cargo run -- --greedy`) usa `Greedy { best_of: tuning.greedy_best_of }`.
    fn build_params(
        tuning: &WhisperConfig,
        force_fresh_context: bool,
        use_greedy: bool,
    ) -> FullParams<'static, 'static> {
        let sampling_strategy = if use_greedy {
            SamplingStrategy::Greedy {
                best_of: tuning.greedy_best_of,
            }
        } else {
            SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            }
        };
        let mut params = FullParams::new(sampling_strategy);
        params.set_language(Some("es"));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_no_context(force_fresh_context);
        params.set_entropy_thold(tuning.entropy_thold);
        params.set_suppress_blank(tuning.suppress_blank);
        // Suprime un conjunto fijo de tokens de vocabulario (símbolos/puntuación sueltos, entre
        // ellos literalmente "(" y ")" — ver whisper.cpp:6095-6100 vendorizado en
        // whisper-rs-sys). No es supresión de ruido de audio: es el fix quirúrgico contra el
        // loop de alucinación "(Portuguesa) (Portuguesa)..." documentado en CLAUDE.local.md
        // (2026-07-31) — bloquea el token que abre esas frases entre paréntesis.
        params.set_suppress_nst(tuning.suppress_nst);
        // Sesga idioma/registro antes de cada chunk (ver CLAUDE.local.md). Nota: `set_initial_prompt`
        // filtra la CString internamente (`into_raw()` sin `Drop` en whisper-rs 0.16.0) — leak de
        // ~100 bytes por llamada, negligible frente al límite de 16GB en un job de horas.
        params.set_initial_prompt(&tuning.initial_prompt);
        params.set_temperature(tuning.temperature);
        params.set_temperature_inc(tuning.temperature_inc);
        params.set_logprob_thold(tuning.logprob_thold);
        params.set_no_speech_thold(tuning.no_speech_thold);
        params.set_single_segment(tuning.single_segment);
        params.set_token_timestamps(tuning.token_timestamps);
        params.set_split_on_word(tuning.split_on_word);
        params.set_n_threads(tuning.n_threads);
        params
    }

    /// Transcribe un chunk y devuelve texto + métricas de calidad (ver `ChunkTranscription`).
    /// `avg_logprob` es la media de `ln(token_probability())` sobre todos los tokens del chunk —
    /// confianza para el payload de Qdrant (ver CLAUDE.local.md, campo `avg_logprob`). Un chunk
    /// sin tokens (silencio puro) devuelve `avg_logprob: 0.0`, `entropy: 0.0`.
    ///
    /// `force_fresh_context`: ver doc de `build_params` — pasado tal cual desde el loop-breaker
    /// de `run_pipeline`, no se cachea acá.
    pub(crate) fn transcribe_chunk(
        &mut self,
        samples: &[f32],
        force_fresh_context: bool,
    ) -> anyhow::Result<ChunkTranscription> {
        let params = Self::build_params(&self.tuning, force_fresh_context, self.use_greedy);
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
