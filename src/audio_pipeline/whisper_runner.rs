use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

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

    pub fn transcribe_chunk(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let params = Self::build_params();
        self.state.full(params, samples)?;

        let mut text = String::new();
        for i in 0..self.state.full_n_segments() {
            let Some(segment) = self.state.get_segment(i) else {
                continue;
            };
            text.push_str(&segment.to_str_lossy()?);
        }
        Ok(text)
    }
}
