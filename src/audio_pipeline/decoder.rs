use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::Context;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::Time;

use crate::audio_pipeline::models::AudioChunk;

/// Frecuencia de salida que espera Whisper (PCM mono).
const SAMPLE_RATE_OUT: u32 = 16_000;
/// 30s @ 16kHz — múltiplo exacto de 160 samples/hop (hop de 10ms de Whisper).
const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE_OUT as usize;
/// 2s de overlap — también múltiplo exacto del hop.
const OVERLAP_SAMPLES: usize = 2 * SAMPLE_RATE_OUT as usize;
const ADVANCE_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES;
/// Tamaño de bloque de entrada para el resampler (frames de audio mono antes de resamplear).
const RESAMPLER_CHUNK_SIZE: usize = 1024;

/// De dónde viene el audio decodificado. Symphonia es el camino principal (puro Rust); el
/// fallback a `ffmpeg` existe porque algunos contenedores mp3/mp4 reales de este proyecto usan
/// AMR-NB, un códec de voz que Symphonia no implementa (ver CLAUDE.local.md: "Códecs no
/// soportados por Symphonia — fallback a ffmpeg").
enum Source {
    Symphonia {
        format: Box<dyn FormatReader>,
        decoder: Box<dyn AudioDecoder>,
        track_id: u32,
    },
    Ffmpeg {
        child: Child,
        stdout: ChildStdout,
        channels: usize,
        rate: u32,
        /// Hasta 3 bytes sobrantes de una lectura anterior que todavía no forman un f32 completo.
        leftover: Vec<u8>,
    },
}

impl Drop for Source {
    /// Sin esto, un corte del pipeline que no llegue a EOF ni pase por un seek explícito (los
    /// únicos dos lugares que hoy llaman `child.wait()`/`kill()` — ej. Whisper falla un chunk y
    /// el `?` de `run_pipeline` propaga el error, o un panic) deja el proceso `ffmpeg` huérfano:
    /// dropear un `std::process::Child` en Rust nunca mata el proceso, solo cierra el handle.
    /// `kill()`/`wait()` sobre un proceso ya terminado (ej. el camino de EOF, que ya hizo su
    /// propio `wait()`) devuelven `Err`, ignorado a propósito — este `Drop` es una red de
    /// seguridad, no la única vía esperada de limpieza.
    fn drop(&mut self) {
        if let Source::Ffmpeg { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Resultado de decodificar un paso (un paquete de Symphonia, o un bloque leído del pipe de
/// ffmpeg): o bien produjo muestras interleaved con su spec, o el stream terminó.
enum PumpOutcome {
    Samples {
        interleaved: Vec<f32>,
        channels: usize,
        rate: u32,
    },
    Eof,
}

/// Decodifica un archivo mp3/mp4, se queda únicamente con el track de audio (nunca instancia un
/// decoder para un track de video de un mp4), resamplea a PCM mono 16kHz con un resampler sinc
/// anti-aliasing (rubato) y entrega chunks de 30s con overlap de 2s + crossfade (Hann) en los
/// bordes internos.
pub struct StreamingDecoder {
    audio_path: String,
    source: Source,
    input_rate: u32,
    resampler: Option<SincFixedIn<f32>>,
    /// Samples mono, sin resamplear, esperando a completar un bloque de tamaño
    /// `RESAMPLER_CHUNK_SIZE` para alimentar al resampler.
    input_accum: Vec<f32>,
    /// Samples mono a 16kHz ya resampleados, listos para recortar en chunks. `VecDeque` para
    /// poder drenar por el frente sin reallocar un `Vec` nuevo por chunk.
    output_buf: VecDeque<f32>,
    /// Posición absoluta (en samples @ 16kHz) del próximo chunk a devolver.
    next_start_sample: usize,
    chunk_index: usize,
    /// `true` una vez que la fuente llegó a EOF y ya se hizo el flush final del resampler.
    exhausted: bool,
}

fn sniff_hint(audio_path: &str) -> Hint {
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(audio_path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    hint
}

fn downmix_interleaved_to_mono(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        out.push(sum / channels as f32);
    }
}

fn build_resampler(input_rate: u32) -> anyhow::Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = SAMPLE_RATE_OUT as f64 / input_rate as f64;
    let resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, RESAMPLER_CHUNK_SIZE, 1)?;
    Ok(resampler)
}

/// Aplica una rampa Hann (raised-cosine) de `taper_len` samples al inicio y/o al final del
/// chunk. Nunca se aplica en el primer inicio ni en el último final del audio completo — solo
/// en las costuras internas entre chunks consecutivos, donde hay contenido real de ambos lados
/// con el que mezclar.
fn apply_edge_taper(samples: &mut [f32], taper_len: usize, fade_in: bool, fade_out: bool) {
    let n = samples.len();
    let taper_len = taper_len.min(n / 2);
    if taper_len == 0 {
        return;
    }
    if fade_in {
        for (i, sample) in samples.iter_mut().enumerate().take(taper_len) {
            let t = i as f32 / taper_len as f32;
            let gain = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            *sample *= gain;
        }
    }
    if fade_out {
        for (i, sample) in samples.iter_mut().rev().enumerate().take(taper_len) {
            let t = i as f32 / taper_len as f32;
            let gain = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            *sample *= gain;
        }
    }
}

/// (format reader, decoder de audio, track_id elegido, sample_rate declarado por el contenedor).
type SymphoniaSetup = (
    Box<dyn FormatReader>,
    Box<dyn AudioDecoder>,
    u32,
    Option<u32>,
);

/// Intenta abrir el archivo con Symphonia y armar un decoder de audio real. Devuelve `Err` si no
/// hay ningún track de audio reconocible, o si el códec del track (ej. AMR-NB) no tiene decoder
/// registrado — en ambos casos el llamador cae a `ffmpeg`.
fn try_symphonia(audio_path: &str) -> anyhow::Result<SymphoniaSetup> {
    let file = File::open(audio_path)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let hint = sniff_hint(audio_path);

    let format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.as_ref().is_some_and(|p| p.is_audio()))
        .ok_or_else(|| {
            anyhow::anyhow!("no se encontró ningún track de audio reconocido por Symphonia")
        })?;

    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .expect("ya filtrado por is_audio() arriba")
        .clone();

    let decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())?;

    Ok((format, decoder, track_id, audio_params.sample_rate))
}

/// Usa `ffprobe` para conocer sample rate/canales del primer stream de audio, y arranca
/// `ffmpeg` con ese archivo como input, emitiendo PCM f32le crudo (sin resamplear — el rate y
/// canales originales se preservan) por stdout. El resampleo a 16kHz sigue haciéndolo rubato,
/// igual que en el camino de Symphonia, para no duplicar esa lógica.
///
/// Requiere `ffmpeg`/`ffprobe` en el PATH. Ver CLAUDE.local.md: esto es una excepción
/// deliberada a "sin dependencias externas", necesaria porque Symphonia no decodifica AMR-NB.
fn build_ffmpeg_source(audio_path: &str, seek_seconds: Option<f32>) -> anyhow::Result<Source> {
    let probe_output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a:0",
            audio_path,
        ])
        .output()
        .context("no se pudo ejecutar ffprobe (¿está instalado y en el PATH?)")?;

    if !probe_output.status.success() {
        anyhow::bail!(
            "ffprobe falló para {audio_path}: {}",
            String::from_utf8_lossy(&probe_output.stderr)
        );
    }

    let probe_json: serde_json::Value = serde_json::from_slice(&probe_output.stdout)?;
    let stream = probe_json["streams"].get(0).ok_or_else(|| {
        anyhow::anyhow!("ffprobe no encontró ningún stream de audio en {audio_path}")
    })?;

    let rate: u32 = stream["sample_rate"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("ffprobe no reportó sample_rate"))?
        .parse()?;
    let channels: usize = stream["channels"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("ffprobe no reportó channels"))?
        as usize;

    let mut args: Vec<String> = vec!["-v".into(), "error".into()];
    if let Some(seconds) = seek_seconds {
        // Seek de input (antes de -i) — rápido, salta al keyframe/posición más cercana.
        args.push("-ss".into());
        args.push(format!("{seconds}"));
    }
    args.extend([
        "-i".into(),
        audio_path.into(),
        "-f".into(),
        "f32le".into(),
        "-acodec".into(),
        "pcm_f32le".into(),
        "-".into(),
    ]);

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .context("no se pudo ejecutar ffmpeg (¿está instalado y en el PATH?)")?;

    let stdout = child.stdout.take().expect("stdout configurado como piped");

    Ok(Source::Ffmpeg {
        child,
        stdout,
        channels,
        rate,
        leftover: Vec::new(),
    })
}

impl StreamingDecoder {
    pub fn new(audio_path: &str) -> anyhow::Result<Self> {
        let (source, declared_rate) = match try_symphonia(audio_path) {
            Ok((format, decoder, track_id, declared_rate)) => (
                Source::Symphonia {
                    format,
                    decoder,
                    track_id,
                },
                declared_rate,
            ),
            Err(symphonia_err) => {
                tracing::warn!(
                    "Symphonia no pudo decodificar {audio_path} ({symphonia_err}); \
                     usando ffmpeg como fallback (ver CLAUDE.local.md: códecs no soportados, \
                     ej. AMR-NB)."
                );
                let source = build_ffmpeg_source(audio_path, None)?;
                let rate = match &source {
                    Source::Ffmpeg { rate, .. } => Some(*rate),
                    Source::Symphonia { .. } => None,
                };
                (source, rate)
            }
        };

        Ok(Self {
            audio_path: audio_path.to_string(),
            source,
            input_rate: declared_rate.unwrap_or(SAMPLE_RATE_OUT),
            resampler: None,
            input_accum: Vec::with_capacity(RESAMPLER_CHUNK_SIZE * 2),
            output_buf: VecDeque::with_capacity(CHUNK_SAMPLES * 2),
            next_start_sample: 0,
            chunk_index: 0,
            exhausted: false,
        })
    }

    /// Reposiciona el decoder al segundo indicado (para retomar desde un checkpoint). Un salto a
    /// 0.0 en un decoder recién abierto es un no-op seguro (`resume_from_chunk` se ignora en ese
    /// caso: un job nuevo siempre empieza en el chunk 0, el default del constructor).
    ///
    /// `resume_from_chunk` es el índice que debe tener el próximo chunk emitido tras el seek —
    /// normalmente `checkpoint.last_chunk + 1`, ya que `last_chunk` es el último chunk que
    /// terminó de procesarse y persistirse antes del corte.
    pub fn seek_seconds(&mut self, seconds: f32, resume_from_chunk: usize) -> anyhow::Result<()> {
        if seconds <= 0.0 {
            return Ok(());
        }

        match &mut self.source {
            Source::Symphonia {
                format,
                decoder,
                track_id,
            } => {
                let time = Time::try_from_secs_f64(seconds as f64)
                    .ok_or_else(|| anyhow::anyhow!("segundo de seek inválido: {seconds}"))?;

                format.seek(
                    SeekMode::Accurate,
                    SeekTo::Time {
                        time,
                        track_id: Some(*track_id),
                    },
                )?;
                decoder.reset();
            }
            Source::Ffmpeg { child, .. } => {
                // No se puede seekear un proceso ffmpeg ya corriendo: se mata y se relanza con
                // `-ss` (seek de input, rápido) apuntando al segundo pedido.
                let _ = child.kill();
                let _ = child.wait();
                self.source = build_ffmpeg_source(&self.audio_path, Some(seconds))?;
            }
        }

        self.input_accum.clear();
        self.output_buf.clear();
        self.resampler = Some(build_resampler(self.input_rate)?);
        self.next_start_sample = (seconds as f64 * SAMPLE_RATE_OUT as f64).round() as usize;
        self.chunk_index = resume_from_chunk;
        self.exhausted = false;

        Ok(())
    }

    fn feed_resampler(&mut self) -> anyhow::Result<()> {
        let resampler = match &mut self.resampler {
            Some(r) => r,
            None => return Ok(()),
        };
        while self.input_accum.len() >= RESAMPLER_CHUNK_SIZE {
            let chunk: Vec<f32> = self.input_accum.drain(..RESAMPLER_CHUNK_SIZE).collect();
            let resampled = resampler.process(&[chunk], None)?;
            self.output_buf.extend(resampled[0].iter().copied());
        }
        Ok(())
    }

    fn flush_at_eof(&mut self) -> anyhow::Result<()> {
        let resampler = match &mut self.resampler {
            Some(r) => r,
            None => return Ok(()),
        };

        if !self.input_accum.is_empty() {
            let remaining: Vec<f32> = std::mem::take(&mut self.input_accum);
            let resampled = resampler.process_partial(Some(&[remaining]), None)?;
            self.output_buf.extend(resampled[0].iter().copied());
        }

        // Empuja cualquier sample retrasado que siga adentro del resampler.
        let tail = resampler.process_partial::<Vec<f32>>(None, None)?;
        self.output_buf.extend(tail[0].iter().copied());

        Ok(())
    }

    /// Decodifica/lee un paso de la fuente activa (Symphonia o ffmpeg) sin tocar el resto del
    /// estado — deja que `pump` decida qué hacer con el resultado, para no mezclar el borrow
    /// mutable de `self.source` con el resto de los campos.
    fn read_one(&mut self) -> anyhow::Result<PumpOutcome> {
        match &mut self.source {
            Source::Symphonia {
                format,
                decoder,
                track_id,
            } => loop {
                match format.next_packet() {
                    Ok(Some(packet)) => {
                        if packet.track_id != *track_id {
                            continue;
                        }

                        match decoder.decode(&packet) {
                            Ok(audio_buf) => {
                                let channels = audio_buf.spec().channels().count();
                                let rate = audio_buf.spec().rate();
                                let mut interleaved: Vec<f32> = Vec::new();
                                audio_buf.copy_to_vec_interleaved(&mut interleaved);
                                return Ok(PumpOutcome::Samples {
                                    interleaved,
                                    channels,
                                    rate,
                                });
                            }
                            Err(SymphoniaError::DecodeError(_)) => {
                                // Paquete corrupto/no decodificable — se descarta y se sigue,
                                // tal como recomienda la documentación de symphonia.
                                continue;
                            }
                            Err(e) => return Err(e.into()),
                        }
                    }
                    Ok(None) => return Ok(PumpOutcome::Eof),
                    Err(e) => return Err(e.into()),
                }
            },
            Source::Ffmpeg {
                stdout,
                channels,
                rate,
                leftover,
                ..
            } => {
                let mut buf = [0u8; 65_536];
                let n = stdout.read(&mut buf)?;
                if n == 0 {
                    return Ok(PumpOutcome::Eof);
                }

                leftover.extend_from_slice(&buf[..n]);
                let usable = leftover.len() - (leftover.len() % 4);
                let interleaved: Vec<f32> = leftover[..usable]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                leftover.drain(..usable);

                Ok(PumpOutcome::Samples {
                    interleaved,
                    channels: *channels,
                    rate: *rate,
                })
            }
        }
    }

    /// Decodifica y resamplea hasta un paso de audio (o procesa el EOF). No garantiza que
    /// `output_buf` tenga un chunk completo — el llamador reintenta hasta tener suficiente.
    fn pump(&mut self) -> anyhow::Result<()> {
        match self.read_one()? {
            PumpOutcome::Samples {
                interleaved,
                channels,
                rate,
            } => {
                if self.resampler.is_none() {
                    self.input_rate = rate;
                    self.resampler = Some(build_resampler(self.input_rate)?);
                }

                downmix_interleaved_to_mono(&interleaved, channels, &mut self.input_accum);
                self.feed_resampler()?;
                Ok(())
            }
            PumpOutcome::Eof => {
                self.flush_at_eof()?;
                self.exhausted = true;
                if let Source::Ffmpeg { child, .. } = &mut self.source {
                    let _ = child.wait();
                }
                Ok(())
            }
        }
    }

    pub fn next_chunk(&mut self) -> anyhow::Result<Option<AudioChunk>> {
        while !self.exhausted && self.output_buf.len() < CHUNK_SAMPLES {
            self.pump()?;
        }

        if self.output_buf.is_empty() {
            return Ok(None);
        }

        let available = self.output_buf.len();
        let take = available.min(CHUNK_SAMPLES);
        let is_first_chunk = self.chunk_index == 0;
        let is_last_chunk = self.exhausted && take == available;

        let mut samples: Vec<f32> = self.output_buf.iter().take(take).copied().collect();
        apply_edge_taper(
            &mut samples,
            OVERLAP_SAMPLES,
            !is_first_chunk,
            !is_last_chunk,
        );

        let start_sample = self.next_start_sample;
        let end_sample = start_sample + take;

        let drop_n = if is_last_chunk {
            take
        } else {
            ADVANCE_SAMPLES.min(take)
        };
        for _ in 0..drop_n {
            self.output_buf.pop_front();
        }
        self.next_start_sample += drop_n;

        let chunk = AudioChunk {
            index: self.chunk_index,
            start_sec: start_sample as f32 / SAMPLE_RATE_OUT as f32,
            end_sec: end_sample as f32 / SAMPLE_RATE_OUT as f32,
            samples,
        };
        self.chunk_index += 1;

        Ok(Some(chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Escribe un WAV PCM16 mono sintético (tono senoidal, sin silencios) — evita depender de
    /// `sample_Media/` (AMR-NB real, requiere el fallback a ffmpeg) para poder probar el
    /// comportamiento de resume del decoder en aislamiento, sin Whisper ni audio real.
    fn write_test_wav(path: &std::path::Path, sample_rate: u32, seconds: f32) {
        let num_samples = (sample_rate as f32 * seconds) as u32;
        let data_size = num_samples * 2; // 16 bits = 2 bytes/sample, mono
        let byte_rate = sample_rate * 2;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_size).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * 440.0 * std::f32::consts::TAU).sin() * i16::MAX as f32 * 0.5;
            buf.extend_from_slice(&(sample as i16).to_le_bytes());
        }

        std::fs::write(path, buf).expect("no se pudo escribir el wav de prueba");
    }

    fn test_wav_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("decoder_resume_test_{name}.wav"))
    }

    #[test]
    fn primer_chunk_de_un_decoder_nuevo_arranca_en_index_0() {
        let path = test_wav_path("nuevo");
        write_test_wav(&path, 16_000, 2.0);

        let mut decoder = StreamingDecoder::new(path.to_str().unwrap()).unwrap();
        let chunk = decoder
            .next_chunk()
            .unwrap()
            .expect("debería haber un chunk");

        assert_eq!(chunk.index, 0);
        assert!(chunk.start_sec.abs() < 0.01);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn seek_seconds_retoma_la_numeracion_de_chunk_desde_resume_from_chunk() {
        let path = test_wav_path("resume");
        write_test_wav(&path, 16_000, 3.0);

        let mut decoder = StreamingDecoder::new(path.to_str().unwrap()).unwrap();
        decoder.seek_seconds(1.0, 7).unwrap();
        let chunk = decoder
            .next_chunk()
            .unwrap()
            .expect("debería haber un chunk tras el seek");

        assert_eq!(
            chunk.index, 7,
            "chunk_index debe retomar en resume_from_chunk, no en 0"
        );
        assert!(
            (chunk.start_sec - 1.0).abs() < 0.01,
            "start_sec debe derivarse del segundo de seek, no del índice: {}",
            chunk.start_sec
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn seek_a_cero_en_decoder_nuevo_es_no_op_e_ignora_resume_from_chunk() {
        let path = test_wav_path("noop");
        write_test_wav(&path, 16_000, 2.0);

        let mut decoder = StreamingDecoder::new(path.to_str().unwrap()).unwrap();
        // seconds <= 0.0 debe ser un no-op seguro — el resume_from_chunk no debe aplicarse,
        // un job nuevo siempre arranca en 0 (el default del constructor).
        decoder.seek_seconds(0.0, 7).unwrap();
        let chunk = decoder
            .next_chunk()
            .unwrap()
            .expect("debería haber un chunk");

        assert_eq!(chunk.index, 0);

        let _ = std::fs::remove_file(&path);
    }
}
