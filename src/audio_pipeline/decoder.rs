use crate::audio_pipeline::models::AudioChunk;

// TODO: implement streaming decode via Symphonia (see CLAUDE.local.md: Context Aware Chunking,
// mandatory anti-aliased resampling to 16kHz mono for Whisper).
pub struct StreamingDecoder;

impl StreamingDecoder {
    pub fn new(_audio_path: &str) -> anyhow::Result<Self> {
        todo!("StreamingDecoder::new is not implemented yet")
    }

    pub fn seek_seconds(&mut self, _seconds: f32) -> anyhow::Result<()> {
        todo!("StreamingDecoder::seek_seconds is not implemented yet")
    }

    pub fn next_chunk(&mut self) -> anyhow::Result<Option<AudioChunk>> {
        todo!("StreamingDecoder::next_chunk is not implemented yet")
    }
}
