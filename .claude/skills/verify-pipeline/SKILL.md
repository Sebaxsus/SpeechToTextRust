---
name: verify-pipeline
description: Review new or changed Rust code in this audio transcription/RAG pipeline against the project's hard architectural constraints (RAM ceiling, streaming, model lifecycle, tdrz turn dedupe, resampling). Use when asked to review, verify, or check pipeline code before considering it done.
---

Read `CLAUDE.local.md` at the project root first — it is the source of truth for the current rules, and they get tuned over time. Do not rely on a cached memory of its contents; re-read it every time this skill runs.

Check the code under review against these categories (skip any category `CLAUDE.local.md` no longer documents):

1. **RAM ceiling (16GB hard limit)** — no full audio file loaded into memory, no giant buffers, no accumulating full corpus/transcript in RAM. Uploads and reads must stream.
2. **Model lifecycle** — Whisper and Ollama must not stay resident permanently; they load, run, and get explicitly released. Only one heavy transcription should run concurrently (Semaphore/bounded channel, not unbounded spawn).
3. **Blocking discipline** — any Whisper (or other CPU-bound) call must run in `tokio::task::spawn_blocking`, never bare `tokio::spawn`.
4. **tdrz turn dedupe** — if the code touches `next_segment_speaker_turn()` or speaker-turn counting, verify it treats the chunk-overlap window as a single decision zone: a persisted last-turn timestamp, a minimum gap threshold, and no per-chunk reset of the speaker counter.
5. **Resampling** — any sample-rate conversion must use a sinc/anti-aliasing resampler (e.g. `rubato`), never linear interpolation or naive decimation.
6. **Persistence format** — results persist incrementally as JSONL, not accumulated in memory then written once at the end.

For each violation found: cite the exact file and line, quote the offending code, and explain concretely what breaks — which constraint, and the realistic failure scenario (e.g. "5-hour file fully buffered → OOM on a 16GB machine"). Do not flag stylistic issues unrelated to these categories — this skill is a constraints gate, not a general code review.

If nothing violates the checklist, say so plainly instead of inventing minor nitpicks.
