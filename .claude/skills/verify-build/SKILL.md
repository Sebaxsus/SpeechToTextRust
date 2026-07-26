---
name: verify-build
description: Run cargo fmt --check, cargo clippy, and cargo test against this Rust project and report any failures. Use before considering a change done, or when asked to verify/check the build, lint, or tests.
---

Run these three commands from the project root, in order, and stop at the first one that fails:

1. `cargo fmt --check` — if it fails, run `cargo fmt` to apply the fix, then re-run `--check` to confirm.
2. `cargo clippy --all-targets` — the project has `[lints.clippy] all = "warn"` in `Cargo.toml`. Report every warning with its file:line; do not silently ignore any.
3. `cargo test` — report which tests failed and their assertion output verbatim.

Note: the crate depends on `whisper-rs`, which compiles a C++ library (whisper.cpp) on first build — expect the first run after a `Cargo.lock`/toolchain change to take noticeably longer than incremental runs.

When reporting results, distinguish between:
- **Pre-existing `todo!()` panics** in stubbed modules (`decoder.rs`, `checkpoint.rs`, `whisper_runner.rs`, `job.rs`, `JsonlWriter::append`) — these are expected until that logic is implemented; don't report them as new regressions unless the test specifically exercises that path and previously passed.
- **Real regressions** — anything that changed status from passing to failing, or new clippy/fmt violations introduced by the change under review.

If all three commands pass cleanly, say so plainly instead of padding the report.
