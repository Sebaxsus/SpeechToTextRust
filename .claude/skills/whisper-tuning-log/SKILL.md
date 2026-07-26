---
name: whisper-tuning-log
description: Keep a dated changelog of Whisper FullParams tuning changes (thresholds, temperature, tdrz, etc.) with the rationale behind each change. Use whenever Whisper params are modified, or when asked to log/record a tuning change.
---

When Whisper `FullParams` values change (in code, or in a discussion with the user about tuning), append an entry to `WHISPER_TUNING_LOG.md` at the project root (create it if it doesn't exist yet, with a one-line header explaining its purpose).

Each entry:

```
## YYYY-MM-DD
- Changed: `param_name` old_value -> new_value
- Why: <the accuracy/stability problem this addresses, or the empirical test result that motivated it>
- Tested on: <what kind of audio, if mentioned — e.g. "phone meeting recording with cross-talk">
```

Ask the user for the "why" if it isn't already stated in the conversation — do not invent a rationale. Keep entries terse; this is a changelog, not a design doc. Never delete or rewrite past entries, only append.

If asked to review tuning history, read this file rather than trying to infer past decisions from git log or code comments.
