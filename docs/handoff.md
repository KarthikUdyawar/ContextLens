# Handoff

## Where things stand

Sprint X1 planned session 3, coded session 4 (R1–R5, R7–R9), closed session 5 (R6, R10–R12) — all 4 `docker-compose.yml` services confirmed running together. **R8/R9's live-acquisition gap closed session 6**: `download_hf.py`/`download_kaggle.py` actually run against real HF/Kaggle/MinIO for the first time — all 6 confirmed sources landed, verified via `mc ls -r local/raw-data` (~557MB total). X1 acquisition is now fully live-verified, not just logic-reviewed.

**Don't re-derive — read the docs, current as of this session:**
- `TODO.md` — all R1–R12 checked, X1 marked closed, R6's note reflects the final shipped config (not the first two attempts).
- `DECISIONS.md` — #28 rewritten to record the *final* vLLM config and all three problems hit getting there (see below). Superseded content from mid-session isn't preserved elsewhere — this entry is the only record.
- `ROADMAP.md`/`PRD.md` — unchanged since session 3, still current. Next sprint (Postgres ingestion + vLLM/DSPy labelling) not started.

## What actually shipped this session (verified working, not just diffed)

- `src/ingest/download_hf.py` / `download_kaggle.py` — `if __name__ == "__main__":` entrypoints (R10). Not smoke-tested this session (needs live Kaggle/MinIO creds) — logic-level tests from session 4 still the only coverage.
  **Session 6 update:** now smoke-tested live. `download_hf.py` needed one fix — `sentiment140` slug → `stanfordnlp/sentiment140` + `revision="refs/convert/parquet"` on `load_dataset()` (DECISIONS.md #30). `download_kaggle.py` ran clean, no changes.
- `src/app/main.py` — `configure_logging()` wired at module top (R11). **Known gap, not fixed:** runs after the `classifier` import, so the first model load isn't logged. Flagged twice now (session 5 diff + this handoff) — Karthik hasn't asked for the reorder, leaving as-is.
- `uv.lock` — committed, 205 packages.
- `docker-compose.yml` `vllm` service — final working state, reached after 3 real failures in sequence, each diagnosed from live logs, not guessed:
  1. **No GPU passthrough** — `deploy.resources.reservations.devices` (nvidia) was missing entirely; container couldn't see the GPU at all.
  2. **CUDA version mismatch** — `vllm/vllm-openai:latest` needed CUDA≥13, Karthik's driver (566.07) only supports 12.7. Pinned to `v0.6.3.post1`.
  3. **DNS resolution failure** — container couldn't resolve `huggingface.co` on WSL2/Docker Desktop's default resolver (`api-1` resolved fine in the same run — vllm-specific). Fixed with explicit `dns: [8.8.8.8, 1.1.1.1]`.
  4. **Gated model 403** — original pick `meta-llama/Llama-3.2-1B-Instruct` needs Meta license acceptance; not approved in time. Swapped to ungated `Qwen/Qwen2.5-1.5B-Instruct`, same size class.
  5. **OOM on cache blocks** — `--gpu-memory-utilization 0.7` left no room for KV cache after ~3GB weights loaded on a 4GB card. Raised to `0.9`, added `--max-num-seqs 4` (down from vLLM's default 256 — this is a single-user dev box).
  - End state confirmed via `curl http://localhost:8001/v1/models` returning a clean model list.
- `.env.example` — `HUGGING_FACE_HUB_TOKEN` added, with a comment pointing at the license-acceptance page.
- `TODO.md`, `DECISIONS.md` — updated and confirmed applied by Karthik (pasted back this turn).

## Security note from this session

Karthik pasted a real Kaggle key and HF token into chat in plaintext at one point. Flagged in-conversation with instructions to rotate both. **Not confirmed whether rotation actually happened** — worth a quick check next session if credential-touching work comes up (don't assume the pasted values are still valid, and don't assume they were rotated either).

## Immediate next steps

- Confirm the Kaggle/HF credential rotation above actually happened, if it matters for next session's work.
- ~~`src/ingest/download_hf.py`/`download_kaggle.py` smoke test~~ — **done session 6**, see above.
- `main.py`'s logging-order gap (classifier loads before `configure_logging()` runs) — still open, still nobody's asked for it, still flagged.
- Next sprint proper — Postgres ingestion of MinIO-landed files, then vLLM/DSPy labelling — is unstarted. `ROADMAP.md` sequencing (item 4 before item 5) still applies; don't start labelling-program design before ingestion lands.
- `--max-num-seqs 4` and `--gpu-memory-utilization 0.9` are tuned for a single local dev box on a 4GB card — if the labelling pass later needs real throughput across ~1M rows, these will need revisiting (this session didn't address throughput, only "does it start and serve").
- `PRD.md`/`ROADMAP.md` source lists still say `sentiment140` — actual working slug is `stanfordnlp/sentiment140`. Minor doc drift, unaddressed.

## Session 6 addendum

- All 6 X1 sources live-verified in MinIO (see "Where things stand"). `stanfordnlp/sentiment140` fix applied and confirmed — see DECISIONS.md #30.
- Pylance `reportMissingImports` on `datasets` in VSCode — not a code bug, interpreter mismatch (`uv run` resolves ad hoc, Pylance needs `.venv` selected explicitly). Fixed by `uv sync --group ingest` + "Python: Select Interpreter" → `.venv/bin/python`. Not logged in DECISIONS.md — pure local dev-env config, no code/infra change.

# Conventions confirmed working this session

- Diagnosing from real command output only — every fix this session (GPU passthrough, CUDA pin, DNS, gated-model swap, OOM tuning) came from reading actual logs/`docker stats`/`nvidia-smi` output Karthik pasted back, never guessed ahead of evidence.
- Flag-don't-silently-fix holds even under pressure to just get something running (e.g. the Llama→Qwen swap was proposed, not applied, until Karthik said go).
- Plaintext secrets pasted into chat get flagged immediately and specifically (rotate at X, rotate at Y), not just a generic "be careful" note.

## Skills active / relevant to next session

`caveman ultra`, `ponytail`, `clean-code`, `tdd` — active throughout session 5, presumed to persist. `handoff` used to produce this doc.
