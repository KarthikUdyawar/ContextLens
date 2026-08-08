# TODO

## Sprint X1 (current) — UV migration + raw data to MinIO

### Tooling — UV migration

- [x] **R1** — Write `pyproject.toml`: runtime deps under `[project.dependencies]` (from `requirements.txt`); `dev` dependency group (from `dev-requirements.txt`); `train` dependency group (training-only deps split out of the current flat dev list — torch training extras, TextBlob, etc.)
  — Done (session 4). Also added, beyond original scope: `ingest` group (minio/kaggle/datasets, needed by R8–R9), `[tool.ruff]`/`[tool.bandit]`/`[tool.mypy]`/`[tool.pytest.ini_options]`/`[tool.coverage]` sections.
- [x] **R2** — Remove `setup.py`, `requirements.txt`, `dev-requirements.txt` — no pip-compat retained (DECISIONS.md #20)
  — Done (session 4).
- [x] **R3** — Bump Python floor 3.10 → 3.12: `pyproject.toml`, `Dockerfile` base image, any CI config referencing 3.10
  — Done (session 4). No CI config existed to update (INFRA.md: none in v1.0.0, tests.yml still deferred).

### Infra — docker-compose.yml

- [x] **R4** — Add `minio` service: image `minio/minio`, API + console ports exposed, default bucket `raw-data` created on startup (DECISIONS.md #22)
  — Done (session 4). Bucket creation via a separate `minio-init` sidecar (mc client), not a startup flag on `minio` itself.
- [x] **R5** — Add `postgres` service: default image, db name `contextlens` (DECISIONS.md #23), default port 5432 — table schema not built yet (next sprint)
  — Done (session 4). Schema still not built, as scoped.
- [x] **R6** — Add `vllm` service: image/model **still open**, see `ROADMAP.md` "Not yet decided"  — placeholder service until model is picked
  — Done (session 5), verified running end-to-end (`curl /v1/models` returns clean). Final model: `Qwen/Qwen2.5-1.5B-Instruct` (DECISIONS.md #28, superseded from initial Llama pick — gated/unapproved). `docker-compose.yml` `vllm` service: pinned image `v0.6.3.post1` (not `:latest` — CUDA version mismatch), `dns:` override for HF resolution, `--gpu-memory-utilization 0.9 --max-num-seqs 4` (tuned after an OOM at 0.7/default). `.env.example` now includes `HUGGING_FACE_HUB_TOKEN`.


### Credentials

- [x] **R7** — Kaggle API credentials: env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`, documented in `.env.example`, never committed
  — Done (session 4). `.env.example` also carries `MINIO_ENDPOINT`/`MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`/`POSTGRES_USER`/`POSTGRES_PASSWORD`, matching compose defaults.

### Download scripts

- [x] **R8** — HF download script: `sentiment140`, `cardiffnlp/tweet_eval` (sentiment config), `bdstar/twitter-sentiment-analysis`, `bdstar/Tweets-Sentiment-Analysis` → MinIO `raw-data/hf/<dataset>/`, raw, unmodified, labels ignored
  — Logic done (session 4), `src/ingest/download_hf.py` (`HfDownloader`), TDD'd incl. partial-failure handling. **Not yet runnable as a script — no `if __name__ == "__main__":` (see R10 below).**
- [x] **R9** — Kaggle download script: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, `tariqsays/sentiment-dataset-with-1-million-tweets` → MinIO `raw-data/kaggle/<dataset>/`, raw, unmodified, labels ignored
  — Logic done (session 4), `src/ingest/download_kaggle.py` (`KaggleDownloader`), same shape/tests. Same R10 gap.

### Follow-ups opened session 4 (small, not originally scoped — closing these closed X1)

- [x] **R10** — Add `if __name__ == "__main__":` entrypoints to `download_hf.py`/`download_kaggle.py` so R8/R9 are actually runnable, not just importable
  — Done (session 5).
- [x] **R11** — Wire `configure_logging()` into those entrypoints and `src/app/main.py` startup
  — Done (session 5). Note: in `main.py`, `configure_logging()` runs after `classifier` import (module-level global load) — logging isn't active for that first model-load. Not reordered — flagged, not fixed, pending your call.
- [x] **R12** — Run `uv lock`, commit `uv.lock` — `Dockerfile`'s `uv sync --frozen` needs it and it's currently missing from the repo
  — Done (session 5). 205 packages resolved, `uv.lock` committed.

**X1 closed (session 5)** — R6 + R10–R12 all done. Next sprint (Postgres ingestion + vLLM/DSPy labelling) not started, not tracked here — see `ROADMAP.md` sequencing.

See `ROADMAP.md` for full sequencing/history, `DECISIONS.md` for this sprint's decisions, `PRD.md` for X1 scope detail.
