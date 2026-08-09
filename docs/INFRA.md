# Infra — Sprint X1

## Container

`Dockerfile`: `python:3.12-slim` base (bumped from 3.10, R3) → `uv sync --frozen` (installs from `pyproject.toml`/`uv.lock`, replaces `pip install .`, R1/R2) → `uvicorn src.app.main:app --workers 1 --host 0.0.0.0 --port 8000` (workers dropped 4→1 in v1.0.1 so `/health`'s `model_loaded` reflects the whole container — see SERVICES.md).

`docker-compose.yml`: 4 services as of X1 — `api` (port `8000:8000`, `mem_limit: 1g`), `minio` (`9000`/`9001`, console + API), `minio-init` sidecar (creates `raw-data` bucket on startup, `mc`-based), `postgres` (`5432`, db `contextlens`, schema not yet built), `vllm` (`8001:8000`, `Qwen/Qwen2.5-1.5B-Instruct`, GPU passthrough required, `dns:` override for HF resolution — see DECISIONS.md #28 for why). No env file checked in (`.env.example` only), no restart policy, `HEALTHCHECK` wired for `api` only (v1.0.1).

**Gap:** nothing copies or downloads `model.pth` into the image (it's gitignored — `COPY ./ .` copies whatever's on disk, not tracked in the repo). A container built from a clean clone starts fine and serves broken predictions silently.

## CI

None in v1.0.0. A `pytest`-only GitHub Actions workflow was drafted this session but deferred to v2.0.0 by decision — see `ROADMAP.md`. No lint step, no Docker build step, no deploy step planned yet either.

## Local dependency install (UV, R1/R2/R20)

`setup.py`/`requirements.txt`/`dev-requirements.txt` removed entirely (DECISIONS.md #20) — replaced by `pyproject.toml` + `uv.lock` (205 packages, committed X1 session 5).

```bash
uv sync --frozen                    # runtime only
uv sync --frozen --group dev        # + dev tooling (pytest, ruff, mypy, jupyter, ...)
uv sync --frozen --group train      # + training-only deps (torch training extras, textblob, ...)
uv sync --frozen --group ingest     # + minio/kaggle/datasets (needed for src/ingest/)
```

Groups are additive, not exclusive — combine as needed (`--group dev --group ingest`, etc). No CUDA-index gap anymore (old `dev-requirements.txt` `torch==2.1.0+cu121` pin dropped v1.0.1, DECISIONS.md #5); `pyproject.toml`'s `torch` pin is untagged.

**VSCode/Pylance note:** `uv sync` must include the relevant group for Pylance to resolve imports (`uv run` resolves ad hoc per-invocation but doesn't update the on-disk `.venv` Pylance reads). Select `.venv/bin/python` as the interpreter after syncing.

## Deployment

None observed — no deploy workflow, no hosting config, no registry push step. `docker-compose.yml` is local-only.

## Gaps for a real deployment

- No env-based config for the `api` service (model path, port hardcoded) — `minio`/`postgres`/`vllm` *do* take env vars (`.env.example`).
- No model artifact delivery step (registry, release asset, or object storage pull).
- No Postgres schema yet — service is up, unused (next sprint).
- `vllm`'s `--max-num-seqs 4`/`--gpu-memory-utilization 0.9` tuned for single-user dev box, not real throughput (see handoff.md).
