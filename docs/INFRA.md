# Infra — v1.0.0

## Container

`Dockerfile`: `python:3.10-slim` base → venv → `pip install .` (installs from `requirements.txt` via `setup.py`, 7 packages) → `uvicorn src.app.main:app --workers 4 --host 0.0.0.0 --port 8000`.

`docker-compose.yml`: single `api` service, port `8000:8000`, `mem_limit: 1g`. No volumes, no env file, no healthcheck, no restart policy.

**Gap:** nothing copies or downloads `model.pth` into the image (it's gitignored — `COPY ./ .` copies whatever's on disk, not tracked in the repo). A container built from a clean clone starts fine and serves broken predictions silently.

## CI

None in v1.0.0. A `pytest`-only GitHub Actions workflow was drafted this session but deferred to v2.0.0 by decision — see `ROADMAP.md`. No lint step, no Docker build step, no deploy step planned yet either.

## Local dependency install

- `requirements.txt` (7 pkgs, runtime) — clean, no GPU-specific pins.
- `dev-requirements.txt` (~140 pkgs, full dev env incl. Jupyter) — pins `torch==2.1.0+cu121`, which requires PyTorch's CUDA wheel index. **`pip install -r dev-requirements.txt` fails on any machine without that extra index configured** (CPU-only dev machines, most CI runners, this container included).

## Deployment

None observed — no deploy workflow, no hosting config, no registry push step. `docker-compose.yml` is local-only.

## Gaps for a real deployment

- No `/health` endpoint for orchestrator probes.
- No env-based config (model path, port, workers all hardcoded).
- No CORS (blocks browser-based UI).
- No model artifact delivery step (registry, release asset, or object storage pull).
