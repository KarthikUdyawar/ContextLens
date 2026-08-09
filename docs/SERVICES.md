# Services — v1.0.0

## FastAPI service (`src/app`)

Local dev: single process, `uvicorn src.app.main:app`. Container: `--workers 1` (Dockerfile, updated v1.0.1) — one worker per container so `/health`'s `model_loaded` state reflects the whole container; scale via replicas, not `--workers`, to avoid per-worker readiness drift and untested classifier memory/GPU contention under multiple in-process model copies.

```text
main.py            — app instance, mounts router, GET /
routers/           — text_sentiment_router.py: 3 POST routes
controllers/        — text_sentiment_controller.py: request handling, calls classifier
interfaces/         — text_sentiment_interface.py: pydantic request/response models
```

## Startup behavior

`controllers/text_sentiment_controller.py` instantiates `TextSentimentClassifier` **at module import time** (global). This means:
- the full BERT model loads before the app can serve any request, including `/health`
- a missing checkpoint doesn't crash startup, but no longer reports healthy either — `classify_sentiment`
  raises `RuntimeError`, `/health` reports real `model_loaded` state and returns 503 when false
  (DECISIONS.md #4, #12, fixed v1.0.1)

## Consumers

- `examples/basic.py` — CLI loop, imports `TextSentimentClassifier` directly (bypasses the API).
- `examples/gui.py` — Tkinter desktop app, also imports the classifier directly, adds a matplotlib radar chart of class probabilities. `iconbitmap(.ico)` call is Windows-only.
- No web/JS frontend exists yet (planned for 2.0 — Streamlit).

## Ingest scripts (X1, not a service — run manually, no server process)

```text
src/ingest/
  download_hf.py       — HfDownloader, 4 HF sources → MinIO raw-data/hf/
  download_kaggle.py   — KaggleDownloader, 2 Kaggle sources → MinIO raw-data/kaggle/
  minio_client.py      — MinioClient, thin SDK wrapper
```

Both scripts run via `if __name__ == "__main__":` (R10), collect per-source failures without stopping (partial-failure tolerant), exit 1 if any source failed. Live-verified end-to-end (all 6 sources landed) — see `DECISIONS.md` #31. Not wired to the FastAPI service or any other consumer — standalone, invoked manually via `uv run --group ingest python -m src.ingest.download_hf`.

## Cross-cutting gaps

- CORS middleware added v1.0.1 (`allow_origins=["*"]`, DECISIONS.md #9) — revisit origin allowlist before Streamlit ships past localhost.
- No auth/rate limiting.
- `/health` added v1.0.1, readiness-aware (DECISIONS.md #12). No `/ready` separate endpoint.
- No request size limits on `TextRequest.text`.

Full endpoint contracts: `API_DOC.md`.
