# Roadmap

## v1.0.0 — shipped, baseline documented (session 1)

- FastAPI service, 3 endpoints, Docker + compose.
- BERT-based classifier, TextBlob-labeled training data.
- CLI + Tkinter GUI examples.
- Added session 1: a trailing-whitespace bug fix, full `docs/` baseline (this set), 14 findings logged in `DECISIONS.md`.
- Test suite (pytest) and CI both drafted session 1, deferred to v2.0.0 by decision.

## v1.0.1 — patch pass (session 2)

Answers the "land before 2.0 or bundle into 2.0" question below: landed now, as a patch pass, not deferred.

- `DECISIONS.md` #1, #3–#6, #8–#12 fixed — see `DECISIONS.md` for per-item detail, `TODO.md` for the checklist.
- New: `classify_sentiment` fails loud (not silently) on a missing checkpoint; `/health` reports real model-loaded state (503 when not loaded); Docker `HEALTHCHECK` wired to it.
- Still open, carried forward: #2, #7, #13, #15 (new — stale README badge).
- One regression introduced and fixed same session — see `DECISIONS.md`'s "Session 2" note.

## v2.0.0 — planned (breaking change, capability track)

Direction confirmed:
- **Model:** fine-tune a newer transformer (DeBERTa / RoBERTa / ModernBERT — candidate shortlist, final pick TBD in PRD).
- **Data:** raw text pulled from HF + Kaggle sources (see below); source labels are retained in the downloaded raw files but are ignored by the ingestion and labelling pipeline — dataset relabeled from scratch via self-hosted vLLM + DSPy.
- **Classes:** v1.0.0's 3-class scheme (negative/neutral/positive) carries over unchanged.
- **Compute:** GPU available for training.
- **Tooling:** UV for dependency management (Python bumped 3.10 → 3.12), pre-commit hooks, `.coderabbit.yaml` for automated PR review, expanded test coverage, GitHub Actions CI (`tests.yml`, drafted session 1, deferred here).
- **UI:** Streamlit frontend (new consumer alongside the existing API/CLI/GUI). Tkinter GUI rework/retirement decision deferred.
- **Pipeline:** re-evaluate `build_datasets.py`/`build_model.py` as proper CLI-invokable modules instead of top-to-bottom scripts.

See `DECISIONS.md` for this sprint's decisions and their rationale (data labelling strategy, storage architecture, tooling migration, etc.) — not restated here.

### Data acquisition & labelling pipeline (sequenced)

New infra, new stages, replacing the old `artifacts/Text_dataset.br` → `build_datasets.py` flow:

+1. Download raw text from HF + Kaggle sources (list below) → land in **MinIO** (bucket `raw-data`). HF sources are exported as a canonical `data.parquet` per dataset; Kaggle sources land as-is.
2. Read each landed file, extract text only, insert into **Postgres** (db `contextlens`) with `label = NULL`.
3. Repeat 1–2 across sources until the table reaches the current milestone target of **~1M rows** (long-term goal 5M, not required now).
4. Batch-label the ~1M rows via self-hosted **vLLM** (Docker) driven by **DSPy** programs, updating `label` in place.
5. Split into train/valid/test from the now-labeled Postgres table (replaces `TrainValidTestSplitter` reading parquet directly).

Sources confirmed for the 1M pass (all short-form Twitter/Reddit text, matching v1.0.0's domain):
- HF: `stanfordnlp/sentiment140`, `cardiffnlp/tweet_eval` (sentiment config), `bdstar/twitter-sentiment-analysis`, `bdstar/Tweets-Sentiment-Analysis`
- Kaggle: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, `tariqsays/sentiment-dataset-with-1-million-tweets`

Not yet decided (blocking full PRD.md for 2.0):
- Dedup/near-duplicate handling across merged HF+Kaggle sources (known overlap risk, e.g. Sentiment140 is repackaged in multiple listed sources) — before or after vLLM labelling.
- vLLM/DSPy labelling program design (direct classify vs CoT-style), throughput/cost budget for scaling 1M → 5M.
- Independent eval set strategy — current plan has no eval holdout independent of the vLLM labeler itself.
- Postgres schema for the labeled text table.
- Whether the Tkinter GUI is retired in favor of Streamlit or kept alongside.

## Production-grade track — planned (not yet scoped), separate from v2.0.0 capability work

New track, distinct from the model/data/UI work above — this one's about making the *service* production-ready, independent of which model or dataset ends up behind it. Not yet decided how it sequences relative to 2.0's capability work (before, after, or in parallel). To be discussed next session.

Not yet decided:
- **Observability:** structured/JSON logging (currently `print()` only), a metrics endpoint (Prometheus-style), and what consumes `/health` beyond Docker's own healthcheck (uptime/alerting).
- **Security:** auth (API key / OAuth / JWT), rate limiting, a request size cap on `TextRequest.text`, and a real CORS origin allowlist before Streamlit ships past localhost (loosened to `allow_origins=["*"]` in v1.0.1, see `DECISIONS.md` #9).
- **Testing/CI/CD:** land the pytest suite + `tests.yml` CI drafted (and deferred) in session 1, add lint (ruff/black) and type-check (mypy) gates, and stand up actual CD — registry push + deploy step (currently `docker-compose.yml` is local-only).
- **Service architecture:** replace the module-level global classifier with `Depends()` for testability/mockability; resolve `--workers 4` × in-process model memory/GPU contention (untested); move hardcoded model path out of 3 files into env-based config; add URL versioning (`/v1/predict`) before any 2.0 response-shape change ships.
- **Model/data governance:** experiment tracking (MLflow/W&B or similar), a real model registry instead of a hardcoded checkpoint path (also closes `DECISIONS.md` #7), and a held-out human-labeled eval set independent of the TextBlob labels.
- **Infra:** multi-stage `Dockerfile` (dev deps currently ship in the runtime image), `docker-compose.yml` restart policy + env file + volumes, and a documented deploy target (k8s / ECS / Cloud Run — currently undefined beyond `docker compose up`).

Rough priority if/when this gets scoped: logging → land the deferred tests/CI → auth/rate-limit/size-cap → env config + `Depends()` → CD → then fold in with the 2.0 capability work.

## Sprint X1 (current)

UV migration + MinIO/Postgres/vLLM compose services + Kaggle creds + raw HF/Kaggle data landed in MinIO. Stops once raw files sit in MinIO — Postgres ingestion is a later sprint. Task checklist (task IDs R1–R9) lives in `TODO.md`; detailed scope in `PRD.md`'s X1 section; decisions in `DECISIONS.md`.

## Sequencing

1. ~~v1.0.0 baseline docs~~ (session 1)
2. ~~v1.0.1 patch pass~~ (session 2)
3. ~~Sprint X1~~ (sessions 3–6) — UV migration + MinIO/Postgres/vLLM compose services + Kaggle creds + raw data landed in MinIO (live-verified session 6)
3.5. ~~CodeRabbit review follow-up~~ (session 7) — checkpoint-load hardening, HF revision pinning, MinIO provenance metadata, one new correctness fix (`torch.argmax` axis→dim), doc-drift cleanup. See `TODO.md`'s "CodeRabbit follow-up" checklist and `DECISIONS.md` #24/#25/#32.
4. Next sprint — extract text → Postgres (`label = NULL`), grow to ~1M rows
5. vLLM/DSPy batch labelling pass over the ~1M rows
6. 2.0.0 brainstorm → `docs/PRD.md` gets superseded by a versioned 2.0 PRD (or a new `docs/PRD-2.0.md` — naming TBD) — **and** production-grade track brainstorm, sequencing between the two TBD
7. Notebook experiments on candidate model + labeled dataset
8. Implementation (pre-commit, coderabbit config, pipeline refactor, Streamlit UI, retrain, production-grade work)
