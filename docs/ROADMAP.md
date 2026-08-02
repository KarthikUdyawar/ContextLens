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
- **Data:** switch from TextBlob-labeled scraped data to a public, human-labeled HF Hub sentiment dataset (dataset TBD).
- **Compute:** GPU available for training.
- **Tooling:** UV for dependency management (replaces pip/venv workflow), pre-commit hooks, `.coderabbit.yaml` for automated PR review, expanded test coverage, GitHub Actions CI (`tests.yml`, drafted session 1, deferred here).
- **UI:** Streamlit frontend (new consumer alongside the existing API/CLI/GUI).
- **Pipeline:** re-evaluate `build_datasets.py`/`build_model.py` as proper CLI-invokable modules instead of top-to-bottom scripts.

Not yet decided (blocking full PRD.md for 2.0):
- Which of DeBERTa/RoBERTa/ModernBERT specifically, and why.
- Which HF dataset(s) — needs a short evaluation against class balance, size, license.
- Whether v1.0.0's 3-class scheme carries over unchanged.
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
 
## Sequencing

1. ~~v1.0.0 baseline docs~~ (session 1)
2. ~~v1.0.1 patch pass~~ (session 2)
3. 2.0.0 brainstorm → `docs/PRD.md` gets superseded by a versioned 2.0 PRD (or a new `docs/PRD-2.0.md` — naming TBD) — **and** production-grade track brainstorm, sequencing between the two TBD
4. Notebook experiments on candidate model + dataset
5. Implementation (UV migration, pre-commit, coderabbit config, pipeline refactor, Streamlit UI, retrain, production-grade work)
