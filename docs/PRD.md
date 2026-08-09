# PRD — ContextLens v1.0.0 (retrospective baseline)

This is a retrospective PRD, written against what v1.0.0 already ships — captured to set a documented baseline before 2.0.0 planning starts.

## Problem

Classify the sentiment (negative/neutral/positive) of short user-supplied text (originally scraped Reddit/Twitter content), and expose that as both a reusable Python library and a served API.

## Users

- Developers integrating sentiment analysis into their own pipeline via the FastAPI service or `pip install .`
- End users running the local Tkinter GUI or CLI example directly, no server involved.

## What v1.0.0 does

- Cleans arbitrary text through an 11-step deterministic preprocessing pipeline (`TextPreprocessor`).
- Classifies cleaned text into 3 sentiment classes using a fine-tuned `bert-base-uncased` model.
- Serves 3 REST endpoints (`/predict`, `/predict-prob`, `/clean`) via FastAPI, containerized with Docker.
- Ships a CLI example and a Tkinter GUI (with a live radar chart of class probabilities) for local, serverless use.
- Provides training scripts + notebooks to reproduce the model from raw scraped data.

## What v1.0.0 does not do

- Does not ship trained weights or the assembled training dataset — both are gitignored, so the API cannot serve real predictions out of a fresh clone without external artifacts.
- Does not authenticate requests, rate-limit, batch, or version model responses.
- Has no web UI (Streamlit UI is a 2.0.0 target).
- Has no CI/CD deployment pipeline (unit test CI added this session; no lint/build/deploy stages).

## Success criteria (as implemented)

`build_model.py` only persists a checkpoint if held-out accuracy exceeds 0.85 — measured against `TextBlob`-derived polarity labels, not independently verified ground truth (see DECISIONS.md #13). No other product-level success metric (latency SLA, user satisfaction, etc.) is defined in the codebase.

## Known limitations

See `DECISIONS.md` for the full, numbered list of correctness, performance, and reliability findings from the v1.0.0 baseline audit.

---

# PRD — Sprint X1 (current, part of the 2.0.0 data track)

This PRD covers X1 only — the first sprint toward 2.0.0's data pipeline. It does not restate 2.0.0's overall direction or history; see `ROADMAP.md` for that. It does not restate why each choice was made; see `DECISIONS.md` for this sprint's decisions.

## Problem

The 2.0.0 retrain needs a large, domain-matched, owned training corpus, replacing v1.0.0's TextBlob-heuristic-labeled, notebook-only-reproducible dataset. Before any labelling or training work can start, raw text has to be acquired at scale and land somewhere durable and queryable. X1 is that first step — acquisition only.

## Users

Internal only — this sprint has no end-user-facing surface. Consumers of X1's output are the next sprint's ingestion step and, ultimately, the 2.0.0 training pipeline.

## Scope

X1 covers acquisition and tooling migration only. No labelling, no Postgres ingestion, no training changes, no model work.

- **Tooling:** migrate to UV — `pyproject.toml` (runtime deps in `[project.dependencies]`, `dev` and `train` dependency groups) replaces `setup.py` + `requirements.txt` + `dev-requirements.txt`. Python floor bumped 3.10 → 3.12.
- **Infra:** add MinIO (bucket `raw-data`), Postgres (db `contextlens`), and vLLM as new services in `docker-compose.yml`, alongside the existing `api` service. Postgres and vLLM are stood up but not yet used — their consumers are the next sprint. vLLM runs `Qwen/Qwen2.5-1.5B-Instruct` on image `vllm/vllm-openai:v0.6.3.post1` (see `DECISIONS.md` #28).
- **Credentials:** Kaggle API auth wired via env vars (`KAGGLE_USERNAME` / `KAGGLE_KEY`) for automated, non-interactive downloads.
- **Download script:** pulls raw text from the confirmed HF + Kaggle sources below and lands it in MinIO's
  `raw-data` bucket, source-prefixed. HF sources are exported as a canonical `data.parquet` per dataset
  (`raw-data/hf/<dataset>/data.parquet`); Kaggle sources land unmodified (`raw-data/kaggle/<dataset>/*.csv`).
  Both unlabeled — source labels ignored (see `DECISIONS.md`).

Task-level breakdown (R1–R9) lives in `TODO.md`.

## Sources (confirmed, short-form Twitter/Reddit text only — no long-form review data)

- HF: `stanfordnlp/sentiment140`, `cardiffnlp/tweet_eval` (sentiment config), `bdstar/twitter-sentiment-analysis`, `bdstar/Tweets-Sentiment-Analysis`
- Kaggle: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, `tariqsays/sentiment-dataset-with-1-million-tweets`

Each source file may carry its own sentiment labels; those are downloaded as-is (part of the file) but not read or used anywhere in X1 or later — see `DECISIONS.md` for why.

## What X1 does

- Produces raw, unlabeled text files sitting in MinIO, sourced from the 6 confirmed HF+Kaggle datasets.
- Stands up MinIO/Postgres/vLLM as running services in `docker-compose.yml`.
- Replaces the project's dependency tooling with UV on Python 3.12.

## What X1 does not do

- No text extraction into Postgres — next sprint.
- No labelling — vLLM/DSPy work is next sprint.
- No dedup or near-duplicate handling across sources — open question, tracked in `ROADMAP.md`.
- No training or model changes.
- No Postgres schema design — deferred to the next sprint's implementation.

## Success criteria

- All 6 confirmed sources downloaded and present in MinIO as raw files.
- `docker-compose.yml` brings up `api`, `minio`, `postgres`, `vllm` cleanly.
- `pip install .` workflow fully replaced by UV; Python 3.12 confirmed working.
- Kaggle downloads authenticate via env credentials, no manual token handling.
