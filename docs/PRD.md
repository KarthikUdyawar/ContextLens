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