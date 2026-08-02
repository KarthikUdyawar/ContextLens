# Roadmap

## v1.0.0 — shipped, baseline documented (this doc set)

- FastAPI service, 3 endpoints, Docker + compose.
- BERT-based classifier, TextBlob-labeled training data.
- CLI + Tkinter GUI examples.
- Added this session: a trailing-whitespace bug fix, full `docs/` baseline (this set), 14 findings logged in `DECISIONS.md`.
- Test suite (pytest) and CI both drafted this session, deferred to v2.0.0 by decision.

## v2.0.0 — planned (breaking change)

Direction confirmed:
- **Model:** fine-tune a newer transformer (DeBERTa / RoBERTa / ModernBERT — candidate shortlist, final pick TBD in PRD).
- **Data:** switch from TextBlob-labeled scraped data to a public, human-labeled HF Hub sentiment dataset (dataset TBD).
- **Compute:** GPU available for training.
- **Tooling:** UV for dependency management (replaces pip/venv workflow), pre-commit hooks, `.coderabbit.yaml` for automated PR review, expanded test coverage, GitHub Actions CI (`tests.yml`, drafted this session, deferred here).
- **UI:** Streamlit frontend (new consumer alongside the existing API/CLI/GUI).
- **Pipeline:** re-evaluate `build_datasets.py`/`build_model.py` as proper CLI-invokable modules instead of top-to-bottom scripts.

Not yet decided (blocking full PRD.md for 2.0):
- Which of DeBERTa/RoBERTa/ModernBERT specifically, and why.
- Which HF dataset(s) — needs a short evaluation against class balance, size, license.
- Whether v1.0.0's 3-class scheme carries over unchanged.
- Whether the Tkinter GUI is retired in favor of Streamlit or kept alongside.
- Fixes from `DECISIONS.md` #1, #3, #6, #8–#12 are cheap, non-breaking, and don't require the retrain — candidates to land *before* 2.0 as a v1.1.0, or bundled into 2.0. Open question.

## Sequencing

1. ~~v1.0.0 baseline docs~~ (this batch)
2. 2.0.0 brainstorm → `docs/PRD.md` gets superseded by a versioned 2.0 PRD (or a new `docs/PRD-2.0.md` — naming TBD)
3. Notebook experiments on candidate model + dataset
4. Implementation (UV migration, pre-commit, coderabbit config, pipeline refactor, Streamlit UI, retrain)
