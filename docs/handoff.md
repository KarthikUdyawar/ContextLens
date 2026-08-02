# Handoff

## Where things stand

v1.0.0 baseline fully documented (session 1). v1.0.1 patch pass done (session 2): 10 of 14 baseline findings fixed (`DECISIONS.md` #1, #3–#6, #8–#12), plus Docker `HEALTHCHECK` wiring and CLI/GUI error handling for the #4 fix. Full per-item detail is in `DECISIONS.md`; don't re-derive it here.

Still open, carried forward: #2, #7, #13 (all deliberate — need retrain/decision, not more patching), #15 (new, cosmetic — stale README badge).

2.0.0 is scoped but **not yet planned in detail** — direction confirmed (see `ROADMAP.md`), full `PRD.md` for 2.0 not written. Do not generate it without resolving `ROADMAP.md`'s "Not yet decided" list first.

A second, separate track was added this session: **production-grade readiness** (observability, security, testing/CI/CD, service architecture, model/data governance, infra). It's independent of which model/dataset ends up in 2.0 — see `ROADMAP.md`'s new "Production-grade track" section for the full list. Not yet scoped into tasks, not yet sequenced against 2.0's capability work. Both tracks need a planning session.

## Confirmed 2.0.0 capability direction (don't re-ask)

- Model: fine-tune newer transformer, shortlist = DeBERTa / RoBERTa / ModernBERT
- Data: public HF Hub dataset (not re-scraped)
- Compute: GPU available
- Tooling: UV, pre-commit, `.coderabbit.yaml`, Streamlit UI
- This is a breaking change → v2.0.0

## Next session should

1. Decide how the two open tracks (2.0 capability, production-grade) relate — sequential, parallel, or production-grade gates 2.0's release. This wasn't decided this session, don't assume either way.
2. Resolve `ROADMAP.md`'s 2.0 "Not yet decided" list (specific model, specific dataset, GUI fate) if capability planning goes first.
3. Scope the production-grade track's "Not yet decided" list into actual tasks if that goes first — it's currently a list of areas, not tickets.
4. Only after whichever track is planned, draft the corresponding PRD (`PRD-2.0.md` and/or a production-readiness doc — naming/existence TBD, don't create either speculatively).
5. Reference `DECISIONS.md`/`TODO.md`/`ROADMAP.md` for backlog and status — don't re-derive them.

## Contract changes from this session worth knowing before touching related code

- `TextSentimentClassifier.classify_sentiment(cleaned_text, ...)` now expects **already-preprocessed** text (param renamed from `input_text`). Callers must call `preprocess_text` first.
- It now **raises `RuntimeError`** if the checkpoint didn't load, instead of silently predicting on random weights. Every caller needs to handle that (API does via a global exception handler in `main.py`; `basic.py`/`gui.py` do via try/except).
- `return_probabilities=True` now returns a plain `list`, not a numpy array.
 
## Skills relevant to next session

`tdd` (for any new pipeline/API code, and for finally landing the deferred pytest suite/CI), `clean-code`, `ponytail`, plus whatever the user invokes for the model experimentation notebooks.
