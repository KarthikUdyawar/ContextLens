# Handoff

## Where things stand

v1.0.0 baseline is fully documented (`docs/*.md` + `PROJECT.tree`), 14 findings logged in `DECISIONS.md`, a real bug fixed in `text_preprocessor.py`. Test suite and CI (`.github/workflows/tests.yml`) were both drafted this session but deferred to v2.0.0 by decision.

2.0.0 is scoped but **not yet planned in detail** — direction is confirmed (see `ROADMAP.md`), a full `PRD.md` for 2.0 has not been written. Do not generate it without resolving the open questions in `ROADMAP.md`'s "Not yet decided" list first — user asked explicitly to brainstorm before PRD generation.

## Confirmed 2.0.0 direction (don't re-ask)

- Model: fine-tune newer transformer, shortlist = DeBERTa / RoBERTa / ModernBERT
- Data: public HF Hub dataset (not re-scraped)
- Compute: GPU available
- Tooling: UV, pre-commit, `.coderabbit.yaml`, Streamlit UI
- This is a breaking change → v2.0.0

## Next session should

1. Resolve `ROADMAP.md`'s open questions (specific model, specific dataset, GUI fate, whether cheap v1 fixes land as v1.1 or get folded into 2.0).
2. Only then draft the 2.0 PRD.
3. Reference `DECISIONS.md`/`TODO.md` for the known-bugs backlog — don't re-derive it.

## Skills relevant to next session

`tdd` (for any new pipeline/API code), `clean-code`, `ponytail`, plus whatever the user invokes for the model experimentation notebooks.
