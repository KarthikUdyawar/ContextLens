# Handoff

## Where things stand

v1.0.0 baseline + v1.0.1 patch pass are done (see prior handoff / `DECISIONS.md` for that history — not restated here). This session planned and doc'd **Sprint X1**, the first sprint of the 2.0.0 data track. No code written yet — this session was planning + docs only.

**Do not re-derive any of this — read the docs, they're current:**
- `ROADMAP.md` — full history, v2.0.0 direction, data acquisition & labelling pipeline (5 sequenced stages), Sprint X1 summary, updated Sequencing list.
- `PRD.md` — v1.0.0 retrospective PRD (unchanged) + new full PRD section for Sprint X1 (Problem/Users/Scope/Sources/What it does & doesn't do/Success criteria).
- `DECISIONS.md` — original 15 v1.0.0 findings (unchanged) + new "Sprint X1 / 2.0.0 data-track decisions" table, #16–23.
- `TODO.md` — rewritten to contain **only** Sprint X1's task list, task IDs **R1–R9**. All older checklists (baseline, docs, old v2.0.0/production-grade lists) were deliberately removed from this file per user request — that content still exists in `ROADMAP.md`/`DECISIONS.md`, just not as a task list anymore.

## Key decisions this session (also in DECISIONS.md #16–23, don't re-litigate)

- **Labelling:** self-hosted **vLLM + DSPy** replaces TextBlob — not adopting a pre-labeled public HF dataset. Reason: surveyed HF 3-class sentiment sets are either heuristic-labeled (same problem as TextBlob) or already model-labeled by someone else's model — neither is an improvement. Self-hosted labelling is owned/inspectable.
- **3-class scheme** (negative/neutral/positive) carries over unchanged into 2.0.0 — not revisited.
- **Storage:** MinIO (raw landing, bucket `raw-data`) → Postgres (extracted text + labels, db `contextlens`) — Postgres chosen over parquet-only specifically so multi-source text stays queryable for analysis later.
- **Sources confirmed** (short-form Twitter/Reddit only, no long-form text): HF `sentiment140`, `cardiffnlp/tweet_eval` (sentiment config), `bdstar/twitter-sentiment-analysis`, `bdstar/Tweets-Sentiment-Analysis`; Kaggle `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, `tariqsays/sentiment-dataset-with-1-million-tweets`. Source-provided labels are downloaded as part of each file but never read/used anywhere.
- **Tooling:** full UV migration — `pyproject.toml` replaces `setup.py`/`requirements.txt`/`dev-requirements.txt` entirely, no pip-compat kept. Dependency groups named `dev` and `train` (simple, matches old runtime/dev split); runtime deps live in base `[project.dependencies]`. Python floor bumped 3.10 → 3.12.
- **Target scale:** ~1M rows for now (long-term goal 5M, not required this phase).

## Sprint X1 scope — where the next session should start coding

Task IDs **R1–R9** are the actual checklist, live in `TODO.md`. Summary:
- R1–R3: UV migration (`pyproject.toml` w/ `dev`+`train` groups, drop old dep files, bump to Python 3.12)
- R4–R6: `docker-compose.yml` — add `minio` (bucket `raw-data`), `postgres` (db `contextlens`), `vllm` (**image/model still open** — not decided yet)
- R7: Kaggle API credentials, env-based (`KAGGLE_USERNAME`/`KAGGLE_KEY`), via `.env.example`
- R8–R9: download scripts (HF sources, Kaggle sources) → land raw files in MinIO under `raw-data/hf/...` and `raw-data/kaggle/...`, unmodified, unlabeled

**X1 stops once raw files are sitting in MinIO.** Text extraction into Postgres (`label = NULL`) is explicitly a *later* sprint, not part of X1 — don't scope-creep into it.

## Known open question blocking R6

vLLM's model/image was never picked this session — flagged in both `TODO.md` (R6) and `ROADMAP.md`'s "Not yet decided" list. Needs a decision before R6 can actually be implemented (R1–R5, R7–R9 have no such blocker).

## Note on R1 (flagged, not yet resolved)

Splitting a `train` dependency group out of the current flat `dev-requirements.txt` is new work, not a mechanical lift — there's no existing train/dev split to copy from. Whoever picks up R1 should expect to actually sort which dev-requirements entries are training-only vs general-dev.

## Skills relevant to next session

`tdd` (once R1–R9 code starts — download scripts, compose services should get tests), `clean-code`, `ponytail` (esp. for the docker-compose additions — keep default configs, avoid over-engineering three brand-new services), `caveman ultra` (communication style, active this session, persists unless user says otherwise).

## Conventions carried over (unchanged, don't re-ask)

- Code changes in `git diff` format, files shared in blockquote (`>`) format.
- Karthik shares code file-by-file on request, confirms each diff batch, prompts explicit completeness checks before moving on.
- Bugs found mid-task get fixed same session, not deferred.
