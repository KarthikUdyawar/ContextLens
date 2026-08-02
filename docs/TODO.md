# TODO

## Baseline (v1.0.0) — cheap, non-breaking — done (session 2)

- [x] Fix double softmax in `predict.py` (DECISIONS.md #1)
- [x] Remove redundant preprocessing call in controller (#3)
- [x] Cache `BertTokenizer` once per classifier instance instead of per-request (#6)
- [x] Migrate `Field(example=...)` to `json_schema_extra` under Pydantic 2.4.2 (#8)
- [x] Add CORS middleware (#9, needed before Streamlit UI can call the API)
- [x] Fix CSV lookup paths to be package-relative, not cwd-relative (#10)
- [x] Sync version string between `setup.py` and `main.py` (#11)
- [x] Add `/health` endpoint (#12)
- [x] Fix `dev-requirements.txt` CUDA-only `torch`/`torchaudio`/`torchvision` pins (#5)
- [x] `classify_sentiment` fails loud (not silent) on missing checkpoint; API maps it to 503; `basic.py`/`gui.py` catch it (#4)
- [x] Docker `HEALTHCHECK` wired to `/health` (INFRA.md gap)
- [ ] Decide + document how `model.pth` reaches a fresh clone/Docker build (#7) — release asset vs registry vs documented manual step. **Still open** — moved to ROADMAP.md's production-grade track, needs a decision next session, not more code.
- [ ] Fix README FastAPI version badge, stale vs pinned `fastapi==0.103.2` (#15, found session 2)
 

## Docs (session 1)

- [x] `ARCHITECTURE.md`, `PIPELINE.md`, `MODELS.md`, `SERVICES.md`, `API_DOC.md`, `STORAGE.md`, `INFRA.md`, `DESIGN.md`, `DECISIONS.md`, `PRD.md`, `PROJECT.tree`, `USER-FLOW.md`, `ROADMAP.md`, `TODO.md`, `handoff.md`

## Docs (session 2)

- [x] `DECISIONS.md` — statuses updated for #1, #3–#6, #8–#12; new #15 logged
- [x] `TODO.md` — this file
- [x] `ROADMAP.md` — new production-grade track added to "Not yet decided"
- [x] `handoff.md` — refreshed for next session

## v2.0.0 (capability track) — pending brainstorm before PRD

- [ ] Pick model: DeBERTa vs RoBERTa vs ModernBERT — shortlist + decision
- [ ] Pick HF dataset — shortlist + decision
- [ ] UV migration plan (pyproject.toml, lockfile, drop requirements.txt/setup.py or keep for compat)
- [ ] pre-commit config (which hooks — black/ruff/mypy?)
- [ ] `.coderabbit.yaml` — review rules/scope
- [ ] Streamlit UI — scope (predict only, or also `/clean`, `/predict-prob` visualization like the existing Tkinter radar chart?)
- [ ] Notebook experiments on new model+data before committing to training pipeline changes
- [ ] Write 2.0 `PRD.md`

## Production-grade track — pending brainstorm, separate from v2.0.0 capability work

See ROADMAP.md's "Not yet decided" section for the full list (observability, security, testing/CI/CD, service architecture, model/data governance, infra). Not scoped into tasks yet — next session.
