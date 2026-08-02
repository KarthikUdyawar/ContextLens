# TODO

## Baseline (v1.0.0) — cheap, non-breaking, do anytime

- [ ] Fix double softmax in `predict.py` (DECISIONS.md #1)
- [ ] Remove redundant preprocessing call in controller (#3)
- [ ] Cache `BertTokenizer` once per classifier instance instead of per-request (#6)
- [ ] Verify `Field(example=...)` actually renders under Pydantic 2.4.2; migrate to `json_schema_extra` if not (#8)
- [ ] Add CORS middleware (#9, needed before Streamlit UI can call the API)
- [ ] Fix CSV lookup paths to be package-relative, not cwd-relative (#10)
- [ ] Sync version string between `setup.py` and `main.py` (#11)
- [ ] Add `/health` endpoint (#12)
- [ ] Decide + document how `model.pth` reaches a fresh clone/Docker build (#7) — release asset vs registry vs documented manual step

## Docs (this session)

- [x] `ARCHITECTURE.md`, `PIPELINE.md`, `MODELS.md`, `SERVICES.md`, `API_DOC.md`, `STORAGE.md`, `INFRA.md`, `DESIGN.md`, `DECISIONS.md`, `PRD.md`, `PROJECT.tree`, `USER-FLOW.md`, `ROADMAP.md`, `TODO.md`, `handoff.md`

## v2.0.0 — pending brainstorm before PRD

- [ ] Pick model: DeBERTa vs RoBERTa vs ModernBERT — shortlist + decision
- [ ] Pick HF dataset — shortlist + decision
- [ ] UV migration plan (pyproject.toml, lockfile, drop requirements.txt/setup.py or keep for compat)
- [ ] pre-commit config (which hooks — black/ruff/mypy?)
- [ ] `.coderabbit.yaml` — review rules/scope
- [ ] Streamlit UI — scope (predict only, or also `/clean`, `/predict-prob` visualization like the existing Tkinter radar chart?)
- [ ] Notebook experiments on new model+data before committing to training pipeline changes
- [ ] Write 2.0 `PRD.md`
