# TODO

## Sprint X1 (current) — UV migration + raw data to MinIO

### Tooling — UV migration

- [ ] **R1** — Write `pyproject.toml`: runtime deps under `[project.dependencies]` (from `requirements.txt`); `dev` dependency group (from `dev-requirements.txt`); `train` dependency group (training-only deps split out of the current flat dev list — torch training extras, TextBlob, etc.)
- [ ] **R2** — Remove `setup.py`, `requirements.txt`, `dev-requirements.txt` — no pip-compat retained (DECISIONS.md #20)
- [ ] **R3** — Bump Python floor 3.10 → 3.12: `pyproject.toml`, `Dockerfile` base image, any CI config referencing 3.10

### Infra — docker-compose.yml

- [ ] **R4** — Add `minio` service: image `minio/minio`, API + console ports exposed, default bucket `raw-data` created on startup (DECISIONS.md #22)
- [ ] **R5** — Add `postgres` service: default image, db name `contextlens` (DECISIONS.md #23), default port 5432 — table schema not built yet (next sprint)
- [ ] **R6** — Add `vllm` service: image/model **still open**, see `ROADMAP.md` "Not yet decided" — placeholder service until model is picked

### Credentials

- [ ] **R7** — Kaggle API credentials: env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`, documented in `.env.example`, never committed

### Download scripts

- [ ] **R8** — HF download script: `sentiment140`, `cardiffnlp/tweet_eval` (sentiment config), `bdstar/twitter-sentiment-analysis`, `bdstar/Tweets-Sentiment-Analysis` → MinIO `raw-data/hf/<dataset>/`, raw, unmodified, labels ignored
- [ ] **R9** — Kaggle download script: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, `tariqsays/sentiment-dataset-with-1-million-tweets` → MinIO `raw-data/kaggle/<dataset>/`, raw, unmodified, labels ignored


**Stops here.** Next sprint (Postgres ingestion + vLLM/DSPy labelling) not started — not tracked until X1 closes.

See `ROADMAP.md` for full sequencing/history, `DECISIONS.md` for this sprint's decisions, `PRD.md` for X1 scope detail.
