# User Flows — v1.0.0

## Flow A — API consumer (developer)

1. `docker compose up` (or `uvicorn src.app.main:app`).
2. `POST /predict/` with `{"text": "..."}`.
3. Reads back `{cleaned_text, sentiment}`.
4. Optionally `POST /predict-prob/` for class confidences.

No auth, no rate limit — flow has no failure/retry branch defined beyond FastAPI's default validation errors.

## Flow B — CLI user (`examples/basic.py`)

1. Run script directly — loads full `TextSentimentClassifier` in-process (own BERT load, own tokenizer, no server).
2. Loop: type text → see cleaned text → see label → see probabilities → prompted `Try again (Y/n)`.
3. Exits on `n`.

## Flow C — GUI user (`examples/gui.py`)

1. Launch Tkinter window (`img/icon.ico` — Windows-only icon load).
2. Type/paste text into the text box.
3. Click "Analyze".
4. Sees predicted label + a radar chart (matplotlib, embedded via `FigureCanvasTkAgg`) of the 3 class probabilities, colored by the winning class.

Flows B and C both instantiate their own `TextSentimentClassifier` — no shared process with the API, so a GUI/CLI user and an API user get independent (but code-identical) inference paths.

## Flow D — Data scientist (offline)

1. Run `notebook/01_get_data.ipynb` → produces `artifacts/Text_dataset.br`.
2. Run `src/pipeline/build_datasets.py` → produces `src/data/{ver}/*.parquet` (local only, not committed).
3. Run `src/pipeline/build_model.py` → trains, conditionally saves `model.pth` + reports under `src/model/{ver}/`.
4. Manually copy `model.pth` to wherever the API/CLI/GUI expects it (`src/model/0.2v/model.pth`, hardcoded in 3 places).

No flow currently publishes or distributes the trained model to Flow A/B/C users — Flow D's output has to be manually placed.
