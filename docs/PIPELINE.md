# Pipeline — v1.0.0

## Stages

1. **`notebook/01_get_data.ipynb`** — collects raw text from Reddit/Twitter, consolidates into `artifacts/Text_dataset.br` (brotli parquet). Not reproducible from a script — notebook-only.
2. **`notebook/02_clean_text.ipynb`** / **`src/utils/text_preprocessor.py`** — HTML unescape → strip tags → strip `@mentions` → strip links → emoticon lookup → emoji→text → lowercase → apostrophe/contraction lookup → abbreviation lookup → strip non-letters → collapse whitespace. Same class used at train time and inference time (good — no train/serve skew here).
3. **`notebook/03_eda.ipynb`** — exploratory only, no pipeline output.
4. **`src/pipeline/build_datasets.py`** — script (not a function, runs top-to-bottom on import):
   - loads `artifacts/Text_dataset.br`, samples `BATCH_SIZE = 30000` rows (`random_state=42`)
   - cleans text via `TextPreprocessor`
   - labels sentiment via `TextBlob(text).sentiment.polarity` (**not human-labeled — polarity-derived pseudo-labels**, see DECISIONS.md #13)
   - filters rows with 0 or ≥100 tokens
   - splits train/valid/test via `TrainValidTestSplitter` (stratified, oversamples minority classes on train only)
   - writes `src/data/0.2v/{train,valid,test}_data.parquet` (**directory not present in repo — must be generated locally, not committed**)
5. **`notebook/04_train_model.ipynb`** / **`src/pipeline/build_model.py`** — script (also runs top-to-bottom):
   - `CustomBERTClassifier(num_classes=3)`, `Adam(lr=1e-5)`, `BCEWithLogitsLoss`
   - one-hot encoded targets, up to 50 epochs, early stop after 3 non-improving epochs on val loss
   - saves `src/model/{VER}/model.pth` only if `test_accuracy > 0.85`
   - writes report artifacts via `ModelReportManager` (confusion matrix, training curves, classification report)
6. **`src/pipeline/predict.py`** — `TextSentimentClassifier`, the only pipeline stage imported by the live API.

## Reproducibility gaps

- Steps 1 and 4/5 have both a notebook version and a script version — no single source of truth, risk of drift.
- `build_datasets.py` / `build_model.py` have no `if __name__ == "__main__"` guard and no CLI args — importing them for testing executes the full pipeline.
- Labels are polarity heuristics (TextBlob), not verified ground truth — model quality is bounded by lexicon-based labeling.
- `src/data/` and `model.pth` are both gitignored — a fresh clone cannot serve predictions or resume training without regenerating from `artifacts/Text_dataset.br`.
