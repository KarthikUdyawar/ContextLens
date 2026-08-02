# Architecture — v1.0.0

## Layers

```
┌─────────────────────────────────────────────┐
│ Notebooks (offline, manual)                  │
│ 01_get_data → 02_clean_text → 03_eda →       │
│ 04_train_model                               │
└───────────────────┬───────────────────────────┘
                    │ produces
┌───────────────────▼───────────────────────────┐
│ src/pipeline (scripts, run manually)          │
│ build_datasets.py → build_model.py            │
│ predict.py (inference, imported by app)       │
└───────────────────┬───────────────────────────┘
                    │ used by
┌───────────────────▼───────────────────────────┐
│ src/app (FastAPI service)                     │
│ router → controller → TextSentimentClassifier │
└─────────────────────────────────────────────────┘

Consumers: examples/basic.py (CLI), examples/gui.py (Tkinter)
```

## Components

| Component                 | Path                                                            | Role                                                                                                          |
| ------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `TextPreprocessor`        | `src/utils/text_preprocessor.py`                                | Pure text cleaning (HTML, mentions, links, emoji, contractions, abbreviations). No model dependency.          |
| `CustomBERTClassifier`    | `src/utils/custom_BERT_classifier.py`                           | `bert-base-uncased` + 3-layer FC head, **softmax applied inside `forward()`**.                                |
| `TextDataset`             | `src/utils/text_dataset.py`                                     | Wraps `BertTokenizer`, tokenizes on `__getitem__`. Tokenizer loaded fresh per instance (see DECISIONS.md #6). |
| `TextSentimentClassifier` | `src/pipeline/predict.py`                                       | Inference facade: preprocess → tokenize → forward → label.                                                    |
| FastAPI app               | `src/app/main.py` + `routers/` + `controllers/` + `interfaces/` | 3 REST endpoints, classifier instantiated once at module import.                                              |

## Request path

`POST /predict` → router → controller (`predict_sentiment`) → `TextSentimentClassifier.classify_sentiment` → preprocess (again, internally) → tokenize → BERT forward → softmax (applied twice, see DECISIONS.md #1) → label.

## Known structural gaps

- No dependency injection — classifier is a module-level global, loaded at import time. Can't swap/mock for tests without loading real BERT weights.
- No shared config/settings module — model path is a string literal duplicated in `controller.py`, `examples/basic.py`, `examples/gui.py`.
- No `/health` endpoint.

See `DECISIONS.md` for the full bug/gap list, `PIPELINE.md` for the training-side detail.
