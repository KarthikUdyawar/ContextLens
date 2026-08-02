# Design — v1.0.0

## Layered API design

`router → interface (pydantic) → controller → classifier`. Standard FastAPI split. Router owns HTTP verbs/paths, interface owns validation/schema (FastAPI validates the request body against it before the controller runs), controller owns request handling, classifier owns ML logic. Clean separation on paper; broken in practice by the module-level global classifier (ARCHITECTURE.md) which couples controller import to model load.

## Preprocessing as a shared, deterministic step

`TextPreprocessor.preprocessing()` is a single pure(-ish) method used identically at training time (`build_datasets.py`) and inference time (`predict.py`). This is a deliberate and correct choice — it avoids train/serve skew, the most common source of silent ML bugs. Its only impurity is reading 3 lookup CSVs from disk in `__init__` (relative path, not cached across instances beyond the object's lifetime).

## Preprocessing step order (11 steps, fixed order, not configurable)

HTML unescape → strip tags → strip mentions → strip links → emoticon lookup → emoji→text → lowercase → apostrophe lookup → abbreviation lookup → strip non-letters → collapse whitespace.

Order matters: lowercasing happens *after* emoji/emoticon lookup (dictionaries are case-sensitive on lookup keys) but *before* apostrophe/abbreviation lookup (dictionaries expect lowercase input). This is correct as implemented but undocumented — a reordering refactor would silently break lookups.

## Model design

Frozen decision: `bert-base-uncased` + shallow 3-layer FC head, softmax baked into the model rather than left as raw logits. See MODELS.md for why this is a problem paired with `BCEWithLogitsLoss`.

## What's explicitly out of scope for v1.0.0

- No auth, no multi-tenant support, no batching, no streaming responses.
- No model versioning/routing (API always loads `src/model/0.2v/model.pth`, hardcoded).
- No frontend beyond a local Tkinter GUI.
