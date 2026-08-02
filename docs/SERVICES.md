# Services — v1.0.0

## FastAPI service (`src/app`)

Single service, single process. `uvicorn src.app.main:app`.

```
main.py            — app instance, mounts router, GET /
routers/           — text_sentiment_router.py: 3 POST routes
controllers/        — text_sentiment_controller.py: request handling, calls classifier
interfaces/         — text_sentiment_interface.py: pydantic request/response models
```

## Startup behavior

`controllers/text_sentiment_controller.py` instantiates `TextSentimentClassifier` **at module import time** (global). This means:
- the full BERT model loads before the app can serve any request, including `/health` (which doesn't exist)
- a missing checkpoint doesn't crash startup — it logs `"Model not loaded."` and continues, so the process reports healthy while serving broken predictions
- there is no way to override the classifier for testing without loading real weights (no `Depends()`)

## Consumers

- `examples/basic.py` — CLI loop, imports `TextSentimentClassifier` directly (bypasses the API).
- `examples/gui.py` — Tkinter desktop app, also imports the classifier directly, adds a matplotlib radar chart of class probabilities. `iconbitmap(.ico)` call is Windows-only.
- No web/JS frontend exists yet (planned for 2.0 — Streamlit).

## Cross-cutting gaps

- No CORS middleware — blocks any browser-based frontend (relevant once the Streamlit UI lands).
- No auth/rate limiting.
- No `/health` or `/ready` endpoint.
- No request size limits on `TextRequest.text`.

Full endpoint contracts: `API_DOC.md`.