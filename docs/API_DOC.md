# API — v1.0.0

Base URL: `http://localhost:8000` · Interactive docs: `/docs` (Swagger), `/redoc`

## `GET /`

Welcome message only.

```json
{ "message": "Welcome to the Text Sentiment Analysis API" }
```

## `POST /predict/`

Predict sentiment label.

**Request**
```json
{ "text": "This is a sample text for sentiment analysis." }
```

**Response**
```json
{
  "cleaned_text": "this is a sample text for sentiment analysis",
  "sentiment": "neutral"
}
```

Note: server-side this calls `preprocess_text()` once for the returned `cleaned_text`, then `classify_sentiment()` which preprocesses the *raw* input again internally — text is cleaned twice per request (DECISIONS.md #3).

## `POST /predict-prob/`

Same as `/predict/`, plus raw class probabilities.

**Response**
```json
{
  "cleaned_text": "this is a sample text for sentiment analysis",
  "sentiment_prob": [0.21, 0.58, 0.21]
}
```
Order: `[negative, neutral, positive]`. **Values are miscalibrated** — see MODELS.md, double-softmax issue.

## `POST /clean/`

Preprocessing only, no model call.

**Response**
```json
{ "cleaned_text": "this is a sample text for sentiment analysis" }
```

## Schema notes

- `interfaces/text_sentiment_interface.py` uses `Field(..., example=...)` — Pydantic v1 syntax, project pins `pydantic==2.4.2`. Verify the Swagger examples actually render; v2 may silently drop this kwarg.
- No error responses documented (no 4xx/5xx schemas) — FastAPI default validation errors only.
- No versioning in the URL path (`/v1/predict` etc.) — relevant if 2.0 changes response shape.
