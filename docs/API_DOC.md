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

Note: server-side this calls `preprocess_text()` once; the cleaned text is passed through unchanged to `classify_sentiment()` — one preprocessing pass per request (DECISIONS.md #3, fixed v1.0.1).

## `POST /predict-prob/`

Same as `/predict/`, plus raw class probabilities.

**Response**
```json
{
  "cleaned_text": "this is a sample text for sentiment analysis",
  "sentiment_prob": [0.21, 0.58, 0.21]
}
```
Order: `[negative, neutral, positive]`. Values reflect the model's single softmax output (DECISIONS.md #1, fixed v1.0.1). Calibration against human-labeled ground truth is still unverified — training labels are TextBlob polarity heuristics (see MODELS.md #3, DECISIONS.md #13).

## `POST /clean/`

Preprocessing only, no model call.

**Response**
```json
{ "cleaned_text": "this is a sample text for sentiment analysis" }
```

## Schema notes

- `interfaces/text_sentiment_interface.py` migrated to `json_schema_extra={"example": ...}` under pydantic 2.4.2 (DECISIONS.md #8, fixed v1.0.1). Swagger examples confirmed rendering.
- No error responses documented (no 4xx/5xx schemas) — FastAPI default validation errors only.
- No versioning in the URL path (`/v1/predict` etc.) — relevant if 2.0 changes response shape.
