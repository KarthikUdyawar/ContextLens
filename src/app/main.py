"""FastAPI Main Application."""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.controllers.text_sentiment_controller import classifier
from src.app.routers.text_sentiment_router import router
from src.pipeline.predict import ModelNotReadyError
from src.utils.logging_config import configure_logging

APP_VERSION = "2.0.1"  # keep in sync with setup.py's `version=` (DECISIONS.md #11)

configure_logging()

app = FastAPI(
    title="Text Sentiment Analysis API",
    description="An API for analyzing the sentiment of text using FastAPI",
    version=APP_VERSION,
    openapi_url="/openapi.json",
    redoc_url="/redoc",
    docs_url="/docs",
)

# allow_credentials=False: no auth/cookie flow exists yet, and browsers reject
# allow_origins=["*"] + allow_credentials=True anyway (DECISIONS.md #9).
# Revisit origin allowlist + credentials together once Streamlit needs auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(
    _request: Request, _exc: ModelNotReadyError
) -> JSONResponse:
    """Checkpoint didn't load -> 503, fixed public detail (no exc text leaked)."""
    return JSONResponse(
        status_code=503, content={"detail": "Service unavailable: model not ready."}
    )


@app.get("/")
async def read_root() -> dict[str, str]:
    """Root endpoint of the Text Sentiment Analysis API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Text Sentiment Analysis API"}


@app.get("/health")
async def health(response: Response) -> dict[str, object]:
    """Liveness/readiness probe endpoint.

    Reports the real checkpoint-loaded state (DECISIONS.md #4) rather than
    just "the process didn't crash" — returns HTTP 503 when the model
    checkpoint failed to load, since `/predict` would otherwise be serving
    from randomly-initialized weights. Accurate per-process; run one worker
    per container (see Dockerfile) so this reflects the whole container.

    Returns:
        dict: Service status, model-loaded flag, and version.
    """
    if not classifier.model_loaded:
        response.status_code = 503
    return {
        "status": "ok" if classifier.model_loaded else "degraded",
        "model_loaded": classifier.model_loaded,
        "version": APP_VERSION,
    }


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8000  # nosec B104: intentional,
        # container needs to bind all interfaces — see INFRA.md
    )  # RUN: uvicorn src.app.main:app --port 8000 --reload
