"""FastAPI Main Application"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app.controllers.text_sentiment_controller import classifier
from src.app.routers.text_sentiment_router import router

APP_VERSION = "1.0.1"  # keep in sync with setup.py's `version=` (DECISIONS.md #11)


app = FastAPI(
    title="Text Sentiment Analysis API",
    description="An API for analyzing the sentiment of text using FastAPI",
    version=APP_VERSION,
    openapi_url="/openapi.json",
    redoc_url="/redoc",
    docs_url="/docs",
)

# No browser frontend exists yet in v1.0.0, but the 2.0 Streamlit UI needs this
# and there's no reason to gate it on that landing (DECISIONS.md #9).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RuntimeError)
async def model_not_loaded_handler(_request: Request, exc: RuntimeError):
    """Turn the "checkpoint not loaded" RuntimeError from `classify_sentiment`
    into a clean 503 instead of an unhandled 500 + traceback (DECISIONS.md #4).
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
async def read_root():
    """Root endpoint of the Text Sentiment Analysis API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Text Sentiment Analysis API"}


@app.get("/health")
async def health(response: Response):
    """Liveness/readiness probe endpoint.

    Reports the real checkpoint-loaded state (DECISIONS.md #4) rather than
    just "the process didn't crash" — returns HTTP 503 when the model
    checkpoint failed to load, since `/predict` would otherwise be serving
    from randomly-initialized weights.

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
        app, host="0.0.0.0", port=8000
    )  # RUN: uvicorn src.app.main:app --port 8000 --reload
