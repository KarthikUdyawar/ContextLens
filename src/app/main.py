"""FastAPI Main Application"""
from fastapi import FastAPI

from src.app.routers.text_sentiment_router import router

app = FastAPI(
    title="Text Sentiment Analysis API",
    description="An API for analyzing the sentiment of text using FastAPI",
    version="1.0.1",
    openapi_url="/openapi.json",
    redoc_url="/redoc",
    docs_url="/docs",
)


@app.get("/")
async def read_root():
    """Root endpoint of the Text Sentiment Analysis API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Text Sentiment Analysis API"}


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=8000
    )  # RUN: uvicorn src.app.main:app --port 8000 --reload
