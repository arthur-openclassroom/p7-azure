import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.model_loader import load_artifacts, predict_sentiment
from api.preprocessing import clean_tweet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    connection_string = os.getenv("APPINSIGHTS_CONNECTION_STRING", "")
    if connection_string:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=connection_string)
    yield


class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    tweet_clean: str


class FeedbackRequest(BaseModel):
    tweet: str
    predicted_sentiment: str
    confidence: float


app = FastAPI(title="Air Paradis - Sentiment API", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    logger.warning(
        "Mauvaise prediction signalee",
        extra={
            "custom_dimensions": {
                "tweet": request.tweet,
                "predicted_sentiment": request.predicted_sentiment,
                "confidence": str(request.confidence),
            }
        },
    )
    return {"status": "recorded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    logger.info("predict called with text=%r", request.text)
    load_artifacts()
    logger.info("artifacts loaded")
    cleaned = clean_tweet(request.text)
    logger.info("cleaned=%r", cleaned)
    result = predict_sentiment(cleaned)
    logger.info("result=%s", result)
    return PredictionResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        tweet_clean=cleaned,
    )
