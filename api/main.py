import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.model_loader import load_artifacts, predict_sentiment
from api.preprocessing import clean_tweet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    tweet_clean: str


app = FastAPI(title="Air Paradis - Sentiment API")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


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
