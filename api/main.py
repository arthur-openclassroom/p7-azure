# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from api.model_loader import load_artifacts, predict_sentiment
from src.preprocessing import clean_tweet


class TweetRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    tweet_clean: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Air Paradis - Sentiment API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    cleaned = clean_tweet(request.text)
    result = predict_sentiment(cleaned)
    return PredictionResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        tweet_clean=cleaned,
    )
