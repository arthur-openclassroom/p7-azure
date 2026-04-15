# api/model_loader.py - classical sklearn model (TF-IDF + LogisticRegression pipeline)
import logging
import os
import pickle

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "classical_model.pkl")

_model = None


def load_artifacts():
    global _model
    if _model is not None:
        return
    logger.info("Loading model from %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file NOT FOUND at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)  # noqa: S301 - trusted local ML model file
    logger.info("Model loaded OK")


def predict_sentiment(cleaned_text: str) -> dict:
    if not cleaned_text:
        return {"sentiment": "negatif", "confidence": 0.5}
    proba = _model.predict_proba([cleaned_text])[0]
    label = int(_model.predict([cleaned_text])[0])
    sentiment = "positif" if label == 1 else "negatif"
    confidence = round(float(proba[label]), 4)
    return {"sentiment": sentiment, "confidence": confidence}
