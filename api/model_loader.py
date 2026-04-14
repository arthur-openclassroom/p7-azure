# api/model_loader.py
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
MAX_LEN = 100

_model = None
_tokenizer = None


def load_artifacts():
    """Load model and tokenizer at startup."""
    global _model, _tokenizer
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model_path = os.path.join(MODEL_DIR, "model")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    _model = load_model(model_path)
    with open(tokenizer_path, "r") as f:
        _tokenizer = tokenizer_from_json(f.read())


def predict_sentiment(cleaned_text: str) -> dict:
    """Predict sentiment from a cleaned tweet text."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if not cleaned_text:
        return {"sentiment": "negatif", "confidence": 0.5}
    seq = _tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    proba = float(_model.predict(padded, verbose=0)[0][0])
    sentiment = "positif" if proba > 0.5 else "negatif"
    confidence = proba if proba > 0.5 else 1 - proba
    return {"sentiment": sentiment, "confidence": round(confidence, 4)}
