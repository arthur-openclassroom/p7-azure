# tests/conftest.py
import pytest
from unittest.mock import patch


def mock_predict_sentiment(text):
    return {"sentiment": "positif", "confidence": 0.95}


@pytest.fixture(autouse=True)
def mock_model():
    with patch("api.model_loader.load_artifacts"), \
         patch("api.main.predict_sentiment", side_effect=mock_predict_sentiment):
        yield
