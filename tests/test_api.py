import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"text": "I love this airline"})
        assert response.status_code == 200

    def test_predict_returns_sentiment(self, client):
        response = client.post("/predict", json={"text": "I love this airline"})
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"] in ["positif", "negatif"]

    def test_predict_returns_confidence(self, client):
        response = client.post("/predict", json={"text": "I love this airline"})
        data = response.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 200

    def test_predict_missing_text(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422
