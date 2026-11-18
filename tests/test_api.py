from fastapi.testclient import TestClient

from src.app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


def test_predict_billing_ticket():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "I was charged twice"})
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Billing"
        assert data["priority"] in {"Low", "Medium", "High"}
        assert 0.0 <= data["confidence"] <= 1.0


def test_predict_technical_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Application crashes immediately on login with 500 error"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Technical"
        assert "priority" in data
        assert isinstance(data["confidence"], float)


def test_predict_empty_text_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422


def test_predict_whitespace_text_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "    "})
        assert response.status_code == 422


def test_predict_missing_field_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422