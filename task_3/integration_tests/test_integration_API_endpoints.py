"""
Integration tests for the FastAPI endpoints.
Run with: pytest tests/test_api.py

These hit the real app (real model, real DB) via FastAPI's TestClient,
so they require the DB and MLflow artifacts to be reachable, same as
running the app normally.
"""

from starlette.testclient import TestClient as TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "input_schema" in data


def test_predict_single():
    payload = {"order_id": "9e6bc602a2466daa94736f31d1319c5d"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data


def test_predict_batch():
    payload = [
        {"order_id": "9e6bc602a2466daa94736f31d1319c5d"},
        {"order_id": "c0e57db4a4a6ef32aa28911d1c07df81"},
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "prediction" in item
        assert "probability" in item
        assert "model_version" in item
