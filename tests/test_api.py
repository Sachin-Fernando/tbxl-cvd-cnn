from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.model_service import ModelService

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __call__(self, batch, training=False):
        assert batch.shape == (1, 1000, 1)
        return np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32)


def test_health_readiness_and_prediction():
    service = ModelService(metadata_path=ROOT / "model_metadata.json", model=FakeModel())
    app = create_app(model_service=service)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}

        response = client.post("/predict", json={"ecg_signal": [0.0] * 1000})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "MI"
    assert body["model_version"] == "V3"
    assert "not a medical device" in body["disclaimer"]


def test_prediction_rejects_wrong_sample_count():
    service = ModelService(metadata_path=ROOT / "model_metadata.json", model=FakeModel())
    app = create_app(model_service=service)

    with TestClient(app) as client:
        response = client.post("/predict", json={"ecg_signal": [0.0] * 999})

    assert response.status_code == 422
