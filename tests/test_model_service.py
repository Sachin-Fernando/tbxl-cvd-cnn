from pathlib import Path

import numpy as np
import pytest

from app.model_service import ModelService

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self, output: list[float]) -> None:
        self.output = np.asarray([output], dtype=np.float32)
        self.last_batch = None

    def __call__(self, batch, training=False):
        self.last_batch = batch
        assert training is False
        return self.output


def make_service(model: FakeModel) -> ModelService:
    return ModelService(metadata_path=ROOT / "model_metadata.json", model=model)


def test_preprocess_matches_training_contract():
    service = make_service(FakeModel([0.1, 0.8, 0.1]))
    signal = np.linspace(-2.0, 3.0, 1000)

    batch = service.preprocess(signal)

    assert batch.shape == (1, 1000, 1)
    assert float(batch.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(batch.std()) == pytest.approx(1.0, abs=1e-6)


def test_predict_maps_probabilities_to_published_class_order():
    service = make_service(FakeModel([0.05, 0.90, 0.05]))

    result = service.predict([0.0] * 1000)

    assert result["prediction"] == "NORM"
    assert result["confidence"] == pytest.approx(0.9)
    assert list(result["probabilities"]) == ["MI", "NORM", "STTC"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("signal", [[0.0] * 999, [0.0] * 1001, [float("nan")] * 1000])
def test_preprocess_rejects_invalid_signal(signal):
    service = make_service(FakeModel([0.1, 0.8, 0.1]))

    with pytest.raises(ValueError):
        service.preprocess(signal)
