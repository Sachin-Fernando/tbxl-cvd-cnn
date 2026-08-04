"""Model loading, preprocessing, and inference without HTTP concerns."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "model_metadata.json"


class ModelService:
    """Owns the model lifecycle and enforces the published input contract."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        model: Any | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata = self._read_metadata(self.metadata_path)
        configured_path = model_path or os.getenv("MODEL_PATH") or self.metadata["artifact"]
        candidate = Path(configured_path)
        self.model_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        self.labels = tuple(self.metadata["classes"])
        self.expected_samples = int(self.metadata["input"]["samples"])
        self.epsilon = float(self.metadata["preprocessing"]["epsilon"])
        self._model = model
        self._inference_lock = Lock()

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not metadata.get("classes") or not metadata.get("artifact"):
                raise ValueError("metadata must define 'classes' and 'artifact'")
            return metadata
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid model metadata at {path}: {exc}") from exc

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the Keras model once; compilation is unnecessary for inference."""
        if self.is_ready:
            return
        if not self.model_path.is_file():
            raise RuntimeError(f"Model artifact not found: {self.model_path}")

        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as exc:  # TensorFlow raises several loader-specific errors.
            raise RuntimeError(f"Unable to load model artifact {self.model_path}: {exc}") from exc

    def preprocess(self, signal: Sequence[float]) -> np.ndarray:
        """Apply the same per-record z-score normalization used in training."""
        array = np.asarray(signal, dtype=np.float32)
        if array.ndim != 1 or array.size != self.expected_samples:
            raise ValueError(f"ecg_signal must contain exactly {self.expected_samples} samples")
        if not np.isfinite(array).all():
            raise ValueError("ecg_signal must contain only finite numeric values")

        normalized = (array - array.mean()) / (array.std() + self.epsilon)
        return normalized.reshape(1, self.expected_samples, 1)

    def predict(self, signal: Sequence[float]) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        batch = self.preprocess(signal)
        with self._inference_lock:
            raw_output = self._model(batch, training=False)

        probabilities = np.asarray(
            raw_output.numpy() if hasattr(raw_output, "numpy") else raw_output,
            dtype=np.float64,
        ).reshape(-1)

        if probabilities.size != len(self.labels) or not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned an invalid probability vector")

        total = float(probabilities.sum())
        if total <= 0:
            raise RuntimeError("Model returned probabilities with a non-positive sum")
        probabilities = probabilities / total

        predicted_index = int(np.argmax(probabilities))
        return {
            "prediction": self.labels[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                label: float(probability)
                for label, probability in zip(self.labels, probabilities, strict=True)
            },
            "model_version": str(self.metadata["model_version"]),
        }
