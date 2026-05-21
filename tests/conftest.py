"""
Shared pytest fixtures.

Creates a minimal trained RandomForest and synthetic landmark data
so tests run without model.p or data.pickle on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


# Labels used in the real classifier.
LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S",
    "space", "T", "U", "V", "W", "X", "Y", "Z",
]

FEATURE_DIM = 42  # 21 landmarks * (x, y)
N_CLASSES = len(LABELS)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic RNG for all tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def synthetic_data(rng) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset: 10 samples per class, 42 features, values in [0, 1)."""
    samples_per_class = 10
    X = rng.random((samples_per_class * N_CLASSES, FEATURE_DIM))
    y = np.repeat(LABELS, samples_per_class)
    return X, y


@pytest.fixture(scope="session")
def trained_model(synthetic_data) -> RandomForestClassifier:
    """A small RandomForest trained on synthetic data.

    n_estimators=5 keeps the fixture fast while still letting the model
    produce predictions with the correct label set.
    """
    X, y = synthetic_data
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    return clf
