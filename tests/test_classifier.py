"""
Tests for classifier I/O: input shape, output types, and label set.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import LABELS, FEATURE_DIM, N_CLASSES


class TestClassifierIO:
    def test_predict_returns_array(self, trained_model, rng):
        X = rng.random((5, FEATURE_DIM))
        preds = trained_model.predict(X)
        assert hasattr(preds, "__len__")
        assert len(preds) == 5

    def test_predict_single_sample(self, trained_model, rng):
        x = rng.random((1, FEATURE_DIM))
        pred = trained_model.predict(x)
        assert len(pred) == 1

    def test_predictions_are_known_labels(self, trained_model, rng):
        X = rng.random((50, FEATURE_DIM))
        preds = trained_model.predict(X)
        for p in preds:
            assert p in LABELS, f"Unexpected label: {p!r}"

    def test_predict_proba_shape(self, trained_model, rng):
        X = rng.random((10, FEATURE_DIM))
        proba = trained_model.predict_proba(X)
        assert proba.shape == (10, N_CLASSES)

    def test_predict_proba_sums_to_one(self, trained_model, rng):
        X = rng.random((10, FEATURE_DIM))
        proba = trained_model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-6)

    def test_predict_proba_non_negative(self, trained_model, rng):
        X = rng.random((10, FEATURE_DIM))
        proba = trained_model.predict_proba(X)
        assert (proba >= 0).all()

    def test_wrong_feature_dim_raises(self, trained_model):
        # sklearn raises ValueError on mismatched feature count
        X_bad = np.random.rand(1, FEATURE_DIM - 1)
        with pytest.raises(ValueError):
            trained_model.predict(X_bad)

    def test_batch_predict_shape(self, trained_model, rng):
        batch_size = 32
        X = rng.random((batch_size, FEATURE_DIM))
        preds = trained_model.predict(X)
        assert len(preds) == batch_size

    def test_deterministic_prediction(self, trained_model):
        # Same input must produce same output.
        x = np.ones((1, FEATURE_DIM)) * 0.5
        p1 = trained_model.predict(x)
        p2 = trained_model.predict(x)
        assert p1[0] == p2[0]

    def test_classes_attribute_matches_labels(self, trained_model):
        # The model's known classes must be a subset of our LABELS list.
        model_classes = set(trained_model.classes_)
        assert model_classes.issubset(set(LABELS)), (
            f"Unexpected classes: {model_classes - set(LABELS)}"
        )


class TestEvalHarness:
    """Integration tests for the eval module logic (no file I/O)."""

    def test_run_eval_writes_metrics(self, trained_model, synthetic_data, tmp_path):
        import json
        from eval import run_eval

        X, y = synthetic_data
        metrics = run_eval(
            model=trained_model,
            data=X,
            labels=y,
            test_size=0.3,
            out_dir=str(tmp_path),
        )

        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            loaded = json.load(f)
        assert "overall_accuracy" in loaded
        assert "per_class" in loaded
        assert 0.0 <= loaded["overall_accuracy"] <= 1.0

    def test_run_eval_writes_confusion_matrix_png(
        self, trained_model, synthetic_data, tmp_path
    ):
        from eval import run_eval

        X, y = synthetic_data
        run_eval(
            model=trained_model,
            data=X,
            labels=y,
            test_size=0.3,
            out_dir=str(tmp_path),
        )

        cm_file = tmp_path / "confusion_matrix.png"
        assert cm_file.exists()
        assert cm_file.stat().st_size > 0

    def test_load_model_missing_file(self, tmp_path):
        from eval import load_model

        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nonexistent.p"))

    def test_load_data_missing_file(self, tmp_path):
        from eval import load_data

        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.pickle"))

    def test_load_data_filters_two_hand_rows(self, tmp_path):
        import pickle

        from eval import load_data

        rng = np.random.default_rng(0)
        rows = [rng.random(FEATURE_DIM).tolist() for _ in range(10)]
        labels = ["A"] * 10
        # Inject a few two-hand (84-feature) rows that must be filtered out.
        rows.insert(3, rng.random(FEATURE_DIM * 2).tolist())
        labels.insert(3, "B")
        rows.append(rng.random(FEATURE_DIM * 2).tolist())
        labels.append("C")

        data_path = tmp_path / "mixed.pickle"
        with open(data_path, "wb") as f:
            pickle.dump({"data": rows, "labels": labels}, f)

        data, loaded_labels = load_data(str(data_path))
        assert data.shape == (10, FEATURE_DIM)
        assert loaded_labels.shape == (10,)
        assert "B" not in loaded_labels
        assert "C" not in loaded_labels
        assert set(loaded_labels) == {"A"}


class TestBenchmark:
    """Tests for benchmark latency measurement."""

    def test_measure_latency_returns_expected_keys(self, trained_model):
        from benchmark import measure_latency

        result = measure_latency(trained_model, n_samples=20, warmup=5)
        for key in ("p50_ms", "p95_ms", "mean_ms", "min_ms", "max_ms", "n_samples"):
            assert key in result

    def test_p95_gte_p50(self, trained_model):
        from benchmark import measure_latency

        result = measure_latency(trained_model, n_samples=50, warmup=5)
        assert result["p95_ms"] >= result["p50_ms"]

    def test_latencies_positive(self, trained_model):
        from benchmark import measure_latency

        result = measure_latency(trained_model, n_samples=20, warmup=5)
        assert result["p50_ms"] > 0
        assert result["p95_ms"] > 0

    def test_n_samples_reported(self, trained_model):
        from benchmark import measure_latency

        result = measure_latency(trained_model, n_samples=30, warmup=5)
        assert result["n_samples"] == 30
