"""
Tests for landmark normalization and feature extraction utilities.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.components.utils import (
    normalize_landmarks,
    validate_feature_vector,
    FEATURE_DIM,
    NUM_LANDMARKS,
)


class TestNormalizeLandmarks:
    def test_output_length(self):
        x = [0.1, 0.3, 0.5]
        y = [0.2, 0.4, 0.6]
        result = normalize_landmarks(x, y)
        assert len(result) == 6

    def test_min_x_becomes_zero(self):
        x = [0.2, 0.5, 0.8]
        y = [0.1, 0.3, 0.5]
        result = normalize_landmarks(x, y)
        # x values at indices 0, 2, 4
        x_out = result[0::2]
        assert min(x_out) == pytest.approx(0.0)

    def test_min_y_becomes_zero(self):
        x = [0.2, 0.5, 0.8]
        y = [0.1, 0.3, 0.5]
        result = normalize_landmarks(x, y)
        y_out = result[1::2]
        assert min(y_out) == pytest.approx(0.0)

    def test_relative_differences_preserved(self):
        x = [0.1, 0.4, 0.7]
        y = [0.0, 0.3, 0.6]
        result = normalize_landmarks(x, y)
        # After normalization, x diffs should be 0.3 apart
        assert result[2] - result[0] == pytest.approx(0.3)
        assert result[4] - result[2] == pytest.approx(0.3)

    def test_all_same_values(self):
        x = [0.5, 0.5, 0.5]
        y = [0.3, 0.3, 0.3]
        result = normalize_landmarks(x, y)
        # All normalized to zero
        assert all(v == pytest.approx(0.0) for v in result)

    def test_single_landmark(self):
        x = [0.7]
        y = [0.2]
        result = normalize_landmarks(x, y)
        assert result == pytest.approx([0.0, 0.0])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            normalize_landmarks([0.1, 0.2], [0.3])

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_landmarks([], [])

    def test_full_hand_landmark_count(self):
        rng = np.random.default_rng(1)
        x = rng.random(NUM_LANDMARKS).tolist()
        y = rng.random(NUM_LANDMARKS).tolist()
        result = normalize_landmarks(x, y)
        assert len(result) == FEATURE_DIM

    def test_output_values_non_negative(self):
        rng = np.random.default_rng(2)
        x = rng.random(NUM_LANDMARKS).tolist()
        y = rng.random(NUM_LANDMARKS).tolist()
        result = normalize_landmarks(x, y)
        assert all(v >= 0.0 for v in result)

    def test_interleaved_xy_ordering(self):
        # Output alternates x, y, x, y, ...
        x = [0.1, 0.3]
        y = [0.2, 0.5]
        result = normalize_landmarks(x, y)
        # First pair: (0.0, 0.0), second pair: (0.2, 0.3)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.2)
        assert result[3] == pytest.approx(0.3)


class TestValidateFeatureVector:
    def test_correct_length_passes(self):
        features = [0.0] * FEATURE_DIM
        assert validate_feature_vector(features) is True

    def test_short_vector_fails(self):
        features = [0.0] * (FEATURE_DIM - 1)
        assert validate_feature_vector(features) is False

    def test_long_vector_fails(self):
        features = [0.0] * (FEATURE_DIM + 1)
        assert validate_feature_vector(features) is False

    def test_empty_vector_fails(self):
        assert validate_feature_vector([]) is False
