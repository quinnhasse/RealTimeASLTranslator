"""
Shared utility functions for landmark processing.

These are the pure-function versions of the logic scattered across
preprocessing.py and camera.py — extracted so they can be imported
and tested independently.
"""

from __future__ import annotations

from typing import Sequence


# Number of hand landmarks MediaPipe returns per hand.
NUM_LANDMARKS = 21

# Expected feature vector length (x, y per landmark).
FEATURE_DIM = NUM_LANDMARKS * 2


def normalize_landmarks(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
) -> list[float]:
    """Normalize raw landmark coordinates relative to the bounding-box origin.

    Subtracts the minimum x and y values so the hand is translated to
    the origin. Output length is always 2 * len(x_coords).

    Args:
        x_coords: Raw x values from MediaPipe landmark objects.
        y_coords: Raw y values from MediaPipe landmark objects, same length.

    Returns:
        Flat list alternating (x - min_x, y - min_y) for each landmark.

    Raises:
        ValueError: If x_coords and y_coords have different lengths, or are empty.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError(
            f"x_coords length {len(x_coords)} != y_coords length {len(y_coords)}"
        )
    if len(x_coords) == 0:
        raise ValueError("x_coords is empty")

    min_x = min(x_coords)
    min_y = min(y_coords)

    result: list[float] = []
    for x, y in zip(x_coords, y_coords):
        result.append(x - min_x)
        result.append(y - min_y)
    return result


def extract_features_from_landmarks(hand_landmarks) -> list[float] | None:
    """Convert a MediaPipe HandLandmark object into a normalized feature vector.

    Args:
        hand_landmarks: A MediaPipe ``NormalizedLandmarkList`` (the element
            returned by ``results.multi_hand_landmarks[i]``).

    Returns:
        Flat list of 42 floats (normalized x, y per landmark), or ``None``
        if the landmark list has an unexpected length.
    """
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    if len(x_coords) != NUM_LANDMARKS:
        return None

    return normalize_landmarks(x_coords, y_coords)


def validate_feature_vector(features: list[float]) -> bool:
    """Return True if *features* has the expected dimension (42 floats).

    Args:
        features: Feature vector produced by :func:`extract_features_from_landmarks`.

    Returns:
        True when ``len(features) == FEATURE_DIM``.
    """
    return len(features) == FEATURE_DIM
