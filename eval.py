"""
Eval harness for the ASL classifier.

Loads model.p and data.pickle, evaluates on a held-out test split,
and writes two artifacts to the current directory:

  metrics.json        — overall accuracy + per-class precision/recall/F1
  confusion_matrix.png — 28x28 heatmap

Usage:
    python eval.py
    python eval.py --model model.p --data data.pickle --test-size 0.2 --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S",
    "space", "T", "U", "V", "W", "X", "Y", "Z",
]


def load_model(model_path: str):
    """Load the pickled RandomForest model from *model_path*."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
    if "model" not in model_dict:
        raise KeyError(f"'model' key missing in {model_path}")
    return model_dict["model"]


def load_data(data_path: str):
    """Load the pickled dataset from *data_path*.

    Returns:
        Tuple of (data: np.ndarray, labels: np.ndarray).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    raw_data = data_dict["data"]
    raw_labels = data_dict["labels"]
    keep = [i for i, x in enumerate(raw_data) if hasattr(x, "__len__") and len(x) == 42]
    data = np.asarray([raw_data[i] for i in keep], dtype=np.float64)
    labels = np.asarray([raw_labels[i] for i in keep])
    return data, labels


def run_eval(
    model,
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    out_dir: str,
    random_state: int = 42,
) -> dict:
    """Run evaluation and write output artifacts.

    Args:
        model: Fitted sklearn estimator.
        data: Feature matrix, shape (N, 42).
        labels: Label array, shape (N,).
        test_size: Fraction of data used for evaluation.
        out_dir: Directory where metrics.json and confusion_matrix.png are written.
        random_state: Seed for the train/test split.

    Returns:
        Dictionary with overall accuracy and per-class metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    _, x_test, _, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        shuffle=True,
        stratify=labels,
        random_state=random_state,
    )

    y_pred = model.predict(x_test)

    # Overall accuracy
    accuracy = float(accuracy_score(y_test, y_pred))

    # Per-class report
    present_labels = sorted(set(y_test) | set(y_pred))
    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "overall_accuracy": accuracy,
        "n_test_samples": int(len(y_test)),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in present_labels
            if label in report
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(present_labels)))
    ax.set_yticks(range(len(present_labels)))
    ax.set_xticklabels(present_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(present_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix  (accuracy {accuracy * 100:.1f}%)")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {cm_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASL classifier")
    p.add_argument("--model", default="model.p", help="Path to model.p")
    p.add_argument("--data", default="data.pickle", help="Path to data.pickle")
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used as the test split (default: 0.2)",
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Directory for metrics.json and confusion_matrix.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model = load_model(args.model)
        data, labels = load_data(args.data)
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    run_eval(model, data, labels, args.test_size, args.out_dir)


if __name__ == "__main__":
    main()
