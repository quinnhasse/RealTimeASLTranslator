"""
Latency benchmark for the ASL classifier.

Runs N predictions on synthetic 42-feature landmark vectors and reports
p50 and p95 single-prediction latency in milliseconds.

The model is a scikit-learn RandomForest, so inference runs on CPU only.
GPU availability is detected and reported but does not change the benchmark.

Usage:
    python benchmark.py
    python benchmark.py --model model.p --n 2000 --warmup 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np


FEATURE_DIM = 42  # 21 landmarks * (x, y)


def load_model(model_path: str):
    """Load the pickled model from *model_path*."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)["model"]


def measure_latency(model, n_samples: int, warmup: int) -> dict[str, float]:
    """Benchmark single-sample inference latency.

    Generates random 42-feature vectors and times each predict() call
    individually. Warmup iterations are discarded before collecting
    the timing sample.

    Args:
        model: Fitted sklearn estimator.
        n_samples: Number of timed predictions to collect.
        warmup: Number of warmup predictions to discard.

    Returns:
        Dictionary with keys p50_ms, p95_ms, mean_ms, min_ms, max_ms.
    """
    rng = np.random.default_rng(seed=0)

    # Warmup — gets the model's internal state (tree lookups, caches) hot.
    warmup_data = rng.random((warmup, FEATURE_DIM))
    for i in range(warmup):
        model.predict(warmup_data[i : i + 1])

    # Timed loop — one prediction at a time to match real-time inference.
    latencies_ms: list[float] = []
    bench_data = rng.random((n_samples, FEATURE_DIM))
    for i in range(n_samples):
        t0 = time.perf_counter()
        model.predict(bench_data[i : i + 1])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n_samples": n_samples,
    }


def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is visible."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ASL classifier latency")
    p.add_argument("--model", default="model.p", help="Path to model.p")
    p.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of timed predictions (default: 1000)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup predictions discarded before timing (default: 100)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        model = load_model(args.model)
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    gpu = gpu_available()
    print(f"Device: CPU (sklearn RandomForest — GPU not used)")
    if gpu:
        print("GPU detected but not applicable for sklearn inference")

    print(f"Warmup: {args.warmup} predictions  |  Benchmark: {args.n} predictions")
    results = measure_latency(model, args.n, args.warmup)

    print()
    print(f"p50 latency:  {results['p50_ms']:.3f} ms")
    print(f"p95 latency:  {results['p95_ms']:.3f} ms")
    print(f"mean latency: {results['mean_ms']:.3f} ms")
    print(f"min latency:  {results['min_ms']:.3f} ms")
    print(f"max latency:  {results['max_ms']:.3f} ms")


if __name__ == "__main__":
    main()
