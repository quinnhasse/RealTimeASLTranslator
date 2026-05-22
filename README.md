# Real-Time ASL Translator

Webcam-based American Sign Language letter translator. MediaPipe extracts 21 hand landmarks per frame; a trained RandomForest classifier predicts the ASL letter in real time.

## Accuracy and latency

| Metric | Value |
|---|---|
| Overall accuracy | ~94–96% on held-out test split |
| p50 inference latency | < 1 ms (CPU, sklearn RandomForest) |
| p95 inference latency | < 2 ms |
| Pipeline bottleneck | MediaPipe landmark extraction (~10–30 ms/frame) |

Run `make eval` to reproduce metrics locally. Run `make benchmark` for latency numbers on your hardware.

## Known failure modes

- **M/N and S/E confusion** — visually similar hand shapes; common swap under poor lighting
- **J and Z not supported** — both require motion; static snapshots produce incorrect predictions
- **Two hands** — only the first detected hand is used; two-hand gestures are out of scope
- **Plain background required** — model trained on white-background images; textured or cluttered backgrounds increase `nothing` predictions
- **Low light** — MediaPipe drops detections below 0.3 confidence in poor lighting

## What it does

- Opens a webcam feed and processes each frame
- MediaPipe Hands detects and tracks hand keypoints (21 landmarks per hand)
- Landmark coordinates are origin-normalized and fed into a RandomForest classifier
- Predicted letter is overlaid on the live video frame
- Rolling buffer debounces noise — a letter registers after 3 consecutive matching frames

## Tech stack

| Component | Library |
|---|---|
| Hand landmark detection | MediaPipe Hands |
| Frame capture | OpenCV |
| Classifier | scikit-learn RandomForestClassifier |
| Eval and benchmarking | eval.py, benchmark.py |

## Setup

**Requirements:** Python 3.9+, a webcam.

```bash
git clone https://github.com/quinnhasse/RealTimeASLTranslator.git
cd RealTimeASLTranslator
pip install -r requirements.txt
```

## Run

```bash
python src/components/camera.py
```

Press `Esc` to quit.

## Eval and benchmarking

```bash
# Writes metrics.json and confusion_matrix.png
make eval

# Reports p50/p95 inference latency
make benchmark

# Run the test suite
make test
```

You need `model.p` and `data.pickle` in the project root for `eval` and `benchmark`. The test suite runs without them — it uses synthetic data.

## Training your own model

```bash
# Collect training data (press a letter key to label the current hand pose)
python src/components/preprocessing.py

# Train the classifier
python src/components/training.py
```

Collected landmark data is saved to `data.pickle`. Labels cover 26 ASL letters (A–Z) plus `nothing` (no hand) and `space`.

## Project structure

```
RealTimeASLTranslator/
├── src/
│   └── components/
│       ├── camera.py          # Real-time inference loop
│       ├── preprocessing.py   # Landmark extraction and data collection
│       ├── training.py        # Model training
│       └── utils.py           # Pure functions: normalization, validation
├── tests/
│   ├── conftest.py            # Shared fixtures (synthetic model + data)
│   ├── test_preprocessing.py  # Normalization and feature extraction tests
│   └── test_classifier.py     # Classifier I/O and eval harness tests
├── eval.py                    # Eval harness → metrics.json + confusion_matrix.png
├── benchmark.py               # Latency benchmark → p50/p95 ms
├── Makefile                   # make eval | make benchmark | make test
├── MODEL_CARD.md              # Dataset, metrics, failure modes
├── model.p                    # Trained RandomForest (pickle)
├── data.pickle                # Collected landmark data (pickle)
└── requirements.txt
```

## CI

Tests run on every push via GitHub Actions. The suite uses synthetic data and does not require `model.p` or `data.pickle`.
