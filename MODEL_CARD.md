# Model card: ASL letter classifier

## Model details

| Field | Value |
|---|---|
| Architecture | RandomForestClassifier (scikit-learn) |
| Input | 42-float vector — normalized (x, y) for 21 MediaPipe hand landmarks |
| Output | One of 28 class labels (A–Z, `nothing`, `space`) |
| Serialization | pickle (`model.p`) |
| Framework | scikit-learn |

## Dataset

| Field | Value |
|---|---|
| Source | [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| Size | ~87,000 images across 29 classes |
| Classes | 26 ASL letters (A–Z) + `nothing` + `space` (J and Z excluded — they require motion) |
| Preprocessing | MediaPipe Hands extracts 21 landmarks per image; coordinates are origin-normalized |
| Split | 50 / 50 train / test (stratified) |

## Eval metrics

Run `make eval` to reproduce these on your local copy of the data.

| Metric | Value |
|---|---|
| Overall accuracy | ~94–96% on the held-out split |
| Per-class accuracy | See `metrics.json` produced by `make eval` |
| Confusion matrix | See `confusion_matrix.png` produced by `make eval` |

Accuracy varies by a few points depending on the random split seed.
The model performs best on static letters with distinctive hand shapes
(A, B, C, L, Y) and worst on visually similar ones (M/N, S/E).

## Latency

Measured on CPU (sklearn RandomForest; GPU is not used).
Run `make benchmark` to reproduce.

| Metric | Typical value |
|---|---|
| p50 single-frame latency | < 1 ms |
| p95 single-frame latency | < 2 ms |

The pipeline bottleneck in real-time use is MediaPipe landmark extraction
(~10–30 ms per frame on CPU), not the classifier.

## Known failure modes

**Confusable letter pairs.**
M/N and S/E share similar hand shapes. The classifier frequently swaps
these pairs, especially when lighting flattens the depth cues MediaPipe
uses to separate overlapping fingers.

**Motion letters excluded.**
J and Z require a motion trajectory, not a static pose. They are not
in the label set. Classifying J or Z will produce an incorrect static
prediction.

**Single-hand constraint.**
The preprocessing pipeline extracts only the first detected hand.
Two-hand gestures or cluttered backgrounds where MediaPipe detects the
wrong hand silently degrade accuracy.

**Background sensitivity.**
The training images use plain white backgrounds. Complex or textured
backgrounds reduce MediaPipe detection confidence and increase the rate
of `nothing` predictions.

**Lighting.**
Low contrast between skin and background (low light, strong backlighting)
drops detection confidence below the 0.3 threshold and produces no
prediction.

**Hand size and aspect ratio.**
Origin-normalization removes scale and position, which is intentional,
but extreme aspect ratios (very wide hands, partial hands at frame edge)
produce out-of-distribution feature vectors.

## Intended use

Real-time fingerspelling recognition from a webcam feed. Not intended
for medical, legal, or accessibility-critical applications without
additional validation.

## Out-of-scope uses

- Full ASL grammar (requires gesture sequences and motion, not static
  landmark snapshots)
- Two-handed signs
- Non-ASL sign languages
