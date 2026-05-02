# Real-Time ASL Translator

Webcam-based American Sign Language translator. Uses MediaPipe to extract hand landmarks per frame and a trained classifier to predict ASL letters and common words in real time.

## What it does

- Opens a webcam feed and processes each frame
- MediaPipe Hands detects and tracks hand keypoints (21 landmarks per hand)
- Landmark coordinates are normalized and fed into a CNN/dense classifier
- Predicted letter or word is overlaid on the live video frame
- Maintains a rolling buffer to form words from sequential letter predictions

## Tech stack

| Component | Library |
|---|---|
| Hand landmark detection | [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) |
| Frame capture | OpenCV (`cv2`) |
| Model | TensorFlow / Keras |
| Inference | NumPy |

## Setup

### Prerequisites

- Python 3.9+
- A webcam

### Install

```bash
git clone https://github.com/quinnhasse/RealTimeASLTranslator.git
cd RealTimeASLTranslator
pip install -r requirements.txt
```

### Run

```bash
python translator.py
```

Press `q` to quit the webcam window.

## Training your own model

The model was trained on hand landmark data (not raw images), which makes it fast and resolution-independent.

```bash
# Collect training data — press a letter key to label the current hand pose
python collect_data.py

# Train the classifier
python train.py

# Evaluate on held-out split
python evaluate.py
```

Collected data is saved to `data/landmarks.csv`. Labels correspond to the 26 ASL letters (A–Z) plus a `NOTHING` class for no hand detected.

## Project structure

```
RealTimeASLTranslator/
├── translator.py      # Real-time inference loop
├── collect_data.py    # Landmark data collection script
├── train.py           # Model training
├── evaluate.py        # Accuracy evaluation on test split
├── model/
│   └── asl_model.h5   # Trained Keras model
├── data/
│   └── landmarks.csv  # Collected training data
└── requirements.txt
```

## Requirements

```
mediapipe
opencv-python
tensorflow
numpy
scikit-learn
```

## Notes

- Works best with a plain background and consistent lighting.
- The rolling buffer debounces single-frame noise — a letter registers after appearing in 3 consecutive frames.
- Extend to full words by adding multi-hand gesture sequences to the training set.
