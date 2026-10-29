"""Predict digits with the saved model -- no retraining.

Usage:
    python predict.py               # 5 random test images
    python predict.py my_digit.png  # your own image: dark digit on light paper
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import MODEL_FILENAME, MNISTLogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / MODEL_FILENAME


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")
    model = MNISTLogisticRegression()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(model, pixels):
    """pixels: float32 array of shape (n, 784), values scaled to 0-1."""
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(pixels)), dim=1)
    confidence, digits = probabilities.max(dim=1)
    return digits.numpy(), confidence.numpy()


def image_to_pixels(image_path):
    # MNIST digits are white on black, 28x28, so invert a dark-on-light photo.
    image = Image.open(image_path).convert("L").resize((28, 28))
    pixels = 255.0 - np.asarray(image, dtype=np.float32)
    return (pixels / 255.0).reshape(1, 784)


if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) == 2:
        digits, confidence = predict(model, image_to_pixels(sys.argv[1]))
        print(f"{sys.argv[1]}: predicted {digits[0]} ({confidence[0]:.1%})")
    else:
        from data_prep import X_test, y_test

        indices = np.random.default_rng().choice(len(X_test), size=5, replace=False)
        digits, confidence = predict(model, X_test[indices])
        for index, digit, conf in zip(indices, digits, confidence):
            mark = "OK " if digit == y_test.iloc[index] else "BAD"
            print(f"[{mark}] test #{index}: true {y_test.iloc[index]}, predicted {digit} ({conf:.1%})")
