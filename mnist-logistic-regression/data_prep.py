#%% Packages
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# PROJECT DIRECTORY
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# DOWNLOAD KAGGLE DATASET (cached after the first run)
# =============================================================================
DATA_DIR = Path(kagglehub.dataset_download("oddrationale/mnist-in-csv"))
print("Dataset folder:", DATA_DIR)

# =============================================================================
# CLASS NAMES: label k is the digit k
# =============================================================================
CLASS_NAMES = [str(digit) for digit in range(10)]


# =============================================================================
# LOAD CSV FUNCTION
# =============================================================================
def load_csv(filename):
    file_path = DATA_DIR / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")

    dataframe = pd.read_csv(file_path)
    # Each row: one label + 784 pixel values.
    if "label" not in dataframe.columns or dataframe.shape[1] != 785:
        raise ValueError(f"{filename} must contain a label column and 784 pixel columns.")
    if dataframe.empty:
        raise ValueError(f"{filename} contains no data.")
    if not dataframe["label"].isin(range(10)).all():
        raise ValueError(f"{filename} contains invalid class labels.")

    pixel_columns = dataframe.drop(columns="label")
    X = pixel_columns.to_numpy(dtype=np.float32)
    y = dataframe["label"].astype("int64")

    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 255:
        raise ValueError(f"{filename} must contain pixel values from 0 to 255.")
    X = X / 255.0
    return X, y, pixel_columns.columns


# =============================================================================
# PREPARE TRAINING AND TEST DATA (supplied split, never mixed)
# =============================================================================
X_train, y_train, train_columns = load_csv("mnist_train.csv")
X_test, y_test, test_columns = load_csv("mnist_test.csv")

if not train_columns.equals(test_columns):
    raise ValueError("Training and test pixel columns must have the same order.")


# =============================================================================
# REPORT + SAMPLE FIGURE (only when run directly)
# =============================================================================
def save_dataset_summary():
    lines = [
        "Dataset Shapes",
        f"X_train: {X_train.shape}",
        f"y_train: {y_train.shape}",
        f"X_test : {X_test.shape}",
        f"y_test : {y_test.shape}",
        "",
        "Image size: 28 x 28 grayscale = 784 features",
        f"Classes: {CLASS_NAMES}",
        f"Pixel range after scaling: {X_train.min():.1f} to {X_train.max():.1f}",
        "",
        "Training class distribution",
    ]
    for digit, count in y_train.value_counts().sort_index().items():
        lines.append(f"  {digit}: {count} images")
    lines.append("Test class distribution")
    for digit, count in y_test.value_counts().sort_index().items():
        lines.append(f"  {digit}: {count} images")

    text = "\n".join(lines)
    print(text)
    (OUTPUT_DIR / "dataset_summary.txt").write_text(text, encoding="utf-8")


def save_sample_digits():
    fig, axes = plt.subplots(2, 5, figsize=(10, 5.2))
    for digit, ax in zip(range(10), axes.flat):
        index = int(np.flatnonzero(y_train.to_numpy() == digit)[0])
        ax.imshow(X_train[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Label: {digit}")
        ax.axis("off")
    fig.suptitle("One training sample per digit")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_digits.png", dpi=150)
    plt.close()
    print("Saved:", OUTPUT_DIR / "sample_digits.png")


if __name__ == "__main__":
    save_dataset_summary()
    save_sample_digits()
