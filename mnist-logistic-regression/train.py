#%% Packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data_prep import CLASS_NAMES, OUTPUT_DIR, X_test, X_train, y_test, y_train
from model import INPUT_SIZE, MODEL_FILENAME, NUM_CLASSES, MNISTLogisticRegression

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
torch.manual_seed(42)
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 40
WEIGHT_DECAY = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =============================================================================
# DATASET
# =============================================================================
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test.values, dtype=torch.long),
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# MODEL, LOSS, OPTIMIZER
# =============================================================================
model = MNISTLogisticRegression().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_params = sum(p.numel() for p in model.parameters())
settings_text = (
    f"Model architecture\n{model}\n"
    f"Input size: {INPUT_SIZE}, output classes: {NUM_CLASSES}\n"
    f"Trainable parameters: {num_params}\n\n"
    f"Training settings\n"
    f"Batch size   : {BATCH_SIZE}\n"
    f"Learning rate: {LEARNING_RATE}\n"
    f"Epochs       : {EPOCHS}\n"
    f"Optimizer    : Adam (weight_decay={WEIGHT_DECAY})\n"
    f"Loss         : CrossEntropyLoss\n"
    f"Seed         : 42\n"
    f"Device       : {DEVICE}\n"
)
print(settings_text)
(OUTPUT_DIR / "model_and_settings.txt").write_text(settings_text, encoding="utf-8")

# =============================================================================
# TRAINING
# =============================================================================
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    avg_loss = running_loss / len(train_dataset)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] Loss: {avg_loss:.4f}")

(OUTPUT_DIR / "training_losses.txt").write_text(
    "\n".join(f"{i + 1},{loss:.6f}" for i, loss in enumerate(train_losses)), encoding="utf-8"
)

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss (cross-entropy)")
plt.title("MNIST: Multiclass Logistic Regression")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
plt.close()

# =============================================================================
# EVALUATION
# =============================================================================
model.eval()
y_true, y_pred, y_conf = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        probabilities = torch.softmax(model(X_batch.to(DEVICE)), dim=1)
        confidence, predictions = probabilities.max(dim=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_conf.extend(confidence.cpu().numpy())
y_true, y_pred, y_conf = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_conf)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, digits=4
)

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
baseline_class = int(baseline.predict(X_test[:1])[0])

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
off_diag = [(cm[a, p], a, p) for a in range(NUM_CLASSES) for p in range(NUM_CLASSES) if a != p]
top_confusions = sorted(off_diag, reverse=True)[:8]

evaluation_text = (
    f"Test accuracy: {accuracy:.4f} ({int((y_true == y_pred).sum())}/{len(y_true)} correct)\n"
    f"Baseline accuracy (always predicts '{baseline_class}'): {baseline_acc:.4f}\n\n"
    f"{report}\n"
    "Most frequent confusions (actual -> predicted: count)\n"
    + "\n".join(f"  {a} -> {p}: {n}" for n, a, p in top_confusions)
    + "\n"
)
print(evaluation_text)
(OUTPUT_DIR / "evaluation.txt").write_text(evaluation_text, encoding="utf-8")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted digit")
plt.ylabel("Actual digit")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()


# =============================================================================
# PREDICTION FIGURES
# =============================================================================
def plot_grid(indices, rows, cols, title, filename):
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows))
    for ax, index in zip(np.atleast_1d(axes).flat, indices):
        ax.imshow(X_test[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        correct = y_true[index] == y_pred[index]
        ax.set_title(
            f"#{index}  True: {y_true[index]}  Pred: {y_pred[index]}\n"
            f"confidence {y_conf[index]:.0%}",
            color="green" if correct else "red", fontsize=10,
        )
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


rng = np.random.default_rng(42)
plot_grid(rng.choice(len(X_test), size=9, replace=False), 3, 3,
          "Nine random test predictions", "predictions.png")

wrong = np.flatnonzero(y_true != y_pred)
if wrong.size:
    chosen = rng.choice(wrong, size=min(3, wrong.size), replace=False)
    plot_grid(chosen, 1, len(chosen), "Incorrect predictions", "incorrect_predictions.png")
    print("Incorrect examples shown (test index):", chosen.tolist())

# =============================================================================
# SAVE MODEL
# =============================================================================
model_path = OUTPUT_DIR / MODEL_FILENAME
torch.save(model.state_dict(), model_path)
print(f"Model saved: {model_path}")
