import torch.nn as nn

INPUT_SIZE = 784   # 28 x 28 pixels
NUM_CLASSES = 10   # digits 0-9
MODEL_FILENAME = "mnist_logistic_model.pth"


class MNISTLogisticRegression(nn.Module):
    """Multiclass logistic regression: one linear layer, no hidden layers."""

    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Raw logits; CrossEntropyLoss applies log-softmax internally.
        return self.linear(x)
