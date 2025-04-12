训练和验证代码，用于训练模型并保存最佳权重。

import numpy as np
from model import ThreeLayerNN
from dataset import load_cifar10

def train_model():
    # Hyperparameters
    input_size = 3072
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10
    learning_rate = 0.01
    reg = 0.001
    epochs = 20
    batch_size = 64

    # Load data
    train_data, train_labels, val_data, val_labels = load_cifar10()

    # Initialize model
    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)

    # Training loop
    for epoch in range(epochs):
        # Shuffle and batch training data
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data, train_labels = train_data[indices], train_labels[indices]

        for i in range(0, train_data.shape[0], batch_size):
            X_batch = train_data[i:i+batch_size]
            y_batch = train_labels[i:i+batch_size]

            # Forward and backward pass
            model.forward(X_batch)
            model.backward(X_batch, y_batch, learning_rate, reg)

        # Validate
        val_preds = model.forward(val_data).argmax(axis=1)
        acc = np.mean(val_preds == val_labels)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {acc:.4f}")
