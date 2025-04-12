# 辅助函数，例如保存模型、加载模型、绘图等。

import numpy as np
import matplotlib.pyplot as plt

def save_model(model, path="checkpoints/best_model.pkl"):
    """保存模型权重到文件"""
    np.save(path, model.params)
    print(f"Model saved to {path}")

def load_model(path="checkpoints/best_model.pkl"):
    """加载模型权重"""
    return np.load(path, allow_pickle=True).item()

def plot_curves(train_losses, val_losses, val_accuracies, output_dir="experiments"):
    """绘制训练过程的损失曲线和验证准确率曲线"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png")
    print(f"Loss curve saved to {output_dir}/loss_curve.png")

    # 绘制验证准确率曲线
    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/accuracy_curve.png")
    print(f"Accuracy curve saved to {output_dir}/accuracy_curve.png")
