#加载训练好的模型权重，并在测试集上评估模型的分类准确率。

import numpy as np
from model import ThreeLayerNN
from dataset import load_cifar10

def test_model(model_path="checkpoints/best_model.pkl"):
    """加载训练好的模型，并在测试集上评估性能"""
    # 加载 CIFAR-10 数据集
    _, _, test_data, test_labels = load_cifar10()

    # 加载模型权重
    model_params = np.load(model_path, allow_pickle=True).item()

    # 初始化模型
    input_size = 3072
    hidden_size1 = model_params['W1'].shape[1]
    hidden_size2 = model_params['W2'].shape[1]
    output_size = 10
    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, output_size)
    model.params = model_params

    # 测试集预测
    predictions = model.forward(test_data).argmax(axis=1)
    accuracy = np.mean(predictions == test_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    test_model()
