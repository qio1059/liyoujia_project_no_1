# Three-Layer Neural Network for CIFAR-10
![CIFAR-10 Datasets](https://github.com/user-attachments/assets/642090c3-53af-449d-afcf-74f5ca8db477)

This repository contains a three-layer neural network implemented from scratch using NumPy, designed for image classification on the CIFAR-10 dataset.

## Features
- Manual implementation of forward and backward propagation
- Support for SGD optimizer, learning rate scheduling, and L2 regularization
- Hyperparameter tuning and visualization of results

## Directory Structure
```markdown
liyoujia_project_no_1/
│
├── data/                     # 数据存放目录
│   ├── cifar-10-batches-py/  # CIFAR-10 数据集解压后的文件夹（自动生成）
│
├── source/                   # 源代码目录
│   ├── dataset.py            # 数据加载和预处理代码
│   ├── model.py              # 三层神经网络模型代码
│   ├── train.py              # 训练和验证代码
│   ├── test.py               # 测试代码
│   ├── utils.py              # 工具函数，如可视化、日志处理等
│   ├── main.py               # 主程序入口，整合训练、验证、测试流程
│   ├── hyperparams.py        # 调参
│
├── checkpoints/              # 训练好的模型权重保存目录
│   ├── best_model.pkl        # 保存的最佳模型权重（通过验证集指标保存）
│
├── experiments/              # 实验结果目录
│   ├── loss_curve.png        # 可视化的训练/验证集损失曲线
│   ├── accuracy_curve.png    # 可视化的验证集准确率曲线
│   ├── first_layer_weights   # 第一层权重的可视化图像
│
├── README.md                 # 项目说明文档，包含训练和测试的使用说明
```

## Dataset

CIFAR-10 数据集可以通过官方页面下载并解压
   ```bash
   python -c "from dataset import download_and_extract_cifar10; download_and_extract_cifar10()"
```


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/three-layer-nn-cifar10.git
   cd three-layer-nn-cifar10
   
2. Install dependencies:
   ```bash
   pip install numpy matplotlab

3. Train the model:
   ```bash
   python source/main.py

4. Test the model:
   ```bash
   python source/test.py


