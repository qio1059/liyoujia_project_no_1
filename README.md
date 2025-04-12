# CIFAR-10 训练与分类
这是一个实现三层神经网络模型(Three-layer neural network from scratch for CIFAR-10 classification)，进行 CIFAR-10 数据集分类的项目。该项目使用NumPy来实现神经网络，并进行图像分类任务。模型训练过程中采用了数据增强、SGD 优化器、学习率衰减、交叉熵损失和 L2 正则化等技术。

## Features
- Manual implementation of forward and backward propagation
- Support for SGD optimizer, learning rate scheduling, and L2 regularization
- Hyperparameter tuning and visualization of results

## Directory Structure
liyoujia_project_no_1/
├── three-layer-nn-cifar10/
│   ├── data/                # Data storage directory
│       ├── cifar-10-batches-py # Extracted CIFAR-10 dataset files (auto-generated)
│   ├── source/              # Source code directory
│       ├── dataset.py       # Data loading and preprocessing code
│       ├── model.py         # Three-layer neural network model code
│       ├── train.py         # Training and validation code
│       ├── test.py          # Testing code
│       ├── utils.py         # Utility functions (e.g., visualization, logging)
│       ├── main.py          # Main entry point for training, validation, and testing
│   ├── checkpoints/         # Directory for saving trained model weights
│       ├── best_model.pkl   # Best model weights (saved by validation metrics)
│   ├── outputs/             # Experiment results directory
│       ├── loss_curve.png   # Training/validation loss curve visualization
│       ├── accuracy_curve.png # Validation accuracy curve visualization
│       ├── hyperparameters.txt # Experimental results under different hyperparameters
│   ├── README.md            # Project documentation with training/testing instructions

## 数据集
CIFAR-10 数据集可以通过官方页面下载并解压
下载后将数据存储在 data/ 文件夹中。  
并将其解压到名为cifar-10-batches-py的文件夹中  

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/three-layer-nn-cifar10.git
   cd three-layer-nn-cifar10
   
3. Install dependencies:
   pip install numpy matplotlab

4. Train the model:
   python source/main.py

5. Test the model:
   python source/test.py


