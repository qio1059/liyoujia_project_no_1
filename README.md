# CIFAR-10 训练与分类
这是一个实现三层神经网络模型(Three-layer neural network from scratch for CIFAR-10 classification)，进行 CIFAR-10 数据集分类的项目。该项目使用NumPy来实现神经网络，并进行图像分类任务。模型训练过程中采用了数据增强、SGD 优化器、学习率衰减、交叉熵损失和 L2 正则化等技术。

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
├── src/                      # 源代码目录
│   ├── dataset.py            # 数据加载和预处理代码
│   ├── model.py              # 三层神经网络模型代码
│   ├── train.py              # 训练和验证代码
│   ├── test.py               # 测试代码
│   ├── utils.py              # 工具函数，如可视化、日志处理等
│   ├── main.py               # 主程序入口，整合训练、验证、测试流程
│
├── checkpoints/              # 训练好的模型权重保存目录
│   ├── best_model.pkl        # 保存的最佳模型权重（通过验证集指标保存）
│
├── experiments/              # 实验结果目录
│   ├── loss_curve.png        # 可视化的训练/验证集损失曲线
│   ├── accuracy_curve.png    # 可视化的验证集准确率曲线
│   ├── hyperparameters.txt   # 记录不同超参数下的实验结果
│
├── README.md                 # 项目说明文档，包含训练和测试的使用说明

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


