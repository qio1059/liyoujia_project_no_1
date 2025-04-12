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
├── outputs/                  # 可视化结果
│   ├── loss curve & validation accuracy.png      
│   ├── weights_Layer1.png   
│   ├── weights_Layer2.png   
│   ├── weights_Layer3.png   
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

## 模型权重下载
训练好的模型权重已上传至云盘：
- **百度网盘**: [下载链接]（https://pan.baidu.com/s/1SJA3qrCe_04EergyW1OA3A?pwd=cr3c 提取码: cr3c）

使用方式：
```python
import pickle

with open('best_model.pkl', 'rb') as f:
    weights = pickle.load(f)
    
# 加载到模型中
model.fc1.W, model.fc1.b = weights['W1'], weights['b1']
model.fc2.W, model.fc2.b = weights['W2'], weights['b2']
model.fc3.W, model.fc3.b = weights['W3'], weights['b3']
```
