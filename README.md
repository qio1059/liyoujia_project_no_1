# CIFAR-10 训练与分类
这是一个实现三层神经网络模型(Three-layer neural network from scratch for CIFAR-10 classification)，进行 CIFAR-10 数据集分类的项目。该项目使用NumPy来实现神经网络，并进行图像分类任务。模型训练过程中采用了数据增强、SGD 优化器、学习率衰减、交叉熵损失和 L2 正则化等技术。

## 项目结构
cifar_nn/  
├── model.py           # 三层神经网络模型定义   
├── train.py           # 训练主逻辑（包含验证）  
├── test.py            # 测试准确率评估  
├── utils.py           # 工具函数（如数据预处理、权重初始化等）  
├── hyperparams.py     # 超参数查找逻辑  
├── visualize.py       # 可视化训练曲线和权重模式  
├── checkpoint/        # 保存模型权重  
├── data/              # 下载并处理后的CIFAR-10数据  
└── main.py            # 训练模型以及可视化结果 

## 数据集

CIFAR-10 数据集可以通过官方页面下载并解压
下载后将数据存储在 data/ 文件夹中。  
并将其解压到名为cifar-10-batches-py的文件夹中  

## 训练模型
python main.py  
训练时会显示每个 epoch 的损失值和验证集准确率，并根据验证集准确率保存最佳模型。
训练完成后，可以使用测试集评估模型的准确率

## 测试模型
python test.py

## 超参数调优
可以使用 hyperparams.py 进行超参数搜索，以选择最优的学习率、批次大小、训练轮数等。
运行以下命令进行超参数搜索：  
python hyperparams.py
