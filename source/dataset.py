# 下载 CIFAR-10 数据集、加载数据并进行预处理

import numpy as np
import pickle
import os
import urllib.request
import tarfile

def download_and_extract_cifar10(data_dir="./data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(file_name):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, file_name)
        print("Download complete.")

    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    return extract_dir

def load_cifar10(data_dir="./data/cifar-10-batches-py"):
    def load_batch(file_path):
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        return batch['data'], batch['labels']

    train_data, train_labels = [], []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        train_data.append(data)
        train_labels.extend(labels)

    test_data, test_labels = load_batch(os.path.join(data_dir, "test_batch"))

    train_data = np.vstack(train_data).astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, np.array(train_labels), test_data, np.array(test_labels)
