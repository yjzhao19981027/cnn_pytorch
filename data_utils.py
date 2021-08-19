import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# 下载CIFAR-10数据集到当前data文件夹中
def get_train_loader():
    train_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)


    # 数据加载准备 (开启数据加载的线程和队列).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=4,  # 该参数表示每次读取的批样本个数
                                               shuffle=True)  # 该参数表示读取时是否打乱样本顺序
    return train_loader

def get_test_loader():
    test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                                train=False,
                                                download=True,
                                                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)
    return test_loader

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 实际使用时使用下面的方式读取每一批（batch）样本
# for images, labels in train_loader:
#     # 在此处添加训练代码
#     pass