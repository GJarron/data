"""
    通过这个损失值就可以知道你目前的值与预测的值差的有多远
"""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 这个类其实就是一个网络，卷积，最大池，等等不停反复最后得到这么个网络，你对这个网络输入东西，会得到反馈
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()#判断误差的一个函数，有很多，这个很常用

tudui = Tudui()
for data in dataloader:
    imgs,target = data
    outputs = tudui(imgs) # 你最后每一个类别的值
    result_loss = loss(outputs,target)# 那个target代表你希望的目标
    print(result_loss)# 这儿就可以打印出误差了