"""
    一张图片可以经过很多步骤，什么卷积，最大池等操作，最后再拿到神经网络里面训练
"""
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# 这个类其实就是一个网络，卷积，最大池，等等不停反复最后得到这么个网络，你对这个网络输入东西，会得到反馈
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 由于卷积、最大化值这些步骤的参数都不一样，因为函数也会有点不同
        # 卷积的指标是这样的。上一张图片为3@32*32，下一张为32@32*32.那个5是默认等于这么多
        self.conv1 = Conv2d(3,32,5,padding=2)
        # 最大化池的指标是这样的。上一张图片32@32*32.下一张图片是32@16*16
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        #
        self.flatten = Flatten()
        # 转化为线型的函数如下，第一个。上一张图片为64@4*4，下一张图片为64
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)
        # 如果上面你觉得太长，也可以这么用
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
        """x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)"""
        x = self.model1(x)
        return x

tudui = Tudui()
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
# 通过下面这个，可以看到输入在这个对象里面怎么走
writer = SummaryWriter("p13")
writer.add_graph(tudui,input)
writer.close()
