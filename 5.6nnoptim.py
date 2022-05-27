"""
    在前面预测出损失值的情况下，这个是求得他的优化值
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
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)# 这个是设置优化器。然后优化梯度是0.01
# 由于单独的一次优化你可能看不出来什么情况，因此连续优化多轮，把每一次优化之后的结果输出，最后判断结果
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,target = data
        outputs = tudui(imgs) # 你最后每一个类别的值
        result_loss = loss(outputs,target)# 那个target代表你希望的目标
        optim.zero_grad()# 首先需要把优化器梯度变为零
        result_loss.backward()#求出每个节点的梯度.注意单词要拼写对
        optim.step()#对每个节点梯度进行优化
        running_loss = running_loss + result_loss # 输出每一轮你最后的这个值与目标值的差异
    print(running_loss)

