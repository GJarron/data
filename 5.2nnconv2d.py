"""
    实现卷积层的操作
"""
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 下面便会采用CIFAR10这个数据集，注意已经转化为tensor数据了
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 再采用一次dataloader技术。一次性拼接64张。其他参数默认
dataloader = DataLoader(dataset,batch_size=64)

# 新建一个神经网络的内核
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 那个stride就是你卷积层每一次挪移的步数，而padding则是你的卷积可以往边缘移的步数
        # 下面这个意味着你的卷积名称为conv1,输入的内容为3，输出为6.卷积层为3*3
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("p11")
step = 0
for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)
    # 通过下面可以很清晰的看出来，你输入前的这个图片是管道是3，输入后的管道是6。
    # print(imgs.shape)
    # print(output.shape)
    # 输入的是torch.Size([64, 3, 32, 32])
    writer.add_image("input", imgs, step,dataformats="NCHW")
    # 输出的是torch.Size([64, 6, 30, 30]).由于这个会报错，所以要改一下
    output = torch.reshape(output , (-1, 3, 30, 30))
    writer.add_image("output", output, step,dataformats="NCHW")
    step +=1

writer.close()



