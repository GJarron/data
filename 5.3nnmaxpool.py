"""
    用最大池化这些操作，就是可以将每一张图片的tensor数据转化为一些比较小的内容，等于降维了
"""
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 确定数据集位置，然后变为tensor类型
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
# 下面合并图片。64合并一张
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,input):
        output = self.maxpool1(input)
        return output

writer = SummaryWriter("p12")

tudui = Tudui()

step = 0

for data in dataloader:
    imgs,targets = data
    # 现在这个后面都得加那个玩意儿,dataformats="NCHW"
    writer.add_image("input",imgs,step,dataformats="NCHW")
    output = tudui(imgs)
    writer.add_image("output",output,step,dataformats="NCHW")
    step += 1

writer.close()
