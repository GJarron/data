"""
    这个就是可以将你图片集里面的内容合并在一起弄出来。
    前面那个只是单独抓取
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备测试的数据集。值得注意的是，下面这个数据集已经把这个图片转化为tensor数据了
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

# dataloader对数据集的改变就在下面体现出来了。首先是你这个dataloder的来源，然后你一次性合并多少张.drop_last如果是true，那么不足64张的就会舍去
# 合成的64张照片是随机的，如果你下面那个shuffle是true，第一次与第二次是不一样的
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 将其写进去tensorboard，方便观察
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs,targets = data
    writer.add_images("test_data",imgs,step)
    step = step+1

writer.close()
