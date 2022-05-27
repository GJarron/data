import torchvision
from torch.utils.tensorboard import SummaryWriter

# 如果想要下载的数据集每一张图片都变为tensor格式，先设计一个tensor的容器
dataset_transform = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor()
])

# 下载数据集的代码如下，如果他检测你已经下载了，就不会再下载一次。然后把那个transform加进来
train_set = torchvision.datasets.CIFAR10(root="./dataset",train = True,transform=dataset_transform,download = True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train = False,transform=dataset_transform,download = True)

"""print(test_set[0])
print(test_set.classes)
img,target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])"""
# print(test_set[0])
writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)# 因为你前面已经把这个图片变为tensor格式了

writer.close()

