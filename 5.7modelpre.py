"""
   这个是用来用别人调好的神经网络，但是他的数据集很大，所以我们没有办法通过下载他的数据集
   然后再下载他的神经网络模型来用的方式.通过修改变为自己想要的。
"""
import torchvision
from torch import nn

# 所以我们采用直接下载他的网络模型，有两种方式，一种是pretrained为true，一种为false
# True的带上训练好的参数，false的不会
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)#一般关注最后一层，最后一层是1000，因此就分了1000类
# 然后可以通过再加一层的方式，让1000个类变为10个类
vgg16_true.add_module('add_linear',nn.Linear(1000,10))
# 如果想加在那个classifier里面，可以
# vgg16_true.classifier.add_module(xxx)
print(vgg16_true)#可以看出来，这就是10个类
# 如果不想要加入，只是单纯修改，这样做
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
