"""
    模型加载有两种方式
"""
import torch
import torchvision

# 第一种加载方式，直接加载，也是加载第一种保存方式加载的东西
model = torch.load("vgg16_method1.pth")
print(model)

# 第二种方式，需要先创建模型格式，然后加载字典数据,最好用他。用方式1，然后保存会有陷阱
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)