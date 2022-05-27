"""
    如何保存模型，有两种方式。如下
    这两种模型保存，第一种是直接保存，第二种保存的是字典格式。所以后面打开也需要注意
"""
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,直接模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2,模型参数（官方参数）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")