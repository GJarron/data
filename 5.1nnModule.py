"""
    神经网络的核心module如何使用
"""
import torch
from torch import nn

# 先定义一个Module类
class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input):
        output = input +1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)