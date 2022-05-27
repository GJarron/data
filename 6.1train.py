"""
    下面是一个完整的训练流程，虽然是拿着最简单的数据集
"""
import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter

# 0 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset",train = True,transform=torchvision.transforms.ToTensor(),
                                         download = True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train = False,transform=torchvision.transforms.ToTensor(),
                                        download = True)
#    判断数据集长度，打印出来看看
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试训练集的长度为：{}".format(test_data_size))

# 1 利用DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 2 创建神经网络来训练.一般神经网络都是放在另一个文件里面，这儿引入调用就行
tudui = Tudui()

# 3 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 4 优化器
#   一般会给优化器设置优化速度。记住1e-2为十的负二次方
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

# 5 设置训练网络中的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加tensorboard可视化观察
writer = SummaryWriter("logs_train")

# 6 开始训练。上面确定了训练十轮，每一轮都会有取出图片，优化梯度的过程，然后将这一轮的训练结果得到后，然后这一轮再去评估一下测试集，
#   看看究竟啥情况
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs,targets = data # 把图片与目标弄出来
        outputs = tudui(imgs)# 放入网络中训练得到结果
        loss = loss_fn(outputs,targets)# 计算得差距值

        # 优化器优化模型
        optimizer.zero_grad()# 降低梯度前，要把优化器梯度清零
        loss.backward()
        optimizer.step()# 进行优化

        total_train_step +=1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)# 显示你的训练次数以及训练次数下你的那个损失值

    # 测试步骤开始
    total_test_loss = 0 # 测试集的总共偏差
    total_accuracy = 0 # 整体准确率的个数
    with torch.no_grad():# 用这个的好处，不用判断梯度了，因为优化在上面优化
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs) # 将测试集的照片经过训练
            loss = loss_fn(outputs,targets) # 测试缺失
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step += 1

    # 模型保存，下面是第一种保存方法，意味着每一轮保存
    # torch.save(tudui,"tudui_{}.pth".format(i))

writer.close()



