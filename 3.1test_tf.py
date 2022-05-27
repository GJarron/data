from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# 主要学习transforms如何使用以及为什么我们需要tensor数据类型
"""
    运行代码后，在命令行，可以输入如下命令打开那个tensorboard
    tensorboard --logdir=logs --port=6007
    会出现链接，你点击他就进去了，注意，这个端口是可以改的。6007这个。
"""
img_path = r'D:\pytorch\dataset\train\ants_image\5650366_e22b7e1065.jpg'
img = Image.open(img_path)# 此时显然不是tensor数据类型，这个时候是PLI类型

# 创建一个写入tensorboard的文件夹
# 你如果后面的跟前面的差距太大，最好先把前面已经创建的log删掉，然后重新创建一个
writer = SummaryWriter("logs")

# 1 下面讲诉如何使用transforms，并将其转化为tensor数据类型
#    为什么要运用这个，因为在神经网络中一定要先转化为tensor数据
tensor_trans = transforms.ToTensor()# 这相当于创建了个转化器
tensor_imgs = tensor_trans(img)# 传入前面为PIL类型数据的图片，然后转化

writer.add_image("Tensor_img",tensor_imgs)

# 然后关闭写入
writer.close()


