"""
    这儿主要讲诉一些常用的transforms函数
    主要把握住，输入是什么？输出是什么？也就是这个函数能用来做什么
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 第一步先创建一个新的tensorboard的log文件夹
writer = SummaryWriter("logs")

# 然后导入你要尝试改变的图片
# 前面都是先给出img的地址，然后再open导入，这个就是直接导入进来
img = Image.open(r'D:\pytorch\dataset\train\ants_image\24335309_c5ea483bb8.jpg')

# 1 第一个方法，ToTensor。可以将图片PIL数据转化为tensor数据
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# 2 第二个方法。Normalize。归一化。你图片是三维的，你确定xyz后会得到一个值，这个值就是这个输入，怎么归一化呢？
#   输入减去均值除以标准差就是归一化后的值
#   可以放心的一件事是，你每次改变的只是容器本身的情况
print(img_tensor[0][0][0])# 归一化前的情况
trans_norm = transforms.Normalize([1,3,7],[9,10,11])# 需要设置均值与标准差，三个三个维度，因此就是这个样子
img_norm = trans_norm(img_tensor)# 图片只能传入tensor的数据才能归一化，这个很显然，之前PIL时数据不是这个数组等等格式
print(img_norm[0][0][0])# 归一化后的情况
writer.add_image("Normalize",img_norm,2)# 这个是弄出归一化后的图片.后面那个需要，你可以改为你的这个新图片放哪儿

# 3 第三个方法，Resize，等比例缩小或者放大。
trans_size = transforms.Resize((555,555))#图片大小改为512*512
img_resize = trans_size(img_tensor)# 现如今可以直接传入tensor的数据格式，之前只能传入PIL的数据格式,但是，这样你没有办法通过打印得到他缩放后大小
writer.add_image("Resize",img_resize,1)

# 4 第四个方法，RandomCrop。随机裁减
trans_random = transforms.RandomCrop(300)# 随机裁剪出一个300*300的图片。用(500,1000)那就是指定高与宽
trans_compose = transforms.Compose([trans_random,trans_totensor])# compose是一个拼接函数。等于将前面的随机裁剪与转化为tensor函数结合起来了
for i in range(10):
    img_crop = trans_compose(img)# 用拼接函数对这个img同时作用
    writer.add_image("RandomCrop",img_crop,i)

# 然后关闭。一定要最后关闭，不要随随便便关闭
writer.close()
