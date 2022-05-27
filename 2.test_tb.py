from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
"""
    通过这个东西可以很迅速知道这个过程中给模型提供了哪些数据
"""
# 也就是把你运行的结果放到这个文件夹里面
write = SummaryWriter("logs")

image_path = r'D:\pytorch\dataset\train\bees_image\29494643_e3410f0d37.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))

# 需要输入进行标签，图片，还有步数。但是对于图片有格式要求。必须为numpy.array或torch.tensor或字符、blobname
# 但是对于你上面的这个numpy.array格式还需要加一个指令
# 如果你后面添加的图片不改训练的标签，只是改了步长，是不会覆盖前面已经添加的图片的。
# 实际上往里面加入tensor格式才是最好的。上面这个方法并不是那么的好
write.add_image("test",img_array,1, dataformats = 'HWC')

# 这个方法就是给你显示一个表盘，表盘的x轴是步数，表盘的y轴是显示的值
for i in range(100):
    # “y=x，是你的title”，而两个i，第一个是显示的值y，第二个是步长x
    # 如果你的title不改，你生成的新内容就在这个上面了。然后挤在一起。会出现拟合现象，解决办法把log文件删掉
    write.add_scalar("y=3x", 3*i, i)

write.close()