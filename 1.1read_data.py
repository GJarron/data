# 从torch常用的工具箱的data里面导入dataset
from torch.utils.data import Dataset
# 使用此方法读取图片
from PIL import Image
# 想要获取所有图片地址用下面的
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        # 在这个类里面，这个函数只要为后面的函数提供一些全局变量
        # 根文件是你的训练集文件夹，然后标签是蚂蚁或者蜜蜂，这两个代表你要训练的标签对象，用os解析出来
        # 然后如何给到全局都能使用你这个解析出来的变量呢？通过self.的方式来实现
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 而图片的文件夹地址，如果是蚂蚁图片那么应该在根文件夹以及蚂蚁名称的目录下，而如果是蜜蜂图片应该是根文件夹加蜜蜂名称’
        self.path = os.path.join(self.root_dir,self.label_dir)
        # 图片名称则在img_path下面,这个需要对文件夹地址下面的内容再进行提取
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        # 此处是一步步提取出文件夹下面图片的地址，猜测是会不断调用，然后idx索引会增加的一个已经写好的函数
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        # 下面运用Image方法去打开图片
        img = Image.open(img_item_path)
        label = self.label_dir
        # 此函数需要返回图片与标签
        return img,label

    def __len__(self):
        # 返回整个列表的长度
        return len(self.img_path)

# 下面实例化一个蚂蚁类
root_dir = 'firstdataset/train'
ants_label_dir = 'ants_image'
ants_dataset = MyData(root_dir,ants_label_dir)
# 同理可以弄个蜜蜂的
bees_label_dir = 'bees_image'
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset+bees_dataset