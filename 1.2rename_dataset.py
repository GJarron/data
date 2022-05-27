import os
# 两个文件夹需要你先自己创建，这个的基本逻辑为
# 从两个已经有图片的文件夹哪儿分离出他们的标签文件夹，以及现在自身存放照片信息的文件夹
#
root_dir = r'D:\pytorch\dataset\train'
target_dir = 'bees_image'
img_path = os.listdir(os.path.join(root_dir,target_dir))
label = target_dir.split('_')[0]
out_dir = 'bees_lable'
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as fp:
        fp.write(label)
