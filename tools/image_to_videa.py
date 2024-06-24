import os
import imageio

# 获取文件夹中所有图片的文件名
image_folder = '/home/tww/Datasets/real/datasets/v1_vis/'
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# 将图片按文件名排序
images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 读取所有图片并将它们添加到一个列表中
image_list = []
for image in images:
    image_path = os.path.join(image_folder, image)
    image_list.append(imageio.imread(image_path))

# 反转图像列表以倒序生成GIF
image_list.reverse()

# 指定输出GIF文件路径
gif_path = '/home/tww/Datasets/real/datasets/v1_vis/animation_reverse.gif'

# 保存GIF文件
imageio.mimsave(gif_path, image_list)
