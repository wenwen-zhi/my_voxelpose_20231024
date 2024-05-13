from PIL import Image
import os


def visualize_images(input_dir, output_path, row_spacing=10, col_spacing=10, small_image_size=None,
                     max_output_width=None,
                     rows = 4,
                     cols = 4,
                     image_format = ".jpg"
                     ):
    # 获取目录下的所有子目录和文件
    dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(dirs)
    images = []  # 存储所有图像的列表
    for d in sorted(dirs):  # 按照文件夹名称排序
        sub_dir = os.path.join(input_dir, d)
        files = [f for f in os.listdir(sub_dir) if f.endswith(image_format)]
        for f in sorted(files):  # 按照文件名称排序
            image_path = os.path.join(sub_dir, f)
            image = Image.open(image_path)
            images.append(image)
            print(image.size)
    # 计算行数和列数
    num_images = len(images)
    if num_images < rows * cols:
        rows = (num_images + cols-1) //cols

    # 确定小图的大小
    if small_image_size is None and images:
        small_image_size = images[0].size
    elif small_image_size is None:
        small_image_size = (224, 224)

    # 创建一个新的白色背景大图
    row_height = small_image_size[1] + row_spacing
    col_width = small_image_size[0] + col_spacing

    print(row_spacing, col_spacing, small_image_size)

    print(cols * col_width - col_spacing, rows * row_height - row_spacing)
    big_image = Image.new('RGB', (cols * col_width - col_spacing, rows * row_height - row_spacing), (255, 255, 255))

    # 将所有图像拼接到大图中
    for i, img in enumerate(images):
        col_idx = i % cols
        row_idx = i // cols
        x_offset = col_idx * col_width
        y_offset = row_idx * row_height
        print(x_offset, y_offset)
        big_image.paste(img.resize(small_image_size), (x_offset, y_offset))

    # 检查大图宽度是否超出最大宽度，如果超出，则按比例缩放
    if max_output_width is not None and big_image.width > max_output_width:
        ratio = max_output_width / big_image.width
        new_height = int(big_image.height * ratio)
        big_image = big_image.resize((max_output_width, new_height))

    # 保存大图
    big_image.save(output_path)


# 示例用法
input_dir = './imgs/'
output_path = './imgs/result.jpg'
visualize_images(input_dir, output_path, row_spacing=0, col_spacing=0, max_output_width=2048, rows=1, cols=3, image_format=".png")
