
import os
from tqdm import tqdm

# 你要重命名文件的目录列表
directories = [
    '/home/tww/Datasets/real/20230312/22186119/rgb/',
    '/home/tww/Datasets/real/20230312/22186129/rgb/',
    '/home/tww/Datasets/real/20230312/22186131/rgb/',
    '/home/tww/Datasets/real/20230312/22186132/rgb/'
]

# 遍历提供的每个目录
for directory in directories:
    # 获取目录下的所有文件
    files = os.listdir(directory)
    # 使用tqdm创建进度条
    for filename in tqdm(files, desc=f'Processing {directory}'):
        # 获取文件的全路径
        old_file = os.path.join(directory, filename)
        # 检查是否是文件
        if os.path.isfile(old_file):
            # 创建新文件名（这里只用了文件名的前六个字符，你需要根据你的要求调整）
            new_filename = filename[:6] + os.path.splitext(filename)[-1]
            # 新文件的全路径
            new_file = os.path.join(directory, new_filename)
            # 确保新文件名不会覆盖现有文件
            if not os.path.exists(new_file):
                # 重命名文件
                os.rename(old_file, new_file)
            else:
                print(f"Cannot rename {old_file} to {new_file} - file already exists.")