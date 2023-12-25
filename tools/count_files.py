import os, glob

def count_files(directory):
    # 获取目录下所有文件夹
    folders = [f for f in glob.glob(directory)]

    for folder in folders:
        file_count = len(os.listdir(folder))  # 获取文件夹中文件数量
        print(f"文件夹 {folder} 中的文件数量为: {file_count}")

# 替换为你的目录路径
directory_path = '/home/tww/Projects/voxelpose-pytorch/data/panoptic-toolbox/16*/hdImgs/*'

count_files(directory_path)