import os

# 指定父文件夹路径
parent_folder = "/home/tww/Datasets/ue/val/"

# 循环处理每个子文件夹
for camera_folder in range(4):  # 假设有4个子文件夹，您可以根据实际情况更改
    folder_path = os.path.join(parent_folder, f"camera{camera_folder}/")
    file_list = os.listdir(folder_path)

    # 循环处理每个文件名
    for filename in file_list:
        if filename.startswith("ath0_run1_3."):
            new_filename = filename.replace("ath0_run1_3.", "ath0_run1.")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print("文件重命名完成。")
