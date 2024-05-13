import json
import os
from glob import glob
import shutil


def process_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        poses = []
        for i in range(0, len(data['shapes']), 23):
            person = []
            for j in range(23):
                person.append(data['shapes'][i + j]['points'][0])
            poses.append(person)
    return poses


def process_camera_dir(camera_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = sorted(glob(os.path.join(camera_dir, '*.json')))
    poses = []

    for img_idx, json_path in enumerate(json_files):
        img_name = "{:06d}.jpg".format(img_idx + 1)
        original_img_path = json_path.replace('.json', '.jpg')
        if os.path.exists(original_img_path):
            shutil.copy(original_img_path, os.path.join(output_dir, img_name))

        frame_poses = process_json(json_path)
        poses.append(frame_poses)

    return poses


def process_datasets(camera_dirs, new_root_dir):
    all_camera_data = {}

    for camera_dir in camera_dirs:
        camera_id = os.path.basename(camera_dir)
        print(f"Processing {camera_id}...")
        output_camera_dir = os.path.join(new_root_dir, camera_id)
        camera_poses = process_camera_dir(camera_dir, output_camera_dir)
        all_camera_data[camera_id] = camera_poses

    pose2d_path = os.path.join(new_root_dir, 'pose2d.json')
    with open(pose2d_path, 'w') as file:
        json.dump(all_camera_data, file, indent=4)


import json


def add_visibility_to_poses(input_file, threshold=10.0):
    """
    为pose2d.json中的每个点添加可见性标记。

    :param input_file: 输入的pose2d.json文件路径。
    :param threshold: 阈值，用于决定点的可见性。当x和y都小于这个阈值时，v设置为0，否则为1。
    """
    # 加载原始的pose2d.json数据
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 遍历每个相机的数据
    for camera_id, frames in data.items():
        for f_idx,frame in enumerate(frames):
            for person in frame:
                for i, point in enumerate(person):
                    x, y = point
                    v = 1 if (x+y)/2> threshold else 0
                    # 更新点，添加可见性标记
                    person[i] = [x, y, v]
            # frames[f_idx] =frame[1:2]

    # 保存修改后的数据回pose2d.json
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
camera_dirs = [
    "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119",
    "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186129",
    "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131",
    "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132",
]
# new_root_dir = "/path/to/new/root/dir"
#
# process_datasets(camera_dirs, new_root_dir)

new_root_dir = "/home/tww/Datasets/real/datasets/v1"  # 新的根目录路径，请替换为实际路径

process_datasets(camera_dirs, new_root_dir)
add_visibility_to_poses(os.path.join(new_root_dir,"pose2d.json"), 80)
