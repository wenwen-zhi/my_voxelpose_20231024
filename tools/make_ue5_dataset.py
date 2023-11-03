import glob
import json
import os

import numpy as np

from load_ue_cameras import load_ue_cameras
from ue_utils import transform_unreal_to_stadium

selected_joints = ["head", "neck_01", "upperarm_correctiveRoot_l", "upperarm_correctiveRoot_r",
                   "lowerarm_correctiveRoot_l", "lowerarm_correctiveRoot_r", "middle_metacarpal_l",
                   "middle_metacarpal_r", "pelvis", "thigh_correctiveRoot_l", "thigh_correctiveRoot_r",
                   "calf_kneeBack_l", "calf_kneeBack_r", "ankle_bck_l", "ankle_bck_r", "ankle_fwd_l", "ankle_fwd_r",
                   "foot_l", "foot_r", "littletoe_02_l", "littletoe_02_r", "bigtoe_02_l", "bigtoe_02_r"]


def load_ue_gt(path, selected_joints=None,unit_factor=1):
    with open(path, 'r') as f:
        data = []
        while True:
            frame_idx = -1
            frame_data = []
            while (line := f.readline().strip()):
                parts = line.split()
                frame_idx = int(parts[1])
                joint_name = parts[2]
                x = float(parts[3].split("=")[1])
                y = float(parts[4].split("=")[1])
                z = float(parts[5].split("=")[1])
                k = float("1")

                # 如果 selected_joints 不为None，并且 joint_name 不在 selected_joints 中，则跳过该关节数据。
                if selected_joints is not None and joint_name not in selected_joints:
                    continue
                pos = [x, -y, z]
                # pos[:, 1] *= -1
                pos = transform_unreal_to_stadium(pos)
                pos*=unit_factor
                pos = np.append(pos, k)
                # print("pos:", pos)

                frame_data.append(pos.tolist())
            if frame_idx != -1:
                data.append(frame_data)
            else:
                break
        return data


# 文件路径列表

def process_gt(gt_file_paths,unit_factor=1):
    # 存储所有帧信息的列表
    all_frames = []

    # 打开每个文件并按帧顺序处理
    for file_path in gt_file_paths:
        data = load_ue_gt(file_path, selected_joints,unit_factor=unit_factor)

        # 确保每个文件的帧数相同（可选）
        # print('len(data)：',len(data))
        # print('len(all_frames)：', len(all_frames))
        if len(data) != len(all_frames) and all_frames:
            raise ValueError("Number of frames in each file must be the same.")

        # 合并帧信息到 all_frames 列表中
        for i, frame_data in enumerate(data):
            if len(all_frames) <= i:
                all_frames.append([])  # 创建一个新的帧数据列表
            all_frames[i].append(frame_data)

    actor3D_data = {"actor3D": all_frames}
    return actor3D_data


def save_gt(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


file_paths1 = ["/home/tww/Datasets/ue/val/ath0_run.txt", "/home/tww/Datasets/ue/val/ath1_run.txt", "/home/tww/Datasets/ue/val/ath2_run.txt"]
file_paths2= ["/home/tww/Datasets/ue/train/ath0_run.txt", "/home/tww/Datasets/ue/train/ath1_run.txt", "/home/tww/Datasets/ue/train/ath2_run.txt"]


save_gt(process_gt(file_paths1),"/home/tww/Datasets/ue/val/actorsGT.json")
save_gt(process_gt(file_paths2),"/home/tww/Datasets/ue/train/actorsGT.json")

load_ue_cameras("/home/tww/Datasets/ue/val/camera.txt", "/home/tww/Datasets/ue/val")
load_ue_cameras("/home/tww/Datasets/ue/train/camera.txt", "/home/tww/Datasets/ue/train")
