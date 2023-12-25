import glob
import json
import os

import numpy as np

from load_ue_cameras import load_ue_cameras
from ue_utils import transform_unreal_to_stadium

selected_joints = ["head", "neck_01", "upperarm_correctiveRoot_l", "upperarm_correctiveRoot_r",
                   "lowerarm_correctiveRoot_l", "lowerarm_correctiveRoot_r", "middle_metacarpal_l",
                   "middle_metacarpal_r", "pelvis", "thigh_correctiveRoot_l", "thigh_correctiveRoot_r",
                   "calf_l", "calf_r", "ankle_bck_l", "ankle_bck_r", "ankle_fwd_l", "ankle_fwd_r",
                   "foot_l", "foot_r", "littletoe_02_l", "littletoe_02_r", "bigtoe_02_l", "bigtoe_02_r"]

def load_ue_gt(path, selected_joints=None,unit_factor=1):
    with open(path, 'r') as f:
        data = []
        while True:
            frame_idx = -1
            frame_data = {}
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
                # print("pos:", pos)


                frame_data[joint_name] = pos
            if frame_idx != -1:
                data.append(frame_data)
            else:
                break
        return data

def save_gt(data, path):
    # 将NumPy数组转换为Python列表
    for frame_data in data:
        for joint_name, pos in frame_data.items():
            frame_data[joint_name] = pos.tolist() if isinstance(pos, np.ndarray) else pos

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


file_paths1 = "/home/tww/Datasets/ue/val/ath0_run.txt"

save_gt(load_ue_gt(file_paths1, selected_joints,unit_factor=1),"/home/tww/Datasets/ue/actorsGT.json")
