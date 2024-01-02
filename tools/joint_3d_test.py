import glob
import json
import os

import numpy as np

from load_ue_cameras import load_ue_cameras
from ue_utils import transform_unreal_to_stadium

selected_joints = ["neck_01","head","pelvis","upperarm_correctiveRoot_l","lowerarm_correctiveRoot_l","middle_metacarpal_l","thigh_correctiveRoot_l","calf_l","foot_l","upperarm_correctiveRoot_r","lowerarm_correctiveRoot_r","middle_metacarpal_r","thigh_correctiveRoot_r","calf_r","foot_r"]

def load_ue_gt(path, selected_joints=None, unit_factor=1):
    with open(path, 'r') as f:
        data = []
        frame_data = []
        frame_idx = -1
        previous_frame_idx = -1

        for line in f:
            line = line.strip()
            if not line:
                if frame_data:
                    data.append(frame_data)
                    frame_data = []
                continue

            parts = line.split()
            frame_idx = int(parts[1])
            if frame_idx != previous_frame_idx and frame_data:
                data.append(frame_data)
                frame_data = []
                previous_frame_idx = frame_idx

            joint_name = parts[2]
            if selected_joints is not None and joint_name not in selected_joints:
                continue

            x = float(parts[3].split("=")[1])
            y = float(parts[4].split("=")[1])
            z = float(parts[5].split("=")[1])
            pos = [x, -y, z]
            pos = transform_unreal_to_stadium(pos)  # 确保这个函数已经被定义
            pos *= unit_factor
            index = selected_joints.index(joint_name) if selected_joints else len(frame_data)
            while len(frame_data) <= index:
                frame_data.append([])  # 确保 frame_data 有足够的空间
            frame_data[index] = [joint_name, pos]

        if frame_data:  # 处理最后一帧数据
            data.append(frame_data)

        return data


def save_gt(data, path):
    converted_data = []
    for frame_data in data:
        frame_dict = {}
        for joint_data in frame_data:
            if len(joint_data) == 2:  # 确保 joint_data 有两个元素可以解包
                joint_name, pos = joint_data
                frame_dict[joint_name] = pos.tolist() if isinstance(pos, np.ndarray) else pos
        if frame_dict:  # 确保不添加空字典
            converted_data.append(frame_dict)

    with open(path, 'w') as f:
        json.dump(converted_data, f, indent=2)


file_paths1 = "/home/tww/Datasets/ue/train/ath0_run.txt"

save_gt(load_ue_gt(file_paths1, selected_joints,unit_factor=1),"/home/tww/Datasets/ue/actorsGT.json")
