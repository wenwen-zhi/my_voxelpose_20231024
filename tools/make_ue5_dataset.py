import json
import os.path
import numpy as np

from scipy.spatial.transform import Rotation as R


def make_ue5_dataset(src_path="/media/tww/ͯ/Dataset/ue5"):
    pass

selected_joints = ["head", "neck_01", "upperarm_correctiveRoot_l", "upperarm_correctiveRoot_r","lowerarm_correctiveRoot_l","lowerarm_correctiveRoot_r","middle_metacarpal_l","middle_metacarpal_r","pelvis","thigh_correctiveRoot_l","thigh_correctiveRoot_r","calf_kneeBack_l","calf_kneeBack_r","ankle_bck_l","ankle_bck_r","ankle_fwd_l","ankle_fwd_r","foot_l","foot_r","littletoe_02_l","littletoe_02_r","bigtoe_02_l","bigtoe_02_r"]



r_unreal_stadium = R.from_euler('z', -90, degrees=True).as_matrix()

def transform_unreal_to_stadium(_pts):
    # placed a cube in the unreal stadium and measured its position
    # origin_coords = [-3428, -4070, 265] # before scale up
    origin_coords = [-3608, -4250, 280]  # with scale up
    # in unreal, 1 unit is 1cm, we have 1 unit = 1m
    scale = 100

    orig = np.array(origin_coords.copy())
    orig[1] *= -1
    #     transform unreal to stadium!!

    pts = _pts.copy()
    pts -= orig
    pts = pts.dot(r_unreal_stadium) / scale
    return pts

def load_ue_gt(path, selected_joints=None):
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
                k=float("1")

                # 如果 selected_joints 不为None，并且 joint_name 不在 selected_joints 中，则跳过该关节数据。
                if selected_joints is not None and joint_name not in selected_joints:
                    continue
                pos=[x, -y, z]
                # pos[:, 1] *= -1
                pos = transform_unreal_to_stadium(pos)
                print("pos:",pos)

                frame_data.append(pos.tolist())
            if frame_idx != -1:
                data.append(frame_data)
            else:
                break
        return data


# def save_gt(data,path):
#     with open(path, 'w') as f:
#         json.dump(data,f,indent=2)
# data=load_ue_gt("/home/tww/Datasets/ue5/ath0_run.txt",selected_joints)
# save_gt(data,"/home/tww/Datasets/ue5/ath0_run.json")
# # print(len(data))

# 文件路径列表
file_paths = ["/home/tww/Datasets/ue/val/ath0_run.txt", "/home/tww/Datasets/ue/val/ath1_run.txt", "/home/tww/Datasets/ue/val/ath2_run.txt"]

# 存储所有帧信息的列表
all_frames = []

# 打开每个文件并按帧顺序处理
for file_path in file_paths:
    data = load_ue_gt(file_path, selected_joints)

    # 确保每个文件的帧数相同（可选）
    if len(data) != len(all_frames) and all_frames:
        raise ValueError("Number of frames in each file must be the same.")

    # 合并帧信息到 all_frames 列表中
    for i, frame_data in enumerate(data):
        if len(all_frames) <= i:
            all_frames.append([])  # 创建一个新的帧数据列表
        all_frames[i].append(frame_data)

def save_gt(data,path):
    with open(path, 'w') as f:
        json.dump(data,f,indent=2)
# all_frames 列表中的每个元素表示一帧数据，按照顺序包含了所有文件的帧信息
print(len(data[0]))
print(len(selected_joints))
actor3D_data = {"actor3D": all_frames}
save_gt(actor3D_data,"/home/tww/Datasets/ue/val/actorsGT.json")
print(all_frames[150][1])


