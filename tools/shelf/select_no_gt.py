from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import scipy.io as scio
import logging
import copy
import os
from collections import OrderedDict

from lib.dataset.JointsDataset import JointsDataset
from lib.utils.cameras_cpu import project_pose


datafile = os.path.join("/home/tww/Projects/voxelpose-pytorch/data/CampusSeq1", 'actorsGT.mat')
data = scio.loadmat(datafile)
actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()  # num_person * num_frame

print ("actor_3d",actor_3d.shape)

empty_count = 0
for frame_idx in range(actor_3d.shape[1]):
    tmp = [actor_3d[p_idx][frame_idx].size for p_idx in range(len(actor_3d))]
    empty_count += sum(tmp) == 0

print("empty", empty_count)

valid_frames = []
for frame_idx in range(0,actor_3d.shape[1]):
    tmp = [actor_3d[p_idx][frame_idx].size for p_idx in range(len(actor_3d))]
    if sum(tmp) != 0:
        print("第一个非零帧的索引：", frame_idx)
        print("对应的数据：", actor_3d[:, frame_idx],actor_3d[:, frame_idx].shape)
        break  # 找到第一个非零帧后退出循环
    else:
        valid_frames.append(frame_idx)

# # json save

non_empty_frame_indices = []  # 用于存储非零帧的索引,总共有831帧

for frame_idx in range(actor_3d.shape[1]):
    tmp = [actor_3d[p_idx][frame_idx].size for p_idx in range(len(actor_3d))]
    if sum(tmp) != 0:
        non_empty_frame_indices.append(frame_idx)

print("非零帧的索引：", non_empty_frame_indices,len(non_empty_frame_indices))



# 初始化一个字典来存储每一帧的非空 'p_id' 数量
non_empty_counts_per_frame = {}

# 遍历每一帧
for frame_idx in range(actor_3d.shape[1]):
    non_empty_count_per_frame = 0  # 初始化每一帧的计数为0
    for p_idx in range(actor_3d.shape[0]):  # 假设 'p_id' 的范围是从0到数组的长度
        if actor_3d[p_idx][frame_idx].size != 0:  # 检查 'p_id' 的大小是否不为空
            non_empty_count_per_frame += 1  # 如果不为空，则计数加一
    non_empty_counts_per_frame[frame_idx] = non_empty_count_per_frame  # 存储每一帧的计数结果

# 输出每一帧中非空 'p_id' 的数量
for frame_idx, count in non_empty_counts_per_frame.items():
    print(f"第 {frame_idx} 帧：非空 'p_id' 数量 - {count}")



# # 删选出actor_3d非零的帧数
# print(actor_3d.shape)
# non_zero_frames = np.count_nonzero(actor_3d != 0)
#
# # 输出帧数
# print("非零帧数:", non_zero_frames)