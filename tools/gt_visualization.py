import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

LIMBS23=np.array([0,1,0,15,0,22,1,2,1,5,1,8,3,4,4,5,6,7,7,8,9,10,9,15,10,11,10,12,10,13,10,14,
                  16,17,16,22,17,18,17,19,17,20,17,21]).reshape((-1, 2)).tolist()


LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

import json

def save_debug_3d_images(actor3D_data, prefix):
    # 提取人数和关节点信息
    num_person = len(actor3D_data[0])  # 假设每帧的人数是一样的
    num_frames = len(actor3D_data)

    # 遍历每一帧
    for i in range(num_frames):
        frame_data = actor3D_data[i]

        # 设置画布大小
        width = 4.0
        height = 4.0
        fig = plt.figure(figsize=(width, height))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                            top=0.95, wspace=0.05, hspace=0.15)

        ax = plt.subplot(1, 1, 1, projection='3d')

        # 遍历每个人物
        for n in range(num_person):
            joints_3d = frame_data[n]
            joint_vis = [1.0] * len(joints_3d)  # 假设所有关节都可见

            for k in eval("LIMBS{}".format(len(joints_3d))):
                if joint_vis[k[0]] and joint_vis[k[1]]:
                    x = [float(joints_3d[k[0]][0]), float(joints_3d[k[1]][0])]
                    y = [float(joints_3d[k[0]][1]), float(joints_3d[k[1]][1])]
                    z = [float(joints_3d[k[0]][2]), float(joints_3d[k[1]][2])]
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2, markeredgewidth=1)
                else:
                    x = [float(joints_3d[k[0]][0]), float(joints_3d[k[1]][0])]
                    y = [float(joints_3d[k[0]][1]), float(joints_3d[k[1]][1])]
                    z = [float(joints_3d[k[0]][2]), float(joints_3d[k[1]][2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2, markeredgewidth=1)

        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.set_title(f'Frame {i + 1}')
        ax.legend()

        # 保存可视化结果
        save_path = f'{prefix}/visualization_frame_{i + 1}.png'
        plt.savefig(save_path)
        plt.close()

# 从JSON文件加载actor3D数据
json_file_path = "/home/tww/Datasets/ue/train/actorsGT.json"

with open(json_file_path, 'r') as json_file:
    actor3D_data = json.load(json_file)
# 使用示例
save_debug_3d_images(actor3D_data["actor3D"], prefix="/home/tww/Datasets/ue/train/gt")


