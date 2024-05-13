import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 全局设置字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei']  # 举例两种常见的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义连线信息
LIMBS23 = np.array(
    [0, 1, 0, 15, 0, 22, 1, 2, 1, 5, 1, 8, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 9, 15, 10, 11, 10, 12, 10, 13, 10, 14,
     16, 17, 16, 22, 17, 18, 17, 19, 17, 20, 17, 21]).reshape((-1, 2)).tolist()


def visualize_pose_joints_and_limbs_no_overlap(pose_json_path, output_image_path):
    # 加载JSON数据
    with open(pose_json_path, 'r') as f:
        data = json.load(f)
    frames = data['frames']

    # 预定义颜色列表
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    # 创建一个大的画布
    fig = plt.figure(figsize=(20, 5))


    # 遍历每一帧
    for frame_idx, frame in enumerate(frames):
        ax = fig.add_subplot(1, 4, frame_idx + 1, projection='3d')

        # 首先绘制所有的连线
        for person_idx, person in enumerate(frame):
            color = colors[person_idx % len(colors)]
            for limb in LIMBS23:
                point_start = person[limb[0]]
                point_end = person[limb[1]]
                ax.plot([point_start[0], point_end[0]], [point_start[1], point_end[1]], [point_start[2], point_end[2]],
                        color)

        # 然后在同一位置绘制空心关节点
        for person_idx, person in enumerate(frame):
            color = colors[person_idx % len(colors)]
            for joint in person:
                ax.scatter(joint[0], joint[1], joint[2], edgecolors=color, facecolors='white', s=20)  # 稍大的关节点大小

        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'帧 {frame_idx + 1}')

    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # 保存整个画布为一张图片
    plt.savefig(output_image_path)
    plt.close()


# 示例调用
# visualize_pose_joints_and_limbs_no_overlap('pose.json', 'output_combined_image_no_overlap.png')

# 示例调用
visualize_pose_joints_and_limbs_no_overlap('/home/tww/Projects/voxelpose-pytorch/output/mydatasetv2/resnet50/synthetic_v2_test/预测结果挑选/result.json', '/home/tww/Projects/voxelpose-pytorch/output/mydatasetv2/resnet50/synthetic_v2_test/预测结果挑选/result.png')
