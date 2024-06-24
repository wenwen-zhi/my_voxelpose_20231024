import cv2
import json
import numpy as np
import os

# 定义骨架连接关系
SKELETON_CONNECTIONS = np.array([0,1,0,15,0,22,1,2,1,5,1,8,3,4,4,5,6,7,7,8,9,10,9,15,10,11,10,12,10,13,10,14,
                  16,17,16,22,17,18,17,19,17,20,17,21]).reshape((-1, 2)).tolist()

# 定义固定颜色列表
COLORS = [
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 青色
    (128, 0, 0),    # 深红色
    (0, 128, 0),    # 深绿色
    (0, 0, 128),    # 深蓝色
    (128, 128, 0),  # 橄榄色
    (128, 0, 128),  # 紫色
    (0, 128, 128)   # 深青色
]

def visualize_frame_poses(frame_data, img_dir, frame_idx, output_path, visibility_threshold=0.1):
    images = []
    color_index = 0  # 用于轮流选择颜色的索引
    for camera_id, poses in frame_data.items():
        img_path = os.path.join(img_dir, camera_id, "{:06d}.jpg".format(frame_idx + 1))
        img = cv2.imread(img_path)
        for person_index, person_pose in enumerate(poses):
            color = COLORS[color_index % len(COLORS)]  # 选择颜色
            color_index += 1
            for x, y, v in person_pose:
                if v >= visibility_threshold:
                    cv2.circle(img, (int(x), int(y)), 3, color, -1)  # 绘制关节点
            # 绘制骨架
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if person_pose[start_idx][2] >= visibility_threshold and person_pose[end_idx][
                    2] >= visibility_threshold:
                    start_point = tuple(map(int, person_pose[start_idx][:2]))
                    end_point = tuple(map(int, person_pose[end_idx][:2]))
                    cv2.line(img, start_point, end_point, color, 2)  # 绘制骨架线
        images.append(img)

    # 如果没有图像，直接返回
    if not images:
        print(f"No images found for frame {frame_idx + 1}. Skipping.")
        return

    # 确保所有图像尺寸相同
    max_height = max(img.shape[0] for img in images)
    total_width = sum(img.shape[1] for img in images)
    combined_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # 并排合并图像
    current_x = 0
    for img in images:
        height, width = img.shape[:2]
        combined_img[:height, current_x:current_x + width, :] = img
        current_x += width

    cv2.imwrite(output_path, combined_img)


def batch_visualize_pose2d(pose2d_path, img_dir, output_dir):
    with open(pose2d_path, 'r') as file:
        data = json.load(file)

    os.makedirs(output_dir, exist_ok=True)

    # 假设所有相机在同一帧中都有图像
    num_frames = len(next(iter(data.values())))
    for frame_idx in range(num_frames):
        frame_data = {camera_id: frames[frame_idx] for camera_id, frames in data.items() if frame_idx < len(frames)}
        output_path = os.path.join(output_dir, f"frame_{frame_idx + 1:06d}.jpg")
        visualize_frame_poses(frame_data, img_dir, frame_idx, output_path)


# 示例使用
pose2d_path = '/home/tww/Datasets/real/datasets/v1/pose2d.json'  # POSE数据文件路径
img_dir = '/home/tww/Datasets/real/datasets/v1'  # 图像目录路径
output_dir = '/home/tww/Datasets/real/datasets/v1_vis'  # 结果保存路径

batch_visualize_pose2d(pose2d_path, img_dir, output_dir)
