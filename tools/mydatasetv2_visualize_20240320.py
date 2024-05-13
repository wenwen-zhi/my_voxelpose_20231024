import cv2
import json
import numpy as np
import os

# 定义骨架连接关系
SKELETON_CONNECTIONS = np.array([0,1,0,15,0,22,1,2,1,5,1,8,3,4,4,5,6,7,7,8,9,10,9,15,10,11,10,12,10,13,10,14,
                  16,17,16,22,17,18,17,19,17,20,17,21]).reshape((-1, 2)).tolist()


def visualize_frame_poses(frame_data, img_dir, frame_idx, output_path, visibility_threshold=0.1):
    images = []
    for camera_id, poses in frame_data.items():
        img_path = os.path.join(img_dir, camera_id, "{:06d}.jpg".format(frame_idx + 1))
        img = cv2.imread(img_path)
        for person_pose in poses:
            for x, y, v in person_pose:
                if v >= visibility_threshold:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # 绘制关节点
            # 绘制骨架
            print(np.array(person_pose).shape)
            print(person_pose)
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if person_pose[start_idx][2] >= visibility_threshold and person_pose[end_idx][
                    2] >= visibility_threshold:
                    start_point = tuple(map(int, person_pose[start_idx][:2]))
                    end_point = tuple(map(int, person_pose[end_idx][:2]))
                    cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # 绘制骨架线
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
