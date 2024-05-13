import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# 定义连接顺序
LIMBS23 = np.array([
    0, 1, 0, 15, 0, 22, 1, 2, 1, 5, 1, 8, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 9, 15, 10, 11, 10, 12, 10, 13,
    10, 14, 16, 17, 16, 22, 17, 18, 17, 19, 17, 20, 17, 21
]).reshape((-1, 2)).tolist()

# 处理单个图像和JSON文件的函数
# 处理单个图像和JSON文件的函数
def process_image_and_json(image_path, json_path):
    image = cv2.imread(image_path)
    with open(json_path, 'r') as file:
        data = json.load(file)

    keypoints_all = [shape['points'][0] for shape in data['shapes']]
    num_keypoints_per_person = 23
    INVISIBLE_POINT_THRESHOLD = 100

    keypoint_radius = 5  # 关键点的半径
    keypoint_thickness = -1  # 关键点的厚度，-1表示填充圆形
    line_thickness = 4  # 连接线的厚度

    for i in range(len(keypoints_all) // num_keypoints_per_person):
        offset = i * num_keypoints_per_person
        keypoints = keypoints_all[offset:offset + num_keypoints_per_person]

        # 首先，绘制所有关键点
        for idx, keypoint in enumerate(keypoints):
            point = tuple(map(int, keypoint))
            if point[0] <= INVISIBLE_POINT_THRESHOLD and point[1] <= INVISIBLE_POINT_THRESHOLD:
                continue
            cv2.circle(image, point, keypoint_radius, (0, 255, 0), keypoint_thickness)

        # 然后，绘制连接线
        for limb in LIMBS23:
            start_idx, end_idx = limb
            start_point = tuple(map(int, keypoints[start_idx]))
            end_point = tuple(map(int, keypoints[end_idx]))

            if (start_point[0] <= INVISIBLE_POINT_THRESHOLD and start_point[1] <= INVISIBLE_POINT_THRESHOLD) or \
                    (end_point[0] <= INVISIBLE_POINT_THRESHOLD and end_point[1] <= INVISIBLE_POINT_THRESHOLD):
                continue

            cv2.line(image, start_point, end_point, (255, 0, 0), line_thickness)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



# 文件路径列表
files = [
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000260_3679389987672.jpg", "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000260_3679389987672.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000261_3679589934760.jpg", "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000261_3679589934760.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000262_3679789882592.jpg", "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000262_3679789882592.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000263_3679989829216.jpg", "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186131/000263_3679989829216.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000260_3679270306568.jpg",
     "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000260_3679270306568.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000261_3679470299592.jpg",
     "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000261_3679470299592.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000262_3679670293024.jpg",
     "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000262_3679670293024.json"),
    ("/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000263_3679870286512.jpg",
     "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186132/000263_3679870286512.json")
]

#
# files = [
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000260_3679376682432.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000260_3679376682432.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000261_3679576584456.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000261_3679576584456.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000262_3679776611888.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000262_3679776611888.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000263_3679976593544.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000263_3679976593544.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000260_3679376682432.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000260_3679376682432.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000261_3679576584456.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000261_3679576584456.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000262_3679776611888.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000262_3679776611888.json"
#     ),
#     (
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000263_3679976593544.jpg",
#         "/home/tww/Datasets/real/户外跑步数据集/标注文件/22186119/000263_3679976593544.json"
#     )
#
# ]

# 遍历所有文件路径对，并进行处理
for image_path, json_path in files:
    process_image_and_json(image_path, json_path)