from __future__ import division
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from compute_proj import compute_proj, project_points_with_proj
from scipy.spatial.transform import Rotation as R

import os, json

def display_array_info(arr, name):
    print(f"[{name}]: mean={arr.mean()} min={arr.min()} max={arr.max()}")


# 实现一个新函数：
# 参数：图片，2D的点
# 返回：新的图像，其中点周围的一个大小为10左右的框被修改为红色。搜索、chatgpt
from PIL import Image, ImageDraw

from PIL import Image, ImageDraw


def highlight_points(image, points, width=1, radius=10):
    """
    修改图片中指定点周围的一个大小为10左右的框为红色

    参数：
        image: PIL Image对象，输入图片
        points: list of tuple, 包含x, y坐标的点列表
        width: int, 矩形边框的宽度，默认为1

    返回值：
        PIL Image对象，修改后的图像
    """
    # 复制原始图像
    new_image = image.copy()

    # 创建绘图对象
    draw = ImageDraw.Draw(new_image)

    # 绘制红色框
    for point in points:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=width)

    return new_image


# 再实现一个函数
# 并排显示多张图片，可以使用matplotlib，参考网上搜的

def display_images(images, titles=None):
    """并排显示多张图片

    Args:
        images: 图片数组，可以是numpy数组或PIL Image对象列表
        titles: 可选的标题列表，长度应该与images一致
    """

    fig = plt.figure(figsize=(100, 25))
    for i, image in enumerate(images):
        # 创建子图
        ax = fig.add_subplot(1, len(images), i + 1)
        # 显示图片
        ax.imshow(image)
        # 添加标题
        if titles is not None:
            ax.set_title(titles[i])
        # 关闭坐标轴
        ax.axis('off')
    # 显示图片
    plt.savefig("proj_test.jpg")
    plt.show()


def rotation_from_azelro(azim, elev, roll):
    r_pan = R.from_euler('z', azim, degrees=True).as_matrix()
    r_tilt = R.from_euler('x', elev, degrees=True).as_matrix()
    r_roll = R.from_euler('y', roll, degrees=True).as_matrix()
    r = r_pan.T.dot(r_tilt).dot(r_roll)
    return r


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


def project_lines(_params, _lines, w=1280, h=720):
    cam_pos = np.array(_params['STADIUM_CamPos'])
    camrot = _params['STADIUM_CamRot']
    fov = np.array(_params['STADIUM_CamFoc'])

    # 单位转为米
    # # _lines /= 1000
    # cam_pos /= 1000

    # print("skel_poses",_lines)
    # cam_pos[1] *= -1
    # cam_pos = transform_unreal_to_stadium(cam_pos)
    # print("transform_poses", cam_pos)

    elev, azim, roll = camrot
    r_raw = rotation_from_azelro(azim + 90, -elev, roll)
    r = r_unreal_stadium.dot(r_raw)

    # print("cam_pos:",cam_pos)
    # print("r:", r)
    # print("fov:",fov)

    F = _params.get('F', 1)
    lf = np.tan(fov / 180 * np.pi / 2) * F
    # print("_lines:",_lines)
    lanes_rot = (np.array(_lines) - cam_pos).dot(r)
    # print("lanes_rot:",lanes_rot)
    pts3d = lanes_rot.reshape(-1, 3)
    pts2d = pts3d / pts3d[:, 1, np.newaxis]
    lines_flat = pts2d.reshape(_lines.shape)
    lines_2d = lines_flat[..., [0, 2]]

    lines_2d = lines_2d / (2 * lf) * w
    lines_2d += (w // 2, h // 2)
    return lines_2d


def get_image_files(dir, regex="/**/*"):
    # get all the files in the directory
    files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        files.extend(glob.glob(dir + regex + ext, recursive=True))
    return files


def get_dataset_images(shelf_dir, cameranum):
    result = {}
    for i in range(cameranum):
        cam = f"camera{i}"
        # result[cam]=sorted(glob.glob(shelf_dir+"/"+cam+"/*.jpeg"))
        result[cam] = []
        for ext in [".jpg", ".jpeg", ".png"]:
            result[cam].extend(glob.glob(shelf_dir + "/" + cam + "/*" + ext, recursive=True))
        result[cam] = sorted(result[cam])
    return result


def main():
    dataset_dir = "/home/tww/Datasets/ue/train"
    frame_id = 100
    # cameras = get_cam(dataset_dir)
    # print("cameras:",cameras)

    # x = np.array([[-4129.479, -2391.506, 367.608], [-4128.343, -2374.888, 419.711], [-4129.799, -2369.448, 428.263],
    #               [-4122.169, -2346.362, 413.858], [-4106.612, -2359.963, 390.207], [-4110.165, -2374.323, 412.761],
    #               [-4155.205, -2403.304, 388.553], [-4154.569, -2405.723, 413.0], [-4146.026, -2380.174, 414.35],
    #               [-4125.592, -2446.308, 308.163], [-4127.969, -2432.691, 289.546], [-4122.865, -2437.44, 290.427],
    #               [-4127.516, -2442.67, 308.263], [-4126.146, -2436.71, 303.376], [-4121.744, -2410.816, 330.18],
    #               [-4120.58, -2393.866, 364.012], [-4137.049, -2376.652, 314.402], [-4144.542, -2378.957, 305.374],
    #               [-4139.387, -2375.981, 301.639], [-4137.494, -2387.427, 321.02], [-4137.274, -2379.997, 315.991],
    #               [-4135.464, -2365.638, 345.78], [-4138.839, -2390.593, 364.394]]
    #              )
    # x=np.array([[15.93637, -4.39876,  0.91479],
    #    [16.08876, -4.40106,  1.43993],
    #    [16.14096, -4.42024,  1.52594],
    #    [16.38658, -4.32365,  1.25865],
    #    [16.21842, -4.12677,  1.14146],
    #    [16.13223, -4.22747,  1.37638],
    #    [15.84205, -4.68175,  1.09508],
    #    [15.73928, -4.61891,  1.3483 ],
    #    [15.99978, -4.56619,  1.39431],
    #    [15.30289, -4.3668 ,  0.53919],
    #    [15.19262, -4.386  ,  0.3881 ],
    #    [15.18077, -4.33372,  0.43353],
    #    [15.65707, -4.32453,  0.61589],
    #    [15.6449 , -4.31192,  0.54053],
    #    [15.96477, -4.31801,  0.52692],
    #    [15.93397, -4.30984,  0.87941],
    #    [16.30326, -4.44826,  0.20769],
    #    [16.45077, -4.49202,  0.15148],
    #    [16.48734, -4.435  ,  0.17017],
    #    [15.77262, -4.39986,  0.23659],
    #    [15.86607, -4.41503,  0.22527],
    #    [15.98023, -4.50866,  0.518  ],
    #    [15.93568, -4.49381,  0.9034 ]])

    #

    GT_file = os.path.join(dataset_dir, "actorsGT.json")
    with open(GT_file, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    x=gt["actor3D"][frame_id][0]
    # x[:, 1] *= -1
    # x = transform_unreal_to_stadium(x)
    # x = torch.from_numpy(x)
    # proj_dict = _get_camera
    # _projection_matrix(dataset_dir)
    x=np.array(x)
    x=x[:,:3]
    print("x:", x)

    projfile = os.path.join(dataset_dir, "cameras.json")
    with open(projfile, 'r', encoding='utf-8') as f:
        _params = json.load(f)

    image_list = []
    cam_images = get_dataset_images(dataset_dir, 4)
    print(len(cam_images['camera1']))
    for id, value in _params.items():
        print("相机：", id)
        # print("x:", x)
        print("参数：：", value)

        pose2d = project_lines(value, x)

        print("Pose2d :", pose2d)

        image_file = cam_images[f'camera{id}'][frame_id]

        img = Image.open(image_file)
        new_image = highlight_points(img, pose2d,9,5)
        image_list.append(new_image)
    display_images(image_list)


if __name__ == '__main__':
    main()
