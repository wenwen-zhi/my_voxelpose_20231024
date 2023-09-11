from __future__ import division
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from compute_proj import compute_proj, project_points_with_proj
from scipy.spatial.transform import Rotation as R


def unfold_camera_param(camera, device=None):
    # print("camera data:",camera)
    print()
    R = torch.as_tensor(np.array(camera['R']).reshape(3, 3), dtype=torch.float, device=device)
    T = torch.as_tensor(np.array(camera['T']).reshape((3, 1)), dtype=torch.float, device=device)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    f = torch.tensor([fx, fy], dtype=torch.float, device=device).reshape(2, 1)
    c = torch.as_tensor(
        np.array([[camera['cx']], [camera['cy']]]),
        dtype=torch.float,
        device=device)
    k = torch.as_tensor(camera['k'], dtype=torch.float, device=device)
    p = torch.as_tensor(camera['p'], dtype=torch.float, device=device)
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    # 打印出这几个的形状，全部打印一下
    # print("X:",x.shape)

    # 转换到相机坐标系下
    xcam = torch.mm(R, torch.t(x) - T)

    """
    通过除以z轴坐标的方法将相机坐标系下的点投影到图像平面上。
    具体来说，我们将相机坐标系下的点 x_{cam}的前两个维度除以第三个维度，得到一个二维的向量 y
    为了避免除以 0 的错误，我们在除法中加上了一个很小的数 1e-5
    """
    y = xcam[:2] / (xcam[2] + 1e-5)
    ###############################################
    # 暂时注释掉：考虑畸变相关参数。
    # 不考虑畸变相关参数，与畸变相关参数为0的情况等同。
    # kexp = k.repeat((1, n)) # 将径向畸变系数 k 扩展成与点数相同的张量
    # r2 = torch.sum(y ** 2, 0, keepdim=True)  # 计算每个点到图像中心的距离的平方
    # r2exp = torch.cat([r2, r2 ** 2, r2 ** 3], 0) # 将距离的平方进行幂次扩展
    # radial = 1 + torch.einsum('ij,ij->j', kexp, r2exp) # 计算径向畸变系数
    # 
    # tan = p[0] * y[1] + p[1] * y[0] # 计算切向畸变系数
    # corr = (radial + 2 * tan).repeat((2, 1)) # 计算总的畸变系数
    # 
    # y = y * corr + torch.ger(torch.cat([p[1], p[0]]).view(-1), r2.view(-1)) # 对点进行畸变校正
    ###############################################
    ypixel = (f * y) + c  # 将点从归一化平面转换到像素坐标系
    # print(ypixel)
    return torch.t(ypixel)


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    return project_point_radial(x, R, T, f, c, k, p)


def project_pose_v2_proj(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera, device=x.device)
    P = compute_proj(x, R, T, f, c, k, p)
    return project_points_with_proj(x, P)


import os, json


def get_cam(dataset_dir):
    cam_file = os.path.join(dataset_dir, "calibration_shelf.json")
    with open(cam_file) as cfile:
        cameras = json.load(cfile)
    for id, cam in cameras.items():
        for k, v in cam.items():
            cameras[id][k] = np.array(v)
    return cameras


def display_array_info(arr, name):
    print(f"[{name}]: mean={arr.mean()} min={arr.min()} max={arr.max()}")

# 使用投影矩阵
def _get_camera_projection_matrix(dataset_dir):
    # 加载投影矩阵
    projfile = os.path.join(dataset_dir, "proj.json")
    with open(projfile, 'r', encoding='utf-8') as f:
        proj = json.load(f)
    return proj


def project_pose3d_to_pose2d_shelf(pose3d: np.ndarray, proj: np.ndarray):
    # pose3d: nx21x3
    # pose3d=np.array(pose3d)/1000

    # 这个函数的本质是：把3d点投影到2d
    # 我们的投影需求是多样化的，比如把pose3d 投影到 pose2d ， 那就是 21x3 -> 21x2
    # 也有可能是把多个pose3d 投影到 pose2d, 那就是 nx21x3 -> nx21x2
    # 甚至是把多张图片里的pose3d都投影到 pose2d 那就是 batch_sizexnx21x3 -> batch_sizexnx21x3
    # 有那么多种需求，那么多种输入形状， 有办法统一计算吗？ 有的
    # 我们的需求本质就是把给的所有3d点都分别投影到3d， 但是要按照给定的形状返回。
    # 21x3 这种形式是比较好处理的， 它的本质是一个21个点的列表， 投影的过程相当于对21个点分别投影， 得到21个2d点再组合为pose2d
    # 所以可以把别的形式都转换为 21x3这种， 例如 2x21x3 的投影， 其实相当于 42x3的投影得到42x2之后再拆分为2个21x2

    # 你能看明白吗？
    pose3d = np.array(pose3d)

    pre_shape = pose3d.shape[:-1]
    # print(pose3d.shape)
    pose3d = pose3d.reshape((-1, 3)) # 这个是Numpy运算， 你先学学吧

    # M = np.array([[1.0, 0.0, 0.0],
    #               [0.0, 0.0, 1.0],
    #               [0.0, 1.0, 0.0]])

    # pose3d[:, 0:3] = pose3d[:, 0:3].dot(M) # 这个可能是我们自己加的？可能是为了方便可视化，有印象吗，好像本来是倒着的，弄了后就正了？
    # # display_array_info(pose3d,"pose3d")

    # print(pose3d.shape)
    proj = np.array(proj)
    # print(proj.shape)
    num_joints = pose3d.shape[0]
    # print(pose3d[:3])
    pose3d = pose3d.T
    # numpy学习一下
    pose3d = np.append(pose3d, np.ones((1, num_joints)), axis=0)

    # 在这里使用投影矩阵
    pose2d = proj.dot(pose3d)  # .dot()：点乘
    # print(pose2d.shape)
    pose2d = pose2d[:2] / pose2d[2:3]
    # 2x21

    pose2d = pose2d.T  # 21x2
    # display_array_info(pose2d,"pose2d")
    pose2d = pose2d.reshape((*pre_shape, 2)).astype(float) # 现在本来不就21*2
    # print("pose2d：")
    # print(pose2d)
    # pose2d[:, 0] *= image_size[0]
    # pose2d[:, 1] *= image_size[1]
    return pose2d


# 实现一个新函数：
# 参数：图片，2D的点
# 返回：新的图像，其中点周围的一个大小为10左右的框被修改为红色。搜索、chatgpt
from PIL import Image, ImageDraw

from PIL import Image, ImageDraw


def highlight_points(image, points, width=1):
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
        draw.rectangle((x - 10, y - 10, x + 10, y + 10), outline='red', width=width)

    return new_image


# 再实现一个函数
# 并排显示多张图片，可以使用matplotlib，参考网上搜的

def display_images(images, titles=None):
    """并排显示多张图片

    Args:
        images: 图片数组，可以是numpy数组或PIL Image对象列表
        titles: 可选的标题列表，长度应该与images一致
    """

    fig = plt.figure(figsize=(20, 5))
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
    fov = np.array( _params['STADIUM_CamFoc'])

    # print("skel_poses",_lines)
    cam_pos[1] *= -1

    cam_pos = transform_unreal_to_stadium(cam_pos)
    print("transform_poses", cam_pos)

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


def get_image_files(dir,regex="/**/*"):
    # get all the files in the directory
    files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        files.extend(glob.glob(dir + regex + ext, recursive=True))
    return files


def get_shelf_images(shelf_dir,cameranum):

    result={}
    for i in range(cameranum):
        cam=f"camera{i}"
        # result[cam]=sorted(glob.glob(shelf_dir+"/"+cam+"/*.jpeg"))
        result[cam]=[]
        for ext in [".jpg", ".jpeg", ".png"]:
            result[cam].extend(glob.glob(shelf_dir+"/"+cam +"/*" + ext, recursive=True))
        result[cam]=sorted(result[cam])
    return result



def main():
    dataset_dir = "/home/tww/Datasets/ue5"
    # cameras = get_cam(dataset_dir)
    # print("cameras:",cameras)

    x = np.array([[-4129.479, -2391.506, 367.608], [-4128.343, -2374.888, 419.711], [-4129.799, -2369.448, 428.263], [-4122.169, -2346.362, 413.858], [-4106.612, -2359.963, 390.207], [-4110.165, -2374.323, 412.761], [-4155.205, -2403.304, 388.553], [-4154.569, -2405.723, 413.0], [-4146.026, -2380.174, 414.35], [-4125.592, -2446.308, 308.163], [-4127.969, -2432.691, 289.546], [-4122.865, -2437.44, 290.427], [-4127.516, -2442.67, 308.263], [-4126.146, -2436.71, 303.376], [-4121.744, -2410.816, 330.18], [-4120.58, -2393.866, 364.012], [-4137.049, -2376.652, 314.402], [-4144.542, -2378.957, 305.374], [-4139.387, -2375.981, 301.639], [-4137.494, -2387.427, 321.02], [-4137.274, -2379.997, 315.991], [-4135.464, -2365.638, 345.78], [-4138.839, -2390.593, 364.394]]
)


    x[:, 1] *= -1
    x = transform_unreal_to_stadium(x)
    x=torch.from_numpy(x)
    proj_dict = _get_camera_projection_matrix(dataset_dir)

    projfile = os.path.join(dataset_dir, "cameras.json")
    with open(projfile, 'r', encoding='utf-8') as f:
        _params = json.load(f)

    image_list = []
    cam_images=get_shelf_images(dataset_dir,4)
    print(len(cam_images['camera1']))
    for id ,value in _params.items():
        print("相机：", id)
        print("x:",x)


        pose2d = project_lines(value,x)

        print("Pose2d 3:", pose2d)

        image_file = cam_images[f'camera{id}'][150]

        img = Image.open(image_file)
        new_image = highlight_points(img, pose2d, 4)
        image_list.append(new_image)
    display_images(image_list)


if __name__ == '__main__':
    main()
