from __future__ import division
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from compute_proj import compute_proj, project_points_with_proj


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



def get_image_files(dir,regex="/**/*"):
    files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        files.extend(glob.glob(dir + regex + ext, recursive=True))
    return files


def get_shelf_images():
    shelf_dir="/home/tww/Downloads/Projects/Shelf"
    result={}
    for i in range(5):
        cam=f"Camera{i}"
        # result[cam]=sorted(glob.glob(shelf_dir+"/"+cam+"/*.png"))
        result[cam]=sorted(glob.glob(shelf_dir+"/"+cam+"/*.png"))
    return result


def main():
    # 测试图片是A， 那就找A的关键点
    # 一个关键点不够，要整个人的位姿 3DPOSE， 复制粘贴就行了

    # dataset_dir = "/home/tww/Datasets/MyPoseDataset/test02"
    # dataset_dir = "/home/tww/Datasets/Shelf"
    dataset_dir = "/home/tww/Downloads/Projects/Shelf"
    cameras = get_cam(dataset_dir)
    print("cameras:",cameras)

    # 不断调整这个点，看看能不能这个点同时投影到四张图像中，人头部所在的位置。 如果能，那就正常
    # 把投影得到的2D点在2D图中画出来，然后把4张可视化图片并排显示。人头部从图片中能看到，对吧？不对 不明白 ，不知道，所以只能不断调整3D点，看能不能投影到4个人的头部，那好笨。
    # 不是预测了21个关键点， 头不应该知道， ？？确实哈，
    # x = torch.from_numpy(np.array([
    #     # [100,100,100]
    #     [0.346248729280798,	0.478715312513051,	0.000367118989958870]
    # ]))
    # x = np.array([[0.233247446109849, -0.0805143698769383, -0.00628071495099039],
    #                 [0.213127189420882, -0.0382974272288146, 0.374692070166597],
    #                 [0.210458160300266, -0.0636842649490193, 0.743002977978701],
    #                 [0.0443096588223963, -0.0942115398575138, 0.681523418080836],
    #                 [0.0218407122718260, -0.112075522020628, 0.388038729780042],
    #                 [0.0780798959858840, -0.224773799999763, -0.0581260015395224],
    #                 [0.288577882795238, -0.0247160041414492, 0.836640588796171],
    #                 [0.316662667820476, -0.0531981039893351, 1.05626365746627],
    #                 [0.246850421463062, -0.0493765172143892, 1.32927804915461],
    #                 [-0.0714535502513722, -0.0543393899852102, 1.28244657294271],
    #                 [-0.141221889336791, -0.115775059209270, 1.02812353114047],
    #                 [-0.0564586441171724, -0.0256353169711736, 0.802870547863909],
    #                 [0.0733976178353363, 0.0135452761275132, 1.43381135958563],
    #                 [0.0796912074828362, 0.0955582779790333, 1.58590389084249]])

    x = np.array([
        [
          -4046.129,
          -3940.839,
          366.556,
          1.0
        ],
        [
          -4051.382,
          -3923.475,
          418.198,
          1.0
        ],
        [
          -4050.926,
          -3917.922,
          426.788,
          1.0
        ],
        [
          -4020.145,
          -3938.62,
          382.279,
          1.0
        ],
        [
          -4020.393,
          -3945.025,
          392.486,
          1.0
        ],
        [
          -4033.52,
          -3928.67,
          409.449,
          1.0
        ],
        [
          -4058.289,
          -3890.727,
          403.473,
          1.0
        ],
        [
          -4078.365,
          -3927.964,
          381.547,
          1.0
        ],
        [
          -4069.403,
          -3930.305,
          406.88,
          1.0
        ],
        [
          -4036.584,
          -3941.752,
          326.053,
          1.0
        ],
        [
          -4038.959,
          -3979.068,
          323.389,
          1.0
        ],
        [
          -4033.711,
          -3978.422,
          328.017,
          1.0
        ],
        [
          -4040.453,
          -3966.963,
          341.589,
          1.0
        ],
        [
          -4039.192,
          -3968.179,
          334.053,
          1.0
        ],
        [
          -4039.801,
          -3936.192,
          332.692,
          1.0
        ],
        [
          -4038.282,
          -3941.949,
          361.832,
          1.0
        ],
        [
          -4045.61,
          -3978.688,
          304.393,
          1.0
        ],
        [
          -4054.174,
          -3943.517,
          290.427,
          1.0
        ],
        [
          -4049.306,
          -3938.468,
          290.87,
          1.0
        ],
        [
          -4047.986,
          -3955.408,
          303.659,
          1.0
        ],
        [
          -4049.503,
          -3946.062,
          302.527,
          1.0
        ],
        [
          -4058.866,
          -3934.647,
          331.8,
          1.0
        ],
        [
          -4055.888,
          -3947.501,
          363.697,
          1.0
        ]
      ])
    x=torch.from_numpy(x)
    # print(arr)

    point_list = []
    # 打开文本文件
    with open('/home/tww/Downloads/Projects/Shelf/pose.txt', 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 假设文件中每行都包含一个3D点，点的坐标使用空格或逗号分隔
            # 假设点的坐标顺序为 x, y, z
            # 可根据实际文件内容进行调整
            # 以下是一个示例的解析方式

            # 使用split()方法将一行内容分割成一个列表，使用空格或逗号作为分隔符
            parts = line.strip().split('\t')  # 或者用 line.strip().split(',') 如果使用逗号作为分隔符
            # 获取每个坐标值
            x = float(parts[0])  # 假设第一个值为x坐标
            y = float(parts[1])  # 假设第二个值为y坐标
            z = float(parts[2])  # 假设第三个值为z坐标
            # 根据需要使用获取到的坐标值进行进一步处理
            # 例如，可以将x、y、z值组合成3D点对象，进行后续的计算、可视化等操作
            point = (x, y, z)
            point_list.append([x,y,z])
    x = torch.from_numpy(np.array(point_list))

    proj_dict = _get_camera_projection_matrix(dataset_dir)

    # 新建一个列表 ？？？？
    # 遍历每个相机，把x投影到相机上，获得2d投影点。
    # 调用函数，把2d点显示在二维图像上，获得新的图像
    # 在for循环中，把这写新图像加入一个列表
    # 调用另一个函数，把列表里的图像并排显示

    image_list = []
    cam_images=get_shelf_images()
    for id, cam in cameras.items():
        print("相机：", id)
        print("x:",x)
        # pose2d = project_pose3d_to_pose2d_shelf(x, proj_dict[id])
        x=np.array(x).astype(np.float32)
        x=torch.from_numpy(x)
        pose2d =project_pose(x*1000,cameras[id])
        print("Pose2d 3:", pose2d)

        image_file = cam_images[f'Camera{id}'][517]

        img = Image.open(image_file)
        new_image = highlight_points(img, pose2d, 5)
        image_list.append(new_image)
    display_images(image_list)


if __name__ == '__main__':
    main()
