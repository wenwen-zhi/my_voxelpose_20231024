import numpy as np

def compute_proj(x, R, T, f, c, k, p):
    import numpy as np

    # 相机内参
    # f = 50  # 焦距
    fx=f[0][0]
    fy=f[1][0]
    px, py = c[0][0], c[1][0]  # 主点坐标
    # aspect_ratio = 640 / 480  # 长宽比

    # 相机外参
    # R = np.array([[1, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 1]])  # 旋转矩阵
    # T = np.array([0, 0, 0])  # 平移矩阵

    # 构建投影矩阵
    K = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0, 1]])  # 内参矩阵
    # K = np.array([[fx, 0, px],
    #               [0, fy, py],
    #               [0, 0, 1]])  # 内参矩阵
    RT = np.hstack((R, T.reshape(3, 1)))  # 外参矩阵
    P = np.dot(K, RT)  # 投影矩阵
    return P


def project_points_with_proj(points_3d, P):
    """
    Input:
    - points_3d: a numpy array of shape (N,3) representing N 3D points
    - P: a numpy array of shape (3,4) representing the camera projection matrix

    Output:
    - points_2d: a numpy array of shape (N,2) representing N 2D points
    """
    # add homogeneous coordinates to 3D points
    # print("P:",P.shape,"points_3d:",points_3d.shape)
    N = points_3d.shape[0]
    points_3d_hom = np.concatenate((points_3d, np.ones((N, 1))), axis=1)

    # project 3D points to 2D points
    points_2d_hom = np.dot(P, points_3d_hom.T).T
    points_2d = (points_2d_hom[:, :2] / points_2d_hom[:, 2:])[:, ::-1]  # divide by z
    return points_2d
