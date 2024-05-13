# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import cv2
import torch


def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


# import cv2
# cv2.Rodrigues

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
    # if not R.shape == (3,3):
    #     R,_=cv2.Rodrigues(R)

    n = x.shape[0]
    # x.T-T ： 21x3
    # print(x.shape)
    # print(R)
    xcam = R.dot(x.T - T)
    # print(xcam.max())
    # raise
    y = xcam[:2] / (xcam[2] + 1e-5)
    # print(xcam[2])
    # print(y.max())
    r2 = np.sum(y ** 2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    # print(f)
    # print(c)
    ypixel = np.multiply(f, y) + c
    # print(ypixel.max())
    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)

#
# def project_pose(pose3d, camera, transpose = False):
#     device = None
#     if isinstance(pose3d, (torch.Tensor)):
#         device = pose3d.device
#         pose3d = pose3d.detach().cpu().numpy()
#     # print("posr3d shape:", pose3d.shape)
#     if transpose:
#         M = np.array([[1.0, 0.0, 0.0],
#                       [0.0, 0.0, 1.0],
#                       [0.0, 1.0, 0.0]])
#
#         pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)  # 这个可能是我们自己加的？可能是为了方便可视化，有印象吗，好像本来是倒着的，弄了后就正了？
#     pose3d = torch.from_numpy(pose3d)
#     if device:
#         pose3d = pose3d.to(device)
#     R, T, f, c, k, p = unfold_camera_param(camera)
#     return project_point_radial(pose3d, R, T, f, c, k, p)


def world_to_camera_frame(x, R, T):
    """
    Args
        x: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 3d points in camera coordinates
    """

    xcam = R.dot(x.T - T)  # rotate and translate
    return xcam.T


def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """

    xcam = R.T.dot(x.T) + T  # rotate and translate
    return xcam.T
