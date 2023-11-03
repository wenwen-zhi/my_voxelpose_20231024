import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def display_array_info(arr, name):
    print(f"[{name}]: mean={arr.mean()} min={arr.min()} max={arr.max()}")


def project_pose3d_to_pose2d(tag: str, pose3d: np.ndarray, proj: np.ndarray = None, width=None, height=None, cam=None):
    # print(f"tag:{tag}, ass:",tag.startswith("association4d"))
    if tag.startswith("association4d"):
        return project_pose3d_to_pose2d_association4d(pose3d, proj, width, height)
    elif tag.startswith("ue"):
        return project_lines(cam, pose3d, w=width, h=height)
    else:
        return project_pose3d_to_pose2d_other(pose3d, proj)


def project_lines(_params, _lines, w=1280, h=720):
    cam_pos = _params['STADIUM_CamPos']
    camrot = _params['STADIUM_CamRot']
    fov = _params['STADIUM_CamFoc']

    r_unreal_stadium = R.from_euler('z', -90, degrees=True).as_matrix()

    def rotation_from_azelro(azim, elev, roll):
        # print("azim",azim)
        r_pan = R.from_euler('z', azim, degrees=True).as_matrix()
        r_tilt = R.from_euler('x', elev, degrees=True).as_matrix()
        r_roll = R.from_euler('y', roll, degrees=True).as_matrix()
        r = r_pan.T.dot(r_tilt).dot(r_roll)
        return r


    cam_pos = torch.tensor(cam_pos).cpu().numpy()
    camrot = torch.tensor(camrot).cpu().numpy()

    if isinstance(fov, torch.Tensor):
        # 将 PyTorch 张量转换成 NumPy 数组
        fov = fov.cpu().numpy()
        # print("fov:",fov)

    # print("skel_poses",_lines)

    # cam_pos /= 1000 # 单位从毫米转换为米
    # _lines /= 1000

    # cam_pos[1] *= -1
    # cam_pos = transform_unreal_to_stadium(cam_pos)

    elev, azim, roll = camrot
    # print("camrot:",camrot)
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


def project_pose3d_to_pose2d_other(pose3d: np.ndarray, proj: np.ndarray):
    '''
    return
    pose2d: [..., 2]
    '''
    # pose3d: 21x3
    pose3d = np.array(pose3d, dtype=np.float32)
    pose3d = np.array(pose3d) / 1000
    proj = np.array(proj)

    # print("pose3d:", pose3d.shape, pose3d.mean(), "proj:", proj.shape)
    pre_shape = pose3d.shape[:-1]
    # print(pose3d.shape)
    pose3d = pose3d.reshape((-1, 3))

    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])
    # pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
    # display_array_info(pose3d,"pose3d")

    # print(pose3d.shape)

    # print(proj.shape)
    num_joints = pose3d.shape[0]
    # print(pose3d[:3])
    pose3d = pose3d.T
    pose3d = np.append(pose3d, np.ones((1, num_joints)), axis=0)

    # 检查哪里出问题了，
    # print("pose3")
    pose2d = proj.dot(pose3d)  # .dot()：点乘
    # print(pose2d.shape)
    pose2d = pose2d[:2] / pose2d[2:3]
    # 2x21

    pose2d = pose2d.T  # 21x2
    # display_array_info(pose2d,"pose2d")
    # pose2d[:, 0] *= width
    # pose2d[:, 1] *= height
    pose2d = pose2d.reshape((*pre_shape, 2)).astype(float)
    # print("pose2d：",pose2d.shape, "width:",width,"height:",height)
    # print(pose2d)

    return pose2d


def project_pose3d_to_pose2d_association4d(pose3d: np.ndarray, proj: np.ndarray, width=None, height=None):
    '''
    return
    pose2d: [..., 2]
    '''
    # pose3d: 21x3
    pose3d = np.array(pose3d) / 1000
    pose3d = np.array(pose3d)
    proj = np.array(proj)

    # print("pose3d:", pose3d.shape, "proj:", proj.shape)
    pre_shape = pose3d.shape[:-1]
    # print(pose3d.shape)
    pose3d = pose3d.reshape((-1, 3))

    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])
    pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
    # display_array_info(pose3d,"pose3d")

    # print(pose3d.shape)

    # print(proj.shape)
    num_joints = pose3d.shape[0]
    # print(pose3d[:3])
    pose3d = pose3d.T
    pose3d = np.append(pose3d, np.ones((1, num_joints)), axis=0)

    # 检查哪里出问题了，
    # print("pose3")
    pose2d = proj.dot(pose3d)  # .dot()：点乘
    # print(pose2d.shape)
    pose2d = pose2d[:2] / pose2d[2:3]
    # 2x21

    pose2d = pose2d.T  # 21x2
    # display_array_info(pose2d,"pose2d")
    pose2d[:, 0] *= width
    pose2d[:, 1] *= height
    pose2d = pose2d.reshape((*pre_shape, 2)).astype(float)
    # print("pose2d：",pose2d.shape, "width:",width,"height:",height)
    # print(pose2d)

    return pose2d


def _test_project_pose3d_to_pose2d():
    pose3d = np.random.random((12800, 3))
    proj = np.random.random((3, 4))
    pose2d = project_pose3d_to_pose2d(pose3d, proj)
    print(pose2d)


if __name__ == '__main__':
    _test_project_pose3d_to_pose2d()
