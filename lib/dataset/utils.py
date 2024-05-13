import random
import numpy as np

from lib.utils.transforms import rotate_points


def isvalid(cameras, project_pose, new_center, bbox, bbox_list):
    new_center_us = new_center.reshape(1, -1)
    vis = 0
    for k, cam in cameras.items():
        width = 360
        height = 288
        loc_2d = project_pose(np.hstack((new_center_us, [[1000.0]])), cam)
        if 10 < loc_2d[0, 0] < width - 10 and 10 < loc_2d[0, 1] < height - 10:
            vis += 1
    if len(bbox_list) == 0:
        return vis >= 2
    bbox_list = np.array(bbox_list)
    x0 = np.maximum(bbox[0], bbox_list[:, 0])
    y0 = np.maximum(bbox[1], bbox_list[:, 1])
    x1 = np.minimum(bbox[2], bbox_list[:, 2])
    y1 = np.minimum(bbox[3], bbox_list[:, 3])

    intersection = np.maximum(0, (x1 - x0) * (y1 - y0))
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_list = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])
    iou_list = intersection / (area + area_list - intersection)

    return vis >= 2 and np.max(iou_list) < 0.01


def calc_bbox(pose, pose_vis):
    index = pose_vis[:, 0] > 0
    bbox = [np.min(pose[index, 0]), np.min(pose[index, 1]),
            np.max(pose[index, 0]), np.max(pose[index, 1])]

    return np.array(bbox)


def default_get_new_center(center_list):
    if len(center_list) == 0 or random.random() < 0.7:
        new_center = np.array([np.random.uniform(-2500.0, 8500.0), np.random.uniform(-1000.0, 10000.0)])
    else:
        xy = center_list[np.random.choice(range(len(center_list)))]
        new_center = xy + np.random.normal(500, 50, 2) * np.random.choice([1, -1], 2)

    return new_center


def gen_pose(cameras, project_pose, pose_db, max_num_poses=5, get_new_center=default_get_new_center,
             compute_center=lambda p: (p[11, :2] + p[12, :2]) / 2):
    nposes = np.random.choice(range(1, max_num_poses))
    bbox_list = []
    center_list = []

    select_poses = np.random.choice(pose_db, nposes)
    joints_3d = np.array([p['joints_3d'] for p in select_poses])
    joints_3d_vis = np.array([p['joints_3d_vis'] for p in select_poses])
    return joints_3d, joints_3d_vis
    print("[gen_pose:initial]joints_3d: ", joints_3d.shape)
    for n in range(0, nposes):

        points = joints_3d[n][:, :2].copy()
        center = compute_center(points)
        # print("joints_3d:",joints_3d.shape,"points: ", points.shape, "center:", center.shape)
        rot_rad = np.random.uniform(-180, 180)

        new_center = get_new_center(center_list)
        new_xy = rotate_points(points, center, rot_rad) - center + new_center

        loop_count = 0
        while not isvalid(cameras, project_pose, new_center, calc_bbox(new_xy, joints_3d_vis[n]), bbox_list):
            loop_count += 1
            if loop_count >= 100:
                break
            new_center = get_new_center(center_list)
            new_xy = rotate_points(points, center, rot_rad) - center + new_center

        if loop_count >= 100:
            nposes = n
            joints_3d = joints_3d[:n]
            joints_3d_vis = joints_3d_vis[:n]
        else:
            center_list.append(new_center)
            bbox_list.append(calc_bbox(new_xy, joints_3d_vis[n]))
            joints_3d[n][:, :2] = new_xy
    print("[gen_pose]joints_3d: ", joints_3d.shape)
    return joints_3d, joints_3d_vis


def compute_human_scale(pose, joints_vis):
    idx = joints_vis[:, 0] == 1
    if np.sum(idx) == 0:
        return 0
    minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
    miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
    return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)


def generate_input_heatmap(joints, joints_vis, image_size, heatmap_size, sigma, joints_weight,
                           use_different_joints_weight, target_type="gaussian"):
    '''
    :param joints:  [[num_joints, 3]]
    :param joints_vis: [num_joints, 3]
    :return: input_heatmap
    '''
    nposes = len(joints)
    num_joints = joints[0].shape[0]
    target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    for i in range(num_joints):
        for n in range(nposes):
            if joints_vis[n][i, 0] == 1:
                target_weight[i, 0] = 1

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros(
            (num_joints, heatmap_size[1], heatmap_size[0]),
            dtype=np.float32)
        feat_stride = image_size / heatmap_size

        for n in range(nposes):
            # obscured = random.random() < 0.05
            # if obscured:
            #     continue
            human_scale = 2 * compute_human_scale(joints[n] / feat_stride, joints_vis[n])
            if human_scale == 0:
                continue

            cur_sigma = sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if joints_vis[n][joint_id, 0] == 0 or \
                        ul[0] >= heatmap_size[0] or \
                        ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                if joint_id in [7, 8]:
                    scale = scale * 0.5 if random.random() < 0.1 else scale
                elif joint_id in [9, 10]:
                    scale = scale * 0.2 if random.random() < 0.1 else scale
                else:
                    scale = scale * 0.5 if random.random() < 0.05 else scale
                g = np.exp(
                    -((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2)) * scale

                # Usable gaussian range
                g_x = max(0,
                          -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0,
                          -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

    if use_different_joints_weight:
        target_weight = np.multiply(target_weight, joints_weight)

    return target, target_weight


def generate_3d_target(joints_3d, space_size, space_center, initial_cube_size):
    num_people = len(joints_3d)

    space_size = space_size
    space_center = space_center
    cube_size = initial_cube_size
    grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
    grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
    grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

    target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
    cur_sigma = 200.0

    for n in range(num_people):
        joint_id = [11, 12]  # mid-hip
        mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
        mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
        mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0

        i_x = [np.searchsorted(grid1Dx, mu_x - 3 * cur_sigma),
               np.searchsorted(grid1Dx, mu_x + 3 * cur_sigma, 'right')]
        i_y = [np.searchsorted(grid1Dy, mu_y - 3 * cur_sigma),
               np.searchsorted(grid1Dy, mu_y + 3 * cur_sigma, 'right')]
        i_z = [np.searchsorted(grid1Dz, mu_z - 3 * cur_sigma),
               np.searchsorted(grid1Dz, mu_z + 3 * cur_sigma, 'right')]
        if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
            continue

        gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]],
                                          indexing='ij')
        g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
        target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

    target = np.clip(target, 0, 1)
    return target
