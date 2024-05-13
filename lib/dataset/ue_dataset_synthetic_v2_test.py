# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json
import pickle
import logging
import os
import copy
import scipy.io as scio

from lib.dataset.JointsDataset import JointsDataset
from lib.utils.transforms import projectPoints
from lib.utils.association4d_utils import project_pose3d_to_pose2d
from collections import OrderedDict

logger = logging.getLogger(__name__)

# TRAIN_LIST = [
#     '160422_ultimatum1',
#     '160224_haggling1',
#     '160226_haggling1',
#     '161202_haggling1',
#     '160906_ian1',
#     '160906_ian2',
#     '160906_ian3',
#     '160906_band1',
#     '160906_band2',
#     '160906_band3',
# ]
# VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']
#
# JOINTS_DEF = {
#     'neck': 0,
#     'nose': 1,
#     'mid-hip': 2,
#     'l-shoulder': 3,
#     'l-elbow': 4,
#     'l-wrist': 5,
#     'l-hip': 6,
#     'l-knee': 7,
#     'l-ankle': 8,
#     'r-shoulder': 9,
#     'r-elbow': 10,
#     'r-wrist': 11,
#     'r-hip': 12,
#     'r-knee': 13,
#     'r-ankle': 14,
#     # 'l-eye': 15,
#     # 'l-ear': 16,
#     # 'r-eye': 17,
#     # 'r-ear': 18,
# }

# LIMBS = [[0, 1],
#          [0, 2],
#          [0, 3],
#          [3, 4],
#          [4, 5],
#          [0, 9],
#          [9, 10],
#          [10, 11],
#          [2, 6],
#          [2, 12],
#          [6, 7],
#          [7, 8],
#          [12, 13],
#          [13, 14]]


# LIMBS = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
#                   1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((-1, 2)).tolist()

LIMBS = np.array([0, 0, 0, 1, 1, 1, 3, 4, 6, 7, 9, 9, 9, 9, 9, 14, 16, 16, 16, 16, 16, 21,
                  1, 15, 22, 2, 5, 8, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22]).reshape(
    (-1, 2)).tolist()


# 处理ue的数据集
class UEDatasetSyntheticV2Test(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None, **kwargs):
        super().__init__(cfg, image_set, is_train, transform, **kwargs)

        # self.pixel_std = 200.0
        # self.joints_def = JOINTS_DEF
        # self.limbs = LIMBS
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self._interval = cfg.DATASET.SAMPLE_INTERVAL
        self.num_frames = cfg.DATASET.NUM_FRAMES
        self.num_views = cfg.DATASET.CAMERA_NUM

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        ## 这里先检查缓存文件是否存在，如果存在就直接加载缓存，否则调用_get_db()生成数据
        if cfg.TRAIN.ENABLE_CACHE and osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'interval': self._interval,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)

    def _get_pose2d(self):
        datafile = os.path.join(self.dataset_root, 'pose2d.json')
        print()
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # for cam in data:
            #     data[cam] = np.array(data[cam])
        return data

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, 'calibration.json')  # 假设新的文件路径
        with open(cam_file) as cfile:
            calib = json.load(cfile)
        cameras = {}
        for cam_id, cam_data in calib.items():  # 假设 calib 直接是一个包含所有相机数据的字典
            sel_cam = {}
            sel_cam['K'] = np.array(cam_data['K']).reshape((3, 3))  # 调整为3x3矩阵
            sel_cam['distCoef'] = np.array(cam_data['distCoeff'])  # 注意字段名的小变化
            sel_cam['R'] = np.array(cam_data['R']).reshape((3, 3))  # 调整为3x3矩阵
            sel_cam['t'] = np.array(cam_data['T']).reshape((3, 1))  # 确保为3x1向量
            sel_cam['imgSize'] = cam_data['imgSize']  # 可选，如果需要图像尺寸信息
            if 'rectifyAlpha' in cam_data:  # 可选，如果需要矫正系数
                sel_cam['rectifyAlpha'] = cam_data['rectifyAlpha']
            cameras[cam_id] = sel_cam
        return cameras

    def _get_db(self):
        width = self.cfg.DATASET.IMAGE_WIDTH
        height = self.cfg.DATASET.IMAGE_HEIGHT
        db = []  # [v1,v2,v3,v1,v2,v3,v1,v2,v3]
        # vi: {joints_2d: [num_person x num_joints x 2] }
        # print(self.sequence_list)
        # for seq in self.sequence_list:
        #     # 加载相关参数
        cameras = self._get_cam()
        pose2d_data = self._get_pose2d()

        for frame_idx in range(self.num_frames):  # 遍历帧
            # 处理当
            if frame_idx % self._interval == 0:

                for camera_name, camera_info in cameras.items():
                    # for camera_name, proj in proj_dict.items():
                    # 处理当前相机（视野）
                    image_path = osp.join(self.dataset_root, camera_name, f"{'%06d' % (frame_idx + 1)}.jpg")
                    all_pose2d = pose2d_data[camera_name][frame_idx]
                    pred_pose2d = all_pose2d
                    if len(all_pose2d) > 0:
                        our_cam = {}
                        our_cam['R'] = camera_info['R']
                        # our_cam['T'] = -np.dot(camera_info['R'].T, camera_info['t']) * 10.0  # cm to mm
                        our_cam['T'] = -np.dot(camera_info['R'].T, camera_info['t'])
                        our_cam['fx'] = np.array(camera_info['K'][0, 0])
                        our_cam['fy'] = np.array(camera_info['K'][1, 1])
                        our_cam['cx'] = np.array(camera_info['K'][0, 2])
                        our_cam['cy'] = np.array(camera_info['K'][1, 2])
                        our_cam['k'] = camera_info['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = camera_info['distCoef'][[2, 3]].reshape(2, 1)

                        db.append({
                            'key': "{}_{}".format(frame_idx, camera_name),
                            'image': image_path,
                            'camera': our_cam,
                            'pred_pose2d': pred_pose2d
                        })
        return db

    # 返回单条数据，dataloader加载数据时会用到
    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        assert self.mode == "test"
        for k in range(self.num_views):
            i, m, ih = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            meta.append(m)
            input_heatmap.append(ih)
        return input, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    # def evaluate(self, preds, recall_threshold=500):
    #     datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
    #     data = scio.loadmat(datafile)
    #     actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()  # num_person * num_frame
    #     num_person = len(actor_3d)
    #     total_gt = 0
    #     match_gt = 0
    #
    #     limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
    #     correct_parts = np.zeros(num_person)
    #     total_parts = np.zeros(num_person)
    #     alpha = 0.5
    #     bone_correct_parts = np.zeros((num_person, 10))
    #
    #     for i, fi in enumerate(self.frame_range):
    #         pred_coco = preds[i].copy()
    #         pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
    #         pred = np.stack([self.coco2shelf3D(p) for p in copy.deepcopy(pred_coco[:, :, :3])])
    #
    #         for person in range(num_person):
    #             gt = actor_3d[person][fi] * 1000.0
    #             if len(gt[0]) == 0:
    #                 continue
    #
    #             mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
    #             min_n = np.argmin(mpjpes)
    #             min_mpjpe = np.min(mpjpes)
    #             if min_mpjpe < recall_threshold:
    #                 match_gt += 1
    #             total_gt += 1
    #
    #             for j, k in enumerate(limbs):
    #                 total_parts[person] += 1
    #                 error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
    #                 error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
    #                 limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
    #                 if (error_s + error_e) / 2.0 <= alpha * limb_length:
    #                     correct_parts[person] += 1
    #                     bone_correct_parts[person, j] += 1
    #             pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
    #             gt_hip = (gt[2] + gt[3]) / 2.0
    #             total_parts[person] += 1
    #             error_s = np.linalg.norm(pred_hip - gt_hip)
    #             error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
    #             limb_length = np.linalg.norm(gt_hip - gt[12])
    #             if (error_s + error_e) / 2.0 <= alpha * limb_length:
    #                 correct_parts[person] += 1
    #                 bone_correct_parts[person, 9] += 1
    #
    #     actor_pcp = correct_parts / (total_parts + 1e-8)
    #     avg_pcp = np.mean(actor_pcp[:3])
    #
    #     bone_group = OrderedDict(
    #         [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
    #          ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])
    #     bone_person_pcp = OrderedDict()
    #     for k, v in bone_group.items():
    #         bone_person_pcp[k] = np.sum(bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)
    #
    #     return actor_pcp, avg_pcp, bone_person_pcp, match_gt / (total_gt + 1e-8)幀

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views  # 帧数
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum(((pose[vis, 0:3] - gt[vis]) * 1000) ** 2, axis=-1)))
                    print("mpjpe", mpjpe, pose[vis, 0:3])
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt
