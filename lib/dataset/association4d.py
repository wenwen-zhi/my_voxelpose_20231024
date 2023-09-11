# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import time

import numpy as np
import json_tricks as json
import pickle
import scipy.io as scio
import logging
import copy
import os
from collections import OrderedDict
import cv2

from lib.dataset.JointsDataset import JointsDataset
from lib.utils.cameras_cpu import project_pose
from lib.utils.association4d_utils import  project_pose3d_to_pose2d


CAMPUS_JOINTS_DEF = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13

}

# LIMBS = [
#     [0, 1],
#     [1, 2],
#     [3, 4],
#     [4, 5],
#     [2, 3],
#     [6, 7],
#     [7, 8],
#     [9, 10],
#     [10, 11],
#     [2, 8],
#     [3, 9],
#     [8, 12],
#     [9, 12],
#     [12, 13]
# ]
LIMBS = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((-1, 2)).tolist()


class Association4D(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        self.pixel_std = 200.0
        # self.joints_def = CAMPUS_JOINTS_DEF
        super().__init__(cfg, image_set, is_train, transform)
        self.limbs = LIMBS
        self.num_joints = 21
        self.cam_list = [0, 1, 2, 3, 4, 5]
        self.num_views = len(self.cam_list)
        # self.frame_range = list(range(350, 471)) + list(range(650, 751))
        # self.frame_range = list(range(1, 3690 + 1))
        self.frame_range = list(range(2735))

        # self.pred_pose2d = self._get_pred_pose2d()
        self.db = self._get_db()

        self.db_size = len(self.db)

    # def _get_pred_pose2d(self):
    #     file = os.path.join(self.dataset_root, "pred_campus_maskrcnn_hrnet_coco.pkl")
    #     with open(file, "rb") as pfile:
    #         logging.info("=> load {}".format(file))
    #         pred_2d = pickle.load(pfile)
    #
    #     return pred_2d
    def _get_actor_3d(self):
        '''
        actorGT.json:
        {
        actor3D:[num_frame][num_person][num_keypoint][3]
        }
        '''
        datafile = os.path.join(self.dataset_root, 'actorsGT.json')
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            actor_3d = data['actor3D']
        return actor_3d

    def _get_actor_2d(self):
        datafile = os.path.join(self.dataset_root, 'actors2D.json')
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            actor_2d = data['actor2D']
        return actor_2d

    def _get_camera_projection_matrix(self):
        # 加载投影矩阵
        projfile=os.path.join(self.dataset_root,"proj.json")
        with open(projfile,'r',encoding='utf-8') as f:
            proj=json.load(f)
        return proj


    def _get_db(self):
        # 大小不确定，可能有错
        ori_width=368
        ori_height=368
        # width = 2048
        # height = 2048
        
        db = []
        cameras = self._get_cam()
        actor_3d = self._get_actor_3d()
        proj_dict=self._get_camera_projection_matrix()
        # f=open("output/debug/association4d.txt",'a')
        for i in self.frame_range:  # for each frame
            for k, cam in cameras.items():  # for each view(camera)
                image = osp.join(k, "%s.jpg" % (i))  # "{k}/{i}.jpg"
                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_poses_vis = []
                num_person = len(actor_3d[i])
                for person in range(num_person):
                    pose3d = np.array(actor_3d[i][person])
                    if len(pose3d[0]) > 0:
                        pose2d = project_pose3d_to_pose2d(pose3d, proj_dict[k])
                        pose2d[:, 0] *= ori_width
                        pose2d[:, 1] *= ori_height
                        # f.write(str(pose2d))

                    pose3d = pose3d * 1000
                    if len(pose3d[0]) > 0:  # person no present
                        all_poses_3d.append(pose3d)
                        all_poses_vis_3d.append(np.ones((self.num_joints, 3)))

                        # print("pose3d 取值：",pose3d.min(),pose3d.max()) #  -314.59999999999997 1565.57 |  -1132.58 1590.7199999999998
                        # print("pose2d 取值：",pose2d.min(),pose2d.max()) #  -94429.40178027563 624.7528680809709
                        # time.sleep(10000)
                        # 检查坐标值是否超出边界

                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                 pose2d[:, 0] <= ori_width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                 pose2d[:, 1] <= ori_height - 1)
                        check = np.bitwise_and(x_check, y_check)

                        joints_vis = np.ones((len(pose2d), 1))
                        joints_vis[np.logical_not(check)] = 0

                        all_poses.append(pose2d)
                        all_poses_vis.append(
                            np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                preds = None
                db.append({
                    'image': osp.join(self.dataset_root, image),
                    'joints_3d': all_poses_3d,
                    'joints_3d_vis': all_poses_vis_3d,
                    'joints_2d': all_poses,
                    'joints_2d_vis': all_poses_vis,
                    'camera': cam,
                    'pred_pose2d': preds
                })
        return db

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, "calibration.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras

    def __getitem__(self, idx):

        # 输入图像，2d关键点标签生成的heatmap，2d权重，...
        inputs, target_heatmap, target_weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for k in range(self.num_views):
            i, th, tw, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            inputs.append(i)
            target_heatmap.append(th)
            target_weight.append(tw)
            input_heatmap.append(ih)
            target_3d.append(t3)
            meta.append(m)

            from matplotlib import pyplot as plt
            if target_heatmap[0] is not None:
                temp_img = target_heatmap[0].sum(axis=0)
                plt.imshow(temp_img)
                plt.savefig("output/debug/heatmap.jpg")

        # print(input)

        # from matplotlib import pyplot as plt
        # # print(target_heatmap[0].shape)
        # temp_img = target_heatmap[0].sum(axis=0)
        # plt.imshow(temp_img)
        # # # plt.show()
        # # print(temp_img.max(),temp_img.min())
        # plt.savefig("output/debug/heatmap.jpg")
        # # input()
        # print(np.where(temp_img!=0))
        # time.sleep(10000)
        return inputs, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds, recall_threshold=500):
        # datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        # data = scio.loadmat(datafile)
        # actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()  # num_person * num_frame
        actor_3d = self._get_actor_3d()
        num_person = len(actor_3d)
        total_gt = 0
        match_gt = 0

        # limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
        limbs = self.limbs
        correct_parts = np.zeros(num_person)
        total_parts = np.zeros(num_person)
        alpha = 0.5
        bone_correct_parts = np.zeros((num_person, 10))  # ????

        for i, fi in enumerate(self.frame_range):
            pred_coco = preds[i].copy()
            pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
            pred = np.stack([self.coco2campus3D(p) for p in copy.deepcopy(pred_coco[:, :, :3])])

            for person in range(num_person):
                gt = actor_3d[person][fi] * 1000.0
                if len(gt[0]) == 0:
                    continue

                mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        bone_group = OrderedDict(
            [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
             ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])
        bone_person_pcp = OrderedDict()
        for k, v in bone_group.items():
            bone_person_pcp[k] = np.sum(bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)

        return actor_pcp, avg_pcp, bone_person_pcp, match_gt / (total_gt + 1e-8)

    @staticmethod
    def coco2campus3D(coco_pose):
        """
        transform coco order(our method output) 3d pose to shelf dataset order with interpolation
        :param coco_pose: np.array with shape 17x3
        :return: 3D pose in campus order with shape 14x3
        """
        campus_pose = np.zeros((14, 3))
        coco2campus = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        campus_pose[0: 12] += coco_pose[coco2campus]

        mid_sho = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
        head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear

        head_bottom = (mid_sho + head_center) / 2  # nose and head center
        head_top = head_bottom + (head_center - head_bottom) * 2
        campus_pose[12] += head_bottom
        campus_pose[13] += head_top

        return campus_pose
