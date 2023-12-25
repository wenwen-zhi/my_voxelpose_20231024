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

#处理ue的数据集
class UEDataset(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None, **kwargs):
        super().__init__(cfg, image_set, is_train, transform, **kwargs)

        # self.pixel_std = 200.0
        # self.joints_def = JOINTS_DEF
        # self.limbs = LIMBS
        self.num_joints = cfg.DATASET.NUM_JOINTS
        if self.image_set == 'train':
            self.sequence_list = ["train"]
            self._interval = cfg.DATASET.SAMPLE_INTERVAL
            # self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = cfg.DATASET.CAMERA_NUM
        elif self.image_set == 'validation':
            self.sequence_list = ["val"]
            self._interval = cfg.DATASET.SAMPLE_INTERVAL
            # self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            self.num_views = cfg.DATASET.CAMERA_NUM

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        ## 这里先检查缓存文件是否存在，如果存在就直接加载缓存，否则调用_get_db()生成数据
        if cfg.TRAIN.ENABLE_CACHE and osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            # assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                # 'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_actor_3d(self, seq):
        datafile = os.path.join(self.dataset_root, seq, 'actorsGT.json')
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            actor_3d = data['actor3D']
        return actor_3d

    # def _get_camera_projection_matrix(self,seq):
    #     # 加载投影矩阵
    #     projfile=os.path.join(self.dataset_root,seq,"proj.json")
    #     with open(projfile,'r',encoding='utf-8') as f:
    #         proj=json.load(f)
    #     return proj
    # def _get_cam(self,dataset_dir,seq):
    #     projfile = os.path.join(dataset_dir,seq, "cameras.json")
    #     with open(projfile, 'r', encoding='utf-8') as f:
    #         _params = json.load(f)
    #     return _params

    def _get_cam(self, dataset_dir, seq):
        projfile = os.path.join(dataset_dir, seq, "cameras.json")
        with open(projfile, 'r', encoding='utf-8') as f:
            s = f.read()
            # print(s)
            cameras = json.loads(s)

        # print("cameras:",cameras)
        for id, cam in cameras.items():
            for k, v in cam.items():
                if isinstance(v, list):
                    cameras[id][k] = np.array(v)
        print(cameras)

        ''''
        
        [1,2]
        经过dataloader 组装后 变成
        [tensor([1,1,1,1]),tensor([2,2,2,2])]
        tensor([[1,2],[1,2],[1,2],[1,2]])
        
        '''
        return cameras

    def _get_db(self):
        # width = 368
        # height = 368
        width = 1280
        height = 720
        db = []  # [v1,v2,v3,v1,v2,v3,v1,v2,v3]
        # vi: {joints_2d: [num_person x num_joints x 2] }
        # print(self.sequence_list)
        for seq in self.sequence_list:
            # 加载相关参数
            cameras = self._get_cam(self.dataset_root, seq)
            # cameras = self._get_cam(self.dataset_root, seq)

            actor_3d = self._get_actor_3d(seq)
            # proj_dict = self._get_camera_projection_matrix(seq)

            # print(proj_dict)

            num_frames = len(actor_3d)
            # num_frames=10

            for frame_idx in range(num_frames):  # 遍历帧
                # 处理当
                if frame_idx % self._interval == 0:
                    pose3d_list = actor_3d[frame_idx]

                    for camera_name, camera_info in cameras.items():
                        # for camera_name, proj in proj_dict.items():
                        # 处理当前相机（视野）
                        image_path = osp.join(seq, "camera" + str(camera_name), f"ath0_run1.{'%04d' % frame_idx}.jpeg")
                        # 接下来加载此图片所对应的所有人的位姿
                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for pose3d in pose3d_list:
                            # 处理当前这个人
                            # pose3d = np.array(body['joints19']).reshape((-1, 4))
                            pose3d = np.array(pose3d)
                            pose3d = pose3d[:self.num_joints]
                            joints_vis = pose3d[:, -1] > 0.1
                            if isinstance(self.root_id, (np.ndarray,list)):
                                if not joints_vis[self.root_id[0]] or not joints_vis[self.root_id[1]]:
                                    continue
                            else:
                                if not joints_vis[self.root_id]:
                                    continue

                            # M = np.array([[1.0, 0.0, 0.0],
                            #               [0.0, 0.0, 1.0],
                            #               [0.0, 1.0, 0.0]])
                            # pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                            # pose3d[:, 0:3] *= 1000

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = project_pose3d_to_pose2d(self.cfg.TAG, pose3d[:, :3], width=width,
                                                                     height=height, cam=camera_info)
                            # if frame_idx == 100:
                            #     print("pose2d", pose2d)


                            all_poses_3d.append(pose3d[:, 0:3])  # 我们也需要x10吗？
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            joints_vis[np.logical_not(check)] = 0

                            # print("check:",check)
                            # print("joints_vis:",joints_vis)

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                        if len(all_poses_3d) > 0:
                            # our_cam = {}
                            # our_cam['R'] = v['R']
                            # our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                            # our_cam['fx'] = np.array(v['K'][0, 0])
                            # our_cam['fy'] = np.array(v['K'][1, 1])
                            # our_cam['cx'] = np.array(v['K'][0, 2])
                            # our_cam['cy'] = np.array(v['K'][1, 2])
                            # our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                            # our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)
                            # proj=proj_dict[camera_name]
                            # proj=np.array(proj)
                            # print(proj)

                            db.append({
                                # 'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                                'key': "{}_{}{}".format(seq, frame_idx, camera_name),
                                'image': osp.join(self.dataset_root, image_path),
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': camera_info,
                                # 'proj':proj
                            })
                            # if frame_idx==100:
                            #     print('joints_2d_db:', db[-1]['joints_2d'])
                            #     print('joints_3d_db:', db[-1]['joints_3d'])
                            #     print('camera_info:', camera_info)

        # print("db[joints_3d]",db[-1]['joints_3d'])
        return db


    # def _get_cam(self, seq):
    #     # se4的参数已经修正过，没有问题。但如果使用seq2,seq5,则需要把里面的参数修正。
    #     cam_file = osp.join(self.dataset_root, seq ,"calibration.json")
    #     with open(cam_file) as cfile:
    #         cameras = json.load(cfile)
    #
    #     for id, cam in cameras.items():
    #         for k, v in cam.items():
    #             cameras[id][k] = np.array(v)
    #     return cameras

 #返回单条数据，dataloader加载数据时会用到
    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))
        if self.mode == "test":
            for k in range(self.num_views):
                i, m, ih = super().__getitem__(self.num_views * idx + k)
                # if i is None:
                #     continue
                input.append(i)
                meta.append(m)
                input_heatmap.append(ih)
            return input, meta, input_heatmap
        else:
            for k in range(self.num_views):
                i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
                if i is None:
                    continue
                input.append(i)
                target.append(t)
                weight.append(w)
                target_3d.append(t3)
                meta.append(m)
                input_heatmap.append(ih)
                # from matplotlib import pyplot as plt
                # if target[0] is not None:
                #     temp_img = target[0].sum(axis=0)
                #     plt.imshow(temp_img)
                #     plt.savefig("output/debug/heatmap.jpg")
            return input, target, weight, target_3d, meta, input_heatmap

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
        gt_num = self.db_size // self.num_views #帧数
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
                    mpjpe = np.mean(np.sqrt(np.sum(((pose[vis, 0:3] - gt[vis])*1000)** 2, axis=-1)))
                    print("mpjpe",mpjpe,pose[vis, 0:3])
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
