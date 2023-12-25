# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import os
import os.path as osp
import pickle
import scipy.io as scio

import json_tricks as json
import numpy as np

from lib.dataset.JointsDataset import JointsDataset

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

LIMBS = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((-1, 2)).tolist()



class MyDataset(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None,**kwargs):
        super().__init__(cfg, image_set, is_train, transform,**kwargs)

        # self.pixel_std = 200.0
        # self.joints_def = JOINTS_DEF
        self.cfg=cfg
        self.limbs = LIMBS
        self.num_joints = cfg.DATASET.NUM_JOINTS


        if self.image_set == 'train':
            raise NotImplementedError("暂不支持训练！")
        elif self.image_set == 'validation':
            # self.sequence_list = ["seq5"]
            self._interval = cfg.DATASET.SAMPLE_INTERVAL
            # self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            self.num_views = cfg.DATASET.CAMERA_NUM

        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        ## 这里先检查缓存文件是否存在，如果存在就直接加载缓存，否则调用_get_db()生成数据
        if  cfg.TRAIN.ENABLE_CACHE and osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            # assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            # assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                # 'sequence_list': self.sequence_list,
                'interval': self._interval,
                # 'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        self.db_size = len(self.db)
    def _get_actor_3d(self,dataset_dir):
        datafile = os.path.join(dataset_dir, 'actorsGT.json')
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            actor_3d = data['actor3D']
        return actor_3d
    def _get_camera_projection_matrix(self,dataset_dir):
        # 加载投影矩阵
        projfile=os.path.join(dataset_dir,"proj.json")
        with open(projfile,'r',encoding='utf-8') as f:
            proj=json.load(f)
        return proj
    def _get_cam(self,dataset_dir):
        cam_file = osp.join(dataset_dir, "calibration.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)
        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)
            cameras[id]["R"]=cameras[id]['R'].reshape((3,3))
            cameras[id]["T"]=cameras[id]['T'].reshape((3,1))
        return cameras
    def _get_db(self):
        db = [] #[v1,v2,v3,v1,v2,v3,v1,v2,v3]
        # 加载相关参数
        cameras = self._get_cam(self.dataset_root)
        proj_dict = self._get_camera_projection_matrix(self.dataset_root)

        for frame_idx in range(self.cfg.DATASET.NUM_FRAMES): # 遍历帧
            # 处理当前帧
            if frame_idx % self._interval== 0:
                for camera_name, camera_info in cameras.items():
                     # 处理当前相机（视野）
                    image_path=osp.join(camera_name,f"{frame_idx+1}_{camera_name}.jpeg")
                    proj=proj_dict[camera_name]
                    proj=np.array(proj)
                    # print(proj)
                    db.append({
                        'key': "{}_{}".format(frame_idx, camera_name),
                        'image': osp.join(self.dataset_root, image_path),
                        'camera': camera_info,
                        'proj':proj
                    })
        return db


    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))
        if self.mode=="test":
            for k in range(self.num_views):
                i, m = super().__getitem__(self.num_views * idx + k)
                if i is None:
                    continue
                input.append(i)
                meta.append(m)
            return input, meta
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

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
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
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
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




