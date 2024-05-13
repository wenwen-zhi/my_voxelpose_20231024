# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.extend(["."])

import argparse
import os

import numpy as np
import time
import json
from datetime import datetime
from lib.utils import cameras
from lib.dataset import get_dataset
from lib.models import get_model

import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from lib.core.config import config
from lib.core.config import update_config
from lib.utils.association4d_utils import project_pose3d_to_pose2d
from lib.utils.utils import create_logger
from lib.utils.vis import save_debug_3d_cubes_for_test, save_batch_image_with_joints_multi, \
    visualize_heatmaps_on_images, save_heatmaps
from lib.utils.vis import save_debug_3d_images_for_test
from lib.utils.transforms import get_affine_transform as get_transform
from lib.utils.transforms import affine_transform_pts_cuda as do_transform


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    # final_output_dir=config.OUTPUT_DIR
    print('final_output_dir', final_output_dir)
    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = get_dataset(config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        mode="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # 模型加载
    print('=> Constructing models ..')
    model = get_model(config.MODEL).get_multi_person_pose_net(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    # print("final_output_dir:",final_output_dir)
    print("loading model file from :", test_model_file)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    # 模型预测
    model.eval()
    preds = []
    pose2d_vis_dir = os.path.join(final_output_dir, 'pose2d_vis')
    heatmaps_2d_vis_dir = os.path.join(final_output_dir, '2d_heatmaps')
    if os.path.exists(pose2d_vis_dir):
        shutil.rmtree(pose2d_vis_dir)
        time.sleep(1e-7)
    os.makedirs(pose2d_vis_dir, exist_ok=True)
    os.makedirs(heatmaps_2d_vis_dir, exist_ok=True)
    preds_to_save = []
    with torch.no_grad():
        # 模型预测
        for i, (inputs, meta, input_heatmaps) in enumerate(tqdm(test_loader)):
            # inputs shape : num_view x batch_size x 3 x h x w
            batch_size = len(inputs[0])

            if config.PREDICT_ON_2DHEATMAP or config.DATASET.TEST_DATASET in ['shelf', "campus"]:
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, input_heatmaps=input_heatmaps)
            elif config.TEST.PREDICT_FROM_IMAGES or config.DATASET.TEST_DATASET in ['panoptic', "association4d",
                                                                                    "association4d_v2", "mydataset",
                                                                                    "shelf_end_to_end",
                                                                                    "campus_end_to_end", "test_shelf",
                                                                                    "ue_dataset"]:
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, views=inputs)
            else:
                raise Exception("不支持的数据集：", config.DATASET.TEST_DATASET)
            print("Pose3d:", pred.shape, "mean:", pred[:, :, :, :3].mean())
            print("heatmaps.shape:", len(heatmaps), heatmaps[0].shape)
            print("inputs.shape", len(inputs), inputs[0].shape)
            # input()
            visualize_heatmaps_on_images(
                inputs, heatmaps, os.path.join(heatmaps_2d_vis_dir, f"batch_{i}.jpg")
            )
            heatmap_dir = os.path.join(config.OUTPUT_DIR, config.DEBUG_HEATMAP_DIR, "2d_heatmaps_v2")
            os.makedirs(heatmap_dir, exist_ok=True)
            # 将3d pose投影到2d图像进行可视化
            for view_idx in range(len(inputs)):
                # joints_2d: [batch_size, num_person , num_joints, 3]
                save_heatmaps(heatmaps[view_idx],
                              os.path.join(heatmap_dir, f"train_heatmp2d_batch{i}_view{view_idx}.jpg"))
                center = meta[view_idx]['center'][0].detach().cpu().numpy()
                scale = meta[view_idx]['scale'][0]

                width, height = center * 2
                trans = torch.as_tensor(
                    get_transform(center, scale, 0, config.NETWORK.IMAGE_SIZE),
                    dtype=torch.float,
                )
                # print("joints_3d_info: mean:", pred.cpu().detach().numpy().mean(axis=2).mean(axis=0))
                # 投影
                pose3d = pred.cpu().detach().numpy()[:, :, :, :3]
                if config.TAG == "ue":
                    cam = {}
                    # print("cameras:",meta[view_idx]['camera'])
                    for k, v in meta[view_idx]['camera'].items():
                        # k:cam_pos v:[pos,pos, pos, pos]
                        # print("k:",k,"v:",v)
                        cam[k] = v[0]
                    xy = project_pose3d_to_pose2d(config.TAG, pose3d, width=width, height=height, cam=cam).astype(
                        np.float32)
                    xy = torch.from_numpy(xy)
                elif "camera" in meta[view_idx]:
                    cam = {}
                    # print("cameras:",meta[view_idx]['camera'][0])
                    for k, v in meta[view_idx]['camera'].items():
                        # print(k,v)
                        # print(k,v.shape)
                        cam[k] = v[0]

                    pose3d = torch.from_numpy(pose3d)
                    pre_shape = pose3d.shape[:-1]
                    pose3d = torch.reshape(pose3d, (-1, 3))
                    xy = cameras.project_pose(pose3d, cam)
                    xy = torch.reshape(xy, (*pre_shape, 2))
                elif 'proj' in meta[view_idx]:

                    joints_2d = project_pose3d_to_pose2d(config.TAG, pose3d,
                                                         meta[view_idx]["proj"].cpu().detach().numpy()[0], width=width,
                                                         height=height).astype(np.float32)
                    xy = torch.from_numpy(joints_2d)
                # print("dementsions:", "j")
                xy = torch.clamp(xy, -1.0, max(width, height))

                # print("[test.py]shapes:", "xy:", xy.shape, "trans:", trans.shape)
                xy_shape = xy.shape

                xy = do_transform(xy.view(-1, 2), trans)
                xy = xy.view(xy_shape)

                joints_2d = xy.detach().cpu().numpy()

                joints_2d_vis = torch.ones((*joints_2d.shape[:3], 1))

                save_batch_image_with_joints_multi(
                    inputs[view_idx], joints_2d, joints_2d_vis, [config.MULTI_PERSON.MAX_PEOPLE_NUM] * batch_size,
                    os.path.join(pose2d_vis_dir, f"joints_2d_batch_{i}_view_{view_idx}.jpg")
                )

            for b in range(pred.shape[0]):
                preds.append(pred[b])
            if isinstance(pred, torch.Tensor):
                preds_to_save.append(pred.detach().cpu().numpy())
            else:
                preds_to_save.append(pred)
            # if isinstance()

            prefix2 = '{}_{:08}'.format(
                os.path.join(final_output_dir, 'test'), i)
            save_debug_3d_cubes_for_test(config, meta[0], grid_centers, prefix2)

            # matplotlib.use('TkAgg')
            save_debug_3d_images_for_test(config, meta[0], pred, prefix2, show=False, save_with_timestamps=config.TEST.SAVE_WITH_TIMESTAMPS)
            # matplotlib.use('Agg')
    preds_to_save = np.array(preds_to_save).tolist()

    if config.RESULTS_DIR:
        json_dir = os.path.join(config.OUTPUT_DIR, config.RESULTS_DIR, "json")
        os.makedirs(json_dir, exist_ok=True)
        data = dict(
            preds_to_save=preds_to_save
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = os.path.join(json_dir, f"pose_preds_{timestamp}.json")

        # 保存数据
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
