# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

sys.path.extend(["."])

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from prettytable import PrettyTable
from tqdm import tqdm

from lib.core.config import config
from lib.core.config import update_config
from lib.utils.utils import create_logger
from lib.models import get_model
from lib.dataset import get_dataset


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
        config, args.cfg, 'eval_map')
    cfg_name = os.path.basename(args.cfg).split('.')[0]

    gpus = [int(i) for i in config.GPUS.split(',')]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = get_dataset(config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

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
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        # print(test_model_file)
        model.module.load_state_dict(torch.load(test_model_file), strict=False) #strict==False非严格匹配
    else:
        raise ValueError('Check the model file for testing!')

    # 模型预测
    model.eval()
    preds = []
    with torch.no_grad():
        # 模型预测
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            if 'panoptic' in config.DATASET.TEST_DATASET or "association4d" in config.DATASET.TEST_DATASET or "association4d_v2" in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(views=inputs, meta=meta)
            elif config.DATASET.TEST_DATASET in ["ue_dataset"]:
                pred, _, _, _, _, _ = model(views=inputs, meta=meta)
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)

            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

        # print("preds",preds)
        # print("inputs",inputs)

        # 计算指标，打印结果
        tb = PrettyTable()
        # #############




        ####  童雯雯看这里！！！！
        ###  把ue_dataset 加到下面这里的判断条件里面

        ##########
        if 'panoptic' in config.DATASET.TEST_DATASET or "association4d" in config.DATASET.TEST_DATASET \
                or "association4d_v2" in config.DATASET.TEST_DATASET or "ue_dataset" in config.DATASET.TEST_DATASET:
            mpjpe_threshold = np.arange(25, 155, 25)

            aps, recs, mpjpe, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
            tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
            tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
            print(tb)
            print(f'MPJPE: {mpjpe:.2f}mm')
        else:

            actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Bone Group'] + [f'Actor {i + 1}' for i in range(len(actor_pcp))] + ['Average']
            for k, v in bone_person_pcp.items():
                tb.add_row([k] + [f'{i * 100:.1f}' for i in v] + [f'{np.mean(v) * 100:.1f}'])
            tb.add_row(['Total'] + [f'{i * 100:.1f}' for i in actor_pcp] + [f'{avg_pcp * 100:.1f}'])
            print(tb)


if __name__ == "__main__":
    main()
