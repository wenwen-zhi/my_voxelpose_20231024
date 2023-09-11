# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import sys

sys.path.extend(["."])
from lib.dataset import get_dataset
from lib.models import get_model

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate_3d
from lib.utils.utils import create_logger
from lib.dataset import get_dataset
from lib.models import get_model

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
        config, args.cfg, 'validate')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

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

    print('=> Constructing models ..')
    model = get_model(config.MODEL).get_multi_person_pose_net(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    print("test_model_file:", test_model_file)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        # state_dict=torch.load(test_model_file)
        # print(list(state_dict.keys()))
        # input()
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    validate_3d(config, model, test_loader, final_output_dir)


if __name__ == '__main__':
    main()
