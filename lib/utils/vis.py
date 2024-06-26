                                   # ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import math
from datetime import datetime

import numpy as np
import torchvision
import torch
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def save_heatmaps(heatmaps: torch.Tensor, save_path: str):
    '''
    heatmaps: batch_size x num_joints x height x width
    '''
    # matplotlib.use("agg")
    # print("heatmaps:",heatmaps.shape)

    image = heatmaps[0].sum(dim=0)
    image = image.detach().cpu().numpy()
    from matplotlib import pyplot as plt
    import matplotlib
    # input()
    matplotlib.use("agg")
    fig = plt.figure()
    # print(image)
    plt.imshow(image)
    # output_dir = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DEBUG_HEATMAP_DIR, "2d_heatmaps")
    # output_dir=os.path.join("/home/tww/Projects/voxelpose-pytorch/output/shelf_end_to_end/multi_person_posenet_50/prn64_cpn80x80x20_960x512_cam5","2d_heatmaps")
    # print("output_dir",output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    # filename = get_next_filename(output_dir)
    plt.savefig(save_path)



# 可视化2d pose, 批量可视化
def save_batch_image_with_joints_multi(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 num_person,
                                 file_name,
                                 nrow=8,
                                 padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_person, num_joints, 3],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    '''
    # print('num_person',num_person)
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    # print("batch_joints.shape:",batch_joints.shape)
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            for n in range(num_person[k]):
                joints = batch_joints[k, n]
                # print("draw joints:", joints.mean(axis=0))
                joints_vis = batch_joints_vis[k, n]
                num_joints = joints.shape[0]  # 获取关节的数量

                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                          (127, 127, 255), (127, 255, 127), (255, 127, 127), (127, 0, 255), (127, 255, 0),
                          (255, 127, 0)]

                # 绘制关节
                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 4,[0, 255, 255], 4) #绘制黄色关节

                # 根据关节点数量自动选择对应的 LIMBS 数组并绘制连线
                color = colors[n % len(colors)]
                for idx, limb in enumerate(eval("LIMBS{}".format(num_joints))):
                    if joints_vis[limb[0], 0] and joints_vis[limb[1], 0]:
                        joint_start = (int(joints[limb[0], 0]), int(joints[limb[0], 1]))
                        joint_end = (int(joints[limb[1], 0]), int(joints[limb[1], 1]))
                        cv2.line(ndarr, joint_start, joint_end, color, 2)  # 不同人线条颜色不一样

            k = k + 1
    # print("ndarr:",ndarr)
    cv2.imwrite(file_name, ndarr)

# 将热图与原始图像合成并保存为一个图像文件，以便进行可视化或分析
def save_batch_heatmaps_multi(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    batch_image = batch_image.flip(1)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    # print("save image:", file_name, "shape:", grid_image.shape)
    # grid_image=np.transpose(grid_image, (2, 0, 1))
    cv2.imwrite(file_name, grid_image)


#每次传入一个相机的数据
def save_debug_images_multi(config, inputs, meta, target, output, prefix):
    if not config.DEBUG.DEBUG:
        return
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints_multi(inputs, meta['joints'], meta['joints_vis'], meta['num_person'], '{}_gt.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps_multi(inputs, target, '{}_hm_gt.jpg'.format(prefix2))
    # 这个地方回头要恢复，切记！！！！！！！！！
    # if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
    #     save_batch_image_with_joints_multi(input, preds, meta['joints_vis'], meta['num_person'],'{}_pred.jpg'.format(prefix1))
    # print("config.DEBUG.SAVE_HEATMAPS_PRED:", config.DEBUG.SAVE_HEATMAPS_PRED)
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        # print("save path:", '{}_hm_pred.jpg'.format(prefix2))
        # print("inputs:", inputs.shape, "outputs:", output.shape,)
        save_batch_heatmaps_multi(inputs, output, '{}_hm_pred.jpg'.format(prefix2))
    # input()

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# # h36m
# LIMBS17 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
#          [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]
#association4D
LIMBS21 = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((2,-1)).T.tolist()
#ue
# LIMBS23=np.array([0,0,0,1,1,1,3,4,6,7,9,9,9,9,9,14,16,16,16,16,16,21,
#                   1,15,22,2,5,8,4,5,7,8,10,11,12,13,14,15,17,18,19,20,21,22]).reshape((-1, 2)).tolist()
LIMBS23=np.array([0,1,0,15,0,22,1,2,1,5,1,8,3,4,4,5,6,7,7,8,9,10,9,15,10,11,10,12,10,13,10,14,
                  16,17,16,22,17,18,17,19,17,20,17,21]).reshape((-1, 2)).tolist()
print(LIMBS23)

def save_debug_3d_images(config, meta, preds, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')
    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    # preds = preds.cpu().numpy()
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        # print("num_person:",num_person)
        for n in range(num_person):
            joint = joints_3d[n]
            # print("gt_joint:",joint)
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    if config.TAG == "ue":
                        ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    if config.TAG == "ue":
                        ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            # print("pred:", pred)
            for n in range(len(pred)):
                joint = pred[n]
                # print('pred_joint:',joint)
                if joint.shape[1]<4 or joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    plt.savefig(file_name)

    plt.close(0)

def save_debug_3d_images_for_test(config, meta, preds, prefix,show=False, save_with_timestamps=False):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')
    if not os.path.exists(dirname1):
        os.makedirs(dirname1)
    prefix = os.path.join(dirname1, basename)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_with_timestamps:
        file_name = prefix + f"_3d_{timestamp}.png"
    else:
        file_name = prefix + f"_3d.png"

    # preds = preds.cpu().numpy()
    # batch_size = meta['num_person'].shape[0]
    batch_size = len(preds)
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)

    for i in range(batch_size):
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    if show:
        plt.show()
        input("正在显示图片，等待中...")
    plt.savefig(file_name)
    plt.close(0)



def save_debug_3d_cubes(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'root_cubes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]
    root_id = config.DATASET.ROOTIDX

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 6.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        roots_gt = meta['roots_3d'][i]
        num_person = meta['num_person'][i]
        # print("num_person",num_person)

        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')

        x = roots_gt[:num_person, 0].cpu()
        y = roots_gt[:num_person, 1].cpu()
        z = roots_gt[:num_person, 2].cpu()
        ax.scatter(x, y, z, c='r')

        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c='b')

        space_size = config.MULTI_PERSON.SPACE_SIZE
        space_center = config.MULTI_PERSON.SPACE_CENTER
        ax.set_xlim(space_center[0] - space_size[0] / 2, space_center[0] + space_size[0] / 2)
        ax.set_ylim(space_center[1] - space_size[1] / 2, space_center[1] + space_size[1] / 2)
        ax.set_zlim(space_center[2] - space_size[2] / 2, space_center[2] + space_size[2] / 2)

    plt.savefig(file_name)
    plt.close(0)

def save_debug_3d_cubes_for_test(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'root_cubes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 6.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c='b')

        space_size = config.MULTI_PERSON.SPACE_SIZE
        space_center = config.MULTI_PERSON.SPACE_CENTER
        ax.set_xlim(space_center[0] - space_size[0] / 2, space_center[0] + space_size[0] / 2)
        ax.set_ylim(space_center[1] - space_size[1] / 2, space_center[1] + space_size[1] / 2)
        ax.set_zlim(space_center[2] - space_size[2] / 2, space_center[2] + space_size[2] / 2)

    plt.savefig(file_name)
    plt.close(0)


import matplotlib.pyplot as plt
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize

from torchvision.transforms.functional import to_tensor


def resize_heatmap(heatmap, target_size):
    """
    Resize a single heatmap to match the target size.
    Adds batch and channel dimensions before resizing and removes them afterward.

    Parameters:
    - heatmap: A single heatmap tensor.
    - target_size: The target size as (width, height).

    Returns:
    - Resized heatmap as a 2D tensor.
    """
    # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    # Resize and remove added dimensions: [1, 1, H', W'] -> [H', W']
    resized_heatmap = resize(heatmap, target_size)[0][0]
    return resized_heatmap


def visualize_heatmaps_on_images(inputs, heatmaps, save_path):
    """
    Visualize multi-view poses with heatmaps overlaid on the original images.

    Parameters:
    - inputs: List of input images with shape [n_cameras, batch_size, 3, H, W]
    - heatmaps: List of heatmaps with shape [n_cameras, batch_size, n_joints, H', W']
    - save_path: Path to save the visualization image
    """
    n_cameras, batch_size = len(inputs), len(inputs[0])
    fig, axes = plt.subplots(batch_size, n_cameras, figsize=(n_cameras * 5, batch_size * 5))

    for batch_idx in range(batch_size):
        for cam_idx in range(n_cameras):
            img = to_pil_image(inputs[cam_idx][batch_idx])
            heatmap = heatmaps[cam_idx][batch_idx].cpu().detach()
            heatmap = torch.mean(heatmap, 0)  # Take the average to get a single heatmap
            heatmap_resized = resize_heatmap(heatmap, img.size[::-1])  # Ensure this matches your image size correctly
            heatmap = np.array(heatmap_resized)
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            overlay = np.array(img).astype(np.float32).copy()
            # overlay[:, :, 0] = overlay[:, :, 0] * (
            #             1 - heatmap_normalized) + heatmap_normalized * 255  # Apply heatmap to red channel
            # overlay[:, :, :]+= heatmap_normalized * 255  # Apply heatmap to red channel

            if batch_size == 1:
                axes[cam_idx].imshow(heatmap_normalized)
                axes[cam_idx].axis('off')
            else:
                axes[batch_idx, cam_idx].imshow(heatmap_normalized)
                axes[batch_idx, cam_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
