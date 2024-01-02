from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

from lib.utils.vis import save_debug_images_multi
from lib.utils.vis import save_debug_3d_images
from lib.utils.vis import save_debug_3d_cubes
from tqdm import tqdm
logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict, device=torch.device('cuda'), dtype=torch.float):

    # 一些统计数据的工具类，例如统计平均损失等
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()

    model.train() #model：nn.module  将模型调整为训练模式 ，会缓存数据,便于反向传播时计算梯度，以及更新batchnorm参数


    if model.module.backbone is not None:
        # Comment out this line if you want to train 2D backbone jointly
        # 把backone调整为测试模式
        model.module.backbone.eval()

    accumulation_steps = 4
    accu_loss_3d = 0

    end = time.time()
    from wk import Timer
    timer = Timer(mute=True)  # mute=True Mute:使安静， 让它不要打印计时器的信息
    # inputs，输入图像
    # targets_2d, 2D真值(热图)
    # weights_2d, 表示关节点是否参与loss计算
    # targets_3d, 3D真值
    # meta, 存储数据集相关信息，例如相机参数
    # input_heatmap 输入2D的heatmap
    for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        # print("targets_3d",targets_3d)
        # print("meta[joints_3d]",meta["joints_3d"].shape)
        timer.step("数据加载")
        if config.TRAIN_2D_ONLY:
            # 模型推理
            result = model(views=inputs, meta=meta,
                           targets_2d=targets_2d,
                           weights_2d=weights_2d,
                           targets_3d=targets_3d[0])
            heatmaps, loss_2d = result #heatmaps is predicted
            loss_2d = loss_2d.mean()
            losses_2d.update(loss_2d.item())
            loss = loss_2d
            losses.update(loss.item())
        else:
            if 'panoptic' in config.DATASET.TEST_DATASET or "association4d" in config.DATASET.TEST_DATASET or "association4d_v2" in config.DATASET.TEST_DATASET or "ue_dataset" in config.DATASET.TEST_DATASET or "shelf_end_to_end" in config.DATASET.TEST_DATASET:
                # print(targets_3d[0].shape)
                result = model(views=inputs, meta=meta,
                               targets_2d=targets_2d,
                               weights_2d=weights_2d,
                               targets_3d=targets_3d[0])

                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = result
            elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                                  input_heatmaps=input_heatmap)

            loss_2d = loss_2d.mean()
            loss_3d = loss_3d.mean()
            loss_cord = loss_cord.mean()

            losses_2d.update(loss_2d.item())
            losses_3d.update(loss_3d.item())
            losses_cord.update(loss_cord.item())



            # 修改前代码#######+
            # loss = loss_2d + loss_3d + loss_cord
            # losses.update(loss.item())
            #
            # if loss_cord > 0:
            #     optimizer.zero_grad()
            #     (loss_2d + loss_cord).backward()
            #     optimizer.step()
            #
            # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
            #     optimizer.zero_grad()
            #     accu_loss_3d.backward()
            #     optimizer.step()
            #     accu_loss_3d = 0.0
            # else:
            #     accu_loss_3d += loss_3d / accumulation_steps
            ##########修改后代码#########

            loss = loss_2d + loss_3d + loss_cord
            losses.update(loss.item())
        # print(loss)
        timer.step("网络预测")
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step() #根据梯度调整模型参数
            optimizer.zero_grad() #梯度清零

        ##########分界线#########

        batch_time.update(time.time() - end)
        end = time.time()

        # print("heatmaps[k]：", torch.stack(heatmaps).shape)
        # print("inputs[k]：", torch.stack(inputs).shape)
        # print("targets_2d[K]_max:",torch.stack(targets_2d).max())
        # print("targets_2d[K]_min:",torch.stack(targets_2d).min())
        # print("targets_2d[K]_mean:",torch.stack(targets_2d).mean())

        if i % config.PRINT_FREQ == 0: #每100次打印一次，config.PRINT_FREQ = 100
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t' \
                  'Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t' \
                  'Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                      data_time=data_time, loss=losses, loss_2d=losses_2d, loss_3d=losses_3d,
                      loss_cord=losses_cord, memory=gpu_memory_usage)
            logger.info(msg) #打印info（debug, warning, error）级别的日志

            # 把相关数据表加入tensorboard，便于后续可视化训练过程
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss_3d', losses_3d.val, global_steps)
            writer.add_scalar('train_loss_cord', losses_cord.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # 遍历每个视野，保存可视化结果
            print('len(inputs)',len(inputs))
            for k in range(len(inputs)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{:08}_{}'.format(
                    os.path.join(output_dir, 'train'), i, view_name)
                save_debug_images_multi(
                    config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)
            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'train'), i)
            if not config.TRAIN_2D_ONLY:
                save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                save_debug_3d_images(config, meta[0], pred, prefix2)
        timer.step("后处理")


def validate_3d(config, model, loader, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(loader):
            data_time.update(time.time() - end)
            if config.TRAIN_2D_ONLY:
                # 模型推理
                result = model(views=inputs, meta=meta,
                               targets_2d=targets_2d,
                               weights_2d=weights_2d,
                               targets_3d=targets_3d[0])
                heatmaps, loss_2d = result  # heatmaps is predicted
            else:
                if 'panoptic' in config.DATASET.TEST_DATASET or 'association4d' in config.DATASET.TEST_DATASET or 'association4d_v2' in config.DATASET.TEST_DATASET or "ue_dataset" in config.DATASET.TEST_DATASET or "shelf_end_to_end" in config.DATASET.TEST_DATASET:
                    pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                                  weights_2d=weights_2d, targets_3d=targets_3d[0])

                elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                    pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                                  input_heatmaps=input_heatmap)
                pred = pred.detach().cpu().numpy()
                for b in range(pred.shape[0]):
                    preds.append(pred[b])

                batch_time.update(time.time() - end)
                end = time.time()
                if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed: {speed:.1f} samples/s\t' \
                          'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Memory {memory:.1f}'.format(
                              i, len(loader), batch_time=batch_time,
                              speed=len(inputs) *
                              inputs[0].size(0) / batch_time.val,
                              data_time=data_time, memory=gpu_memory_usage)
                    logger.info(msg)


            for k in range(len(inputs)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{:08}_{}'.format(
                    os.path.join(output_dir, 'validation'), i, view_name)
                save_debug_images_multi(
                    config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix)

            prefix2 = '{}_{:08}'.format(
                os.path.join(output_dir, 'validation'), i)
            if not config.TRAIN_2D_ONLY:
                save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                save_debug_3d_images(config, meta[0], pred, prefix2)

    print("preds", preds)
    if not config.TRAIN_2D_ONLY:
        metric = None
        if 'panoptic' in config.DATASET.TEST_DATASET or 'association4d' in config.DATASET.TEST_DATASET :
            aps, _, mpjpe, recall = loader.dataset.evaluate(preds)
            msg = 'ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
                  'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
                  'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                      aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                      aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
                  )
            logger.info(msg)
            metric = np.mean(aps)
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET or "ue_dataset" in config.DATASET.TEST_DATASET or "shelf_end_to_end" in config.DATASET.TEST_DATASET:
            actor_pcp, avg_pcp, _, recall = loader.dataset.evaluate(preds)
            # print("data:", actor_pcp[0]*100, actor_pcp[1]*100, actor_pcp[2]*100, avg_pcp[0]*100, recall )
            # print( ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
            #           pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp[0]*100, recall=recall))
            # print("actor_pcp",actor_pcp)
            # print("pcp_avg", avg_pcp )
            msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
                  ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}'.format(
                      pcp_1=actor_pcp[0]*100, pcp_2=actor_pcp[1]*100, pcp_3=actor_pcp[2]*100, pcp_avg=avg_pcp[0]*100, recall=recall)
            logger.info(msg)
            metric = np.mean(avg_pcp)

        return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
