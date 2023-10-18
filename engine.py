# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from ctypes import util
import math
from operator import gt
import sys
import os
from xml.etree.ElementPath import xpath_tokenizer
from matplotlib import image
import numpy as np
from typing import Iterable, Optional
import pdb
import torch
import cv2
import torch.nn as nn
from pathlib import Path

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import matplotlib.pyplot as plt


def train_one_epoch_our(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch + 1)
    print_freq = 10
    cls_criterion = torch.nn.CrossEntropyLoss().to(torch.device("cuda"))

    for samples, targets, paths in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            x_logits, attn_logits = model(samples, targets=targets)

            (loss, learn_weights, loss_list) = model.module.get_loss(
                x_logits, attn_logits, targets
            )

        loss_x, loss_csm, loss_ocm_s, loss_ocm_a = loss_list

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_x=loss_x.item())
        metric_logger.update(loss_csm=loss_csm.item())
        metric_logger.update(loss_ocm_s=loss_ocm_s.item())
        metric_logger.update(loss_ocm_a=loss_ocm_a.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if learn_weights != []:
        print("** Learning Weights: ", end="")
        for i, w in enumerate(learn_weights):
            print("Weight[{_id:d}]: {weight:.3f}  ".format(_id=i, weight=w), end="")
        print("\n** ** **")
    print("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_our(data_loader, model, device, args=None, threshold_loc=-1):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    if args.input_size == 224:
        cam_wh = 14
    else:
        cam_wh = 24

    LocSet = []
    IoUSet = []
    IoUSetTop5 = []

    for images, target, paths, bboxes in metric_logger.log_every(
        data_loader, 10, header
    ):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)  # xyxy
        ims = utils.tensor2image(
            images.clone().detach().cpu(), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        )

        # compute output
        with torch.cuda.amp.autocast():
            (output, cams, _, _, _) = model(images, return_cam=True)

        # TODO classification
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if threshold_loc != -1:
            # TODO localization
            _, logits = output.topk(5, 1, True, True)
            for _b in range(images.shape[0]):
                cam_map_ = torch.mean(cams[_b, logits[_b, 0:1], :, :], 0).view(
                    cam_wh, cam_wh, 1
                )
                cam_map_r = cam_map_.detach().cpu().numpy()
                cam_map_ = utils.resize_cam(
                    cam_map_r, size=(args.input_size, args.input_size)
                )
                estimated_bbox, thr_gray_heatmap = utils.get_bboxes(
                    cam_map_, cam_thr=threshold_loc
                )

                # * compute loc acc
                max_iou = -1
                iou = utils.IoU(bboxes[_b], estimated_bbox)
                # 1. GT acc
                if iou > max_iou:
                    max_iou = iou
                LocSet.append(max_iou)
                draw_iou = max_iou
                # 2. top1
                temp_loc_iou = max_iou
                if logits[_b][0] != target[_b]:
                    max_iou = -1
                IoUSet.append(max_iou)
                # 3. top5
                max_iou = -1
                for i in range(5):
                    if logits[_b][i] == target[_b]:
                        max_iou = temp_loc_iou
                        break
                IoUSetTop5.append(max_iou)

        batch_size = images.shape[0]
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        if threshold_loc != -1:
            # * compute loc acc
            loc_acc_top1 = np.sum(np.array(IoUSet) >= 0.5) / len(IoUSet)
            loc_acc_top5 = np.sum(np.array(IoUSetTop5) >= 0.5) / len(IoUSetTop5)
            loc_acc_gt = np.sum(np.array(LocSet) >= 0.5) / len(LocSet)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Loc Acc@1 \033[32m{top1:.3f}\033[0m Acc@5 {top5:.3f} GT \033[32m{gt:.3f}\033[0m TH \033[4m{th:.3f}\033[0m TestNum {tn:d}".format(
            top1=loc_acc_top1 * 100,
            top5=loc_acc_top5 * 100,
            gt=loc_acc_gt * 100,
            th=threshold_loc,
            tn=len(LocSet),
        )
    )
    print(
        "* Acc@1 \033[32m{top1.global_avg:.3f} \033[0m Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )

    if threshold_loc != -1:
        print(
            "{top1_loc:.3f} {top5_loc:.3f} {gt_loc:.3f} {th:.3f} {top1_cls.global_avg:.3f} {top5_cls.global_avg:.3f}".format(
                top1_loc=loc_acc_top1 * 100,
                top5_loc=loc_acc_top5 * 100,
                gt_loc=loc_acc_gt * 100,
                th=threshold_loc,
                top1_cls=metric_logger.acc1,
                top5_cls=metric_logger.acc5,
            )
        )
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, [
            loc_acc_top1,
            loc_acc_top5,
            loc_acc_gt,
        ]

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
