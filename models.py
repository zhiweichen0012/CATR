# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import pdb
import torch
import torch.nn as nn
from functools import partial
import ipdb
from einops import rearrange
import torch.nn.functional as F
import math

from vision_transformer import VisionTransformer, _cfg

from timm.models.registry import register_model

import att_CNN


__all__ = [
    "deit_small_patch16_224_CATR_cub",
    "deit_small_patch16_224_CATR_imnet",
]


class CATR_cub(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(
            self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.att_layer = att_CNN.cnn(
            72,
            self.num_classes,
            c_base=8,
        )
        self.att_cls = nn.Conv2d(
            self.num_classes, self.num_classes, kernel_size=1, padding=0, bias=False
        )

        self.AutomaticWeightedLoss = AutomaticWeightedLoss(num=4)

        self.head.apply(self._init_weights)

        self.loss = torch.nn.CrossEntropyLoss().to(torch.device("cuda"))
        self.loss_mse = torch.nn.MSELoss().to(torch.device("cuda"))

        self.block0 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )

        self.block_catconv = nn.Conv2d(9, 1, kernel_size=3, padding=1, bias=False)
        self.block_catbn = nn.BatchNorm2d(1)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, targets=None, return_cam=False):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        attn_weights_c = attn_weights.copy()

        self.attn_map = attn_weights.copy()
        self.attn_map = torch.stack(self.attn_map)  # 12 * B * H * N * N
        self.attn_map = torch.mean(self.attn_map, dim=0)  #  B * H * N * N
        self.attn_map = self.attn_map.sum(1)[:, 0, 1:].reshape([-1, 14, 14])

        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()

        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights_c = torch.stack(attn_weights_c)  # ([12, 128, 6, 197, 197])
        attn_weights_c_merge = rearrange(
            attn_weights_c[..., 0, 1:],
            "h b m (hi wi) -> b (h m) hi wi",
            hi=int(p**0.5),
        )  # [128 72 14 14]
        attn_out1 = self.att_layer(attn_weights_c_merge)
        attn_out2 = self.att_cls(attn_out1)
        attn_logits = self.avgpool(attn_out2).squeeze(3).squeeze(2)

        # TODO learning map
        attn_weights_c_head = rearrange(
            attn_weights_c[..., 0, 1:],
            "h b m (hi wi) ->h b m hi wi",
            hi=int(p**0.5),
        )  # [12 128 6 14 14]
        x_block0 = self.block0(attn_weights_c_head[0])
        x_block1 = self.block1(attn_weights_c_head[1])
        x_block2 = self.block2(attn_weights_c_head[2])
        x_block3 = self.block3(attn_weights_c_head[3])
        x_block4 = self.block4(attn_weights_c_head[4])
        x_block5 = self.block5(attn_weights_c_head[5])
        x_block6 = self.block6(attn_weights_c_head[6])
        x_block7 = self.block7(attn_weights_c_head[7])
        x_block8 = self.block8(attn_weights_c_head[8])

        x_block = torch.sigmoid(
            self.block_catbn(
                self.block_catconv(
                    torch.cat(
                        [
                            x_block0,
                            x_block1,
                            x_block2,
                            x_block3,
                            x_block4,
                            x_block5,
                            x_block6,
                            x_block7,
                            x_block8,
                        ],
                        1,
                    )
                )
            )
        )
        self.pre_map = x_block
        if self.training:
            return x_logits, attn_logits
        else:
            _, logits = x_logits.topk(1, 1, True, True)

            attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=0)  #  B * H * N * N

            feature_map = x_patch.detach().clone()  # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            cams = attn_weights.sum(1)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            cams = cams * (feature_map + attn_out2.detach().clone())  # B * C * 14 * 14
            attn_weights_all = rearrange(
                attn_weights_c_merge,
                "b (h m) hi wi -> b m h hi wi",
                h=6,
            )
            feats = (feature_map + attn_out2.detach().clone()) / 2.0

            return (x_logits, cams, attn_weights_all, feats, x_block)

    def get_loss(self, x_logits, attn_logits, targets):
        loss_x = self.loss(x_logits, targets)
        loss_csm = self.loss(attn_logits, targets)
        pre_map_label = self.attn_map

        attn_map_flatt = self.attn_map.flatten(1)
        atn_v, _ = attn_map_flatt.sort()  # small to large
        atn_min = atn_v[:, int(attn_map_flatt.shape[1] * 0.25)].view(
            attn_map_flatt.shape[0], 1, 1
        )

        self.attn_map = torch.where(
            self.attn_map > torch.ones_like(self.attn_map) * atn_min,
            torch.zeros_like(self.attn_map),
            self.attn_map,
        )
        loss_ocm_s = torch.sum(self.attn_map) / 196.0

        loss_ocm_a = self.loss_mse(self.pre_map, pre_map_label)

        loss, params = self.AutomaticWeightedLoss(
            loss_x, loss_csm, loss_ocm_s, loss_ocm_a
        )

        return loss, params, [loss_x, loss_csm, loss_ocm_s, loss_ocm_a]


class CATR_imnet(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(
            self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.att_layer = att_CNN.cnn(
            72,
            self.num_classes,
            c_base=256,
        )
        self.att_cls = nn.Conv2d(
            self.num_classes, self.num_classes, kernel_size=1, padding=0, bias=False
        )

        self.AutomaticWeightedLoss = AutomaticWeightedLoss(num=4)

        self.head.apply(self._init_weights)

        self.loss = torch.nn.CrossEntropyLoss().to(torch.device("cuda"))
        self.loss_mse = torch.nn.MSELoss().to(torch.device("cuda"))

        self.block0 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
        )

        self.block_catconv = nn.Conv2d(10, 1, kernel_size=3, padding=1, bias=False)
        self.block_catbn = nn.BatchNorm2d(1)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, targets=None, return_cam=False):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        attn_weights_c = attn_weights.copy()
        # * attn weights
        self.attn_map = attn_weights.copy()
        self.attn_map = torch.stack(self.attn_map)  # 12 * B * H * N * N
        self.attn_map = torch.mean(self.attn_map, dim=0)  #  B * H * N * N
        self.attn_map = self.attn_map.sum(1)[:, 0, 1:].reshape([-1, 14, 14])

        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        # ATT LOSS
        attn_weights_c = torch.stack(attn_weights_c)  # ([12, 128, 6, 197, 197])
        attn_weights_c_merge = rearrange(
            attn_weights_c[..., 0, 1:],
            "h b m (hi wi) -> b (h m) hi wi",
            hi=int(p**0.5),
        )  # [128 72 14 14]
        attn_out1 = self.att_layer(attn_weights_c_merge)
        attn_out2 = self.att_cls(attn_out1)
        attn_logits = self.avgpool(attn_out2).squeeze(3).squeeze(2)

        # TODO learning map
        attn_weights_c_head = rearrange(
            attn_weights_c[..., 0, 1:],
            "h b m (hi wi) ->h b m hi wi",
            hi=int(p**0.5),
        )  # [12 128 6 14 14] 
        x_block0 = self.block0(attn_weights_c_head[0])
        x_block1 = self.block1(attn_weights_c_head[1])
        x_block2 = self.block2(attn_weights_c_head[2])
        x_block3 = self.block3(attn_weights_c_head[3])
        x_block4 = self.block4(attn_weights_c_head[4])
        x_block5 = self.block5(attn_weights_c_head[5])
        x_block6 = self.block6(attn_weights_c_head[6])
        x_block7 = self.block7(attn_weights_c_head[7])
        x_block8 = self.block8(attn_weights_c_head[8])
        x_block9 = self.block9(attn_weights_c_head[9])

        x_block = torch.sigmoid(
            self.block_catbn(
                self.block_catconv(
                    torch.cat(
                        [
                            x_block0,
                            x_block1,
                            x_block2,
                            x_block3,
                            x_block4,
                            x_block5,
                            x_block6,
                            x_block7,
                            x_block8,
                            x_block9,
                        ],
                        1,
                    )
                )
            )
        )

        if self.training:
            self.pre_map = x_block
            return x_logits, attn_logits
        else:
            _, logits = x_logits.topk(1, 1, True, True)
            self.pre_map = x_block

            attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=0)  #  B * H * N * N

            feature_map = x_patch.detach().clone()  # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            cams = attn_weights.sum(1)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)
            cams = (
                cams * (feature_map + attn_out2.detach().clone()) / 2.0
            )  # B * C * 14 * 14
            attn_weights_all = rearrange(
                attn_weights_c_merge,
                "b (h m) hi wi -> b m h hi wi",
                h=6,
            )
            feats = (feature_map + attn_out2.detach().clone()) / 2.0
            return (x_logits, cams, attn_weights_all, feats, x_block)

    def get_loss(self, x_logits, attn_logits, targets):
        loss_x = self.loss(x_logits, targets)
        loss_csm = self.loss(attn_logits, targets)
        pre_map_label = self.attn_map

        attn_map_flatt = self.attn_map.flatten(1)
        atn_v, _ = attn_map_flatt.sort()  # small to large
        atn_min = atn_v[:, int(attn_map_flatt.shape[1] * 0.1)].view(
            attn_map_flatt.shape[0], 1, 1
        )

        self.attn_map = torch.where(
            self.attn_map > torch.ones_like(self.attn_map) * atn_min,
            torch.zeros_like(self.attn_map),
            self.attn_map,
        )

        loss_ocm_s = torch.sum(self.attn_map) / 196.0

        loss_ocm_a = self.loss_mse(self.pre_map, pre_map_label)

        loss, params = self.AutomaticWeightedLoss(
            loss_x, loss_csm, loss_ocm_s, loss_ocm_a
        )

        return loss, params, [loss_x, loss_csm, loss_ocm_s, loss_ocm_a]


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum, self.params


@register_model
def deit_small_patch16_224_CATR_cub(pretrained=False, **kwargs):
    model = CATR_cub(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def deit_small_patch16_224_CATR_imnet(pretrained=False, **kwargs):
    model = CATR_imnet(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
