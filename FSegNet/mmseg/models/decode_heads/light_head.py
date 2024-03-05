# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Linear
from einops import rearrange

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class upSamp(nn.Module):
    def __init__(self, in_channels, out_channels, up_ratio):
        super(upSamp, self).__init__()
        self.upLiner = ConvModule(in_channels, out_channels * up_ratio * up_ratio, kernel_size=1,
                                  norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.out_channels = out_channels
        self.up_ratio = up_ratio

    def forward(self, feature):
        feature = self.upLiner(feature)
        feature = rearrange(feature, 'b (p1 p2 c) h w-> b c (h p1) (w p2)', p1=self.up_ratio, p2=self.up_ratio,
                            c=self.out_channels)

        return feature


@MODELS.register_module()
class LightHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        for i in range(num_inputs):
            self.ups.append(
                upSamp(in_channels=self.channels,
                       out_channels=self.channels // 2,
                       up_ratio=2 ** i
                       ))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs // 2,
            out_channels=self.channels // 2,
            kernel_size=(3, 7),
            padding=(1, 3),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.dropuout = nn.Dropout(p=0.1, inplace=False)
        self.last_conv = ConvModule(
            in_channels=self.channels // 2,
            out_channels=self.channels // 4,
            kernel_size=(7, 3),
            padding=(3, 1),
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_seg = ConvModule(in_channels=self.channels // 4, out_channels=self.num_classes, kernel_size=3,
                                   stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.convs[idx](x)
            outs.append(
                self.ups[idx](x)
            )
        out = self.fusion_conv(self.dropuout(torch.cat(outs, dim=1)))
        out = self.conv_seg(self.last_conv(out))

        return out