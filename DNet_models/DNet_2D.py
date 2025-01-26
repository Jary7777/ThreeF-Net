#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/4/12 15:03
# File:DNet_2D.py
# ------------------------------

import torch
import torch.nn as nn

from DNet_blocks_2D import Encoder, Decoder, Bottleneck, Convblock, DFF
from torchinfo import summary

class DNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            depths=[2, 2, 2, 2],
            feat_size=[64, 128, 256, 512],
            bottom_feat=1024,
            drop_path_rate=0
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.bottom_feat = bottom_feat

        self.dnet_down = Encoder(
            in_chans=self.in_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate
        )

        self.dnet_up = Decoder(
            in_chans=self.in_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate
        )

        self.bottleneck = Bottleneck(
            in_chans=self.feat_size[-1],
            out_chans=self.bottom_feat
        )

        self.saliency = Convblock(in_channels, self.feat_size[0])
        self.final_layer = Convblock(self.feat_size[0], self.feat_size[0])
        self.dff = DFF(self.feat_size[0])

        self.output_layer = nn.Conv2d(self.feat_size[0], self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        outs = self.dnet_down(x)
        hidden = self.bottleneck(outs[-1])
        output = self.dnet_up(hidden, outs)
        top = self.saliency(x)
        output = self.dff(output, top)
        output = self.final_layer(output)
        output = self.output_layer(output)
        return output


if __name__ == '__main__':

    data = torch.rand((4, 3, 256, 256))

    model = DNet(
        in_channels=3,
        out_channels=1,
        depths=[2, 2, 2, 2],
        feat_size=[32, 64, 128, 256],
        bottom_feat=512,
        drop_path_rate=0
    )
    summary(model, (4, 3, 256, 256))

    # import hiddenlayer as hl
    # g = hl.build_graph(model, data,transforms=None)
    # g.save("network_architecture.pdf")
    # del g