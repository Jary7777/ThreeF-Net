#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/4/12 14:44
# File:DNet_blocks_3D.py
# ------------------------------

from torchinfo import summary
import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
    '''多层感知机'''
    def __init__(self, dim):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)
        self.dwconv = nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4) #可分离卷积
        self.act = nn.GELU() #激活
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DLK(nn.Module):
    '''空间注意力'''
    def __init__(self, dim):
        super().__init__()
        #多尺度特征提取膨胀卷积层
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, groups=dim, dilation=2)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=6, groups=dim, dilation=3)
        self.att_conv3 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=12, groups=dim, dilation=4)
        self.att_conv4 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim, dilation=1)

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=7, padding=3),
            nn.Sigmoid()
        )  #计算空间注意力权重

    def forward(self, x):
        # 通过不同卷积层提取多尺度特征
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)
        att3 = self.att_conv3(att2)
        att4 = self.att_conv4(att3)

        # 将不同尺度的特征拼接起来
        att = torch.cat([att1, att2, att3, att4], dim=1)

        # 计算平均值和最大值
        avg_att = torch.mean(att, dim=1, keepdim=True)  # 平均值
        max_att, _ = torch.max(att, dim=1, keepdim=True)  # 最大值

        # 拼接平均值和最大值，生成空间注意力图
        att = torch.cat([avg_att, max_att], dim=1)  # batch_size, 2, height, width

        # 通过空间注意力机制调整权重
        att = self.spatial_se(att)  # 计算空间注意力权重

        # 将不同尺度的特征与注意力权重结合
        output = (
                att1 * att[:, 0, :, :].unsqueeze(1) +
                att2 * att[:, 1, :, :].unsqueeze(1) +
                att3 * att[:, 2, :, :].unsqueeze(1) +
                att4 * att[:, 3, :, :].unsqueeze(1)
        )

        # 最终输出结果与输入特征相加，形成残差连接
        output = output + x
        return output


#动态大核DLK方法
class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)  #注意力门控
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)  #1x1 conv
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class DLKBlock(nn.Module):
    '''DLKs Block'''
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim, eps=1e-6) #输入归一化
        self.attn = DLKModule(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)

        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)

        return x

class Encoder(nn.Module):
    ''' Encoder class'''
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  #存储下采样层
        stem = nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)  #网络的起点
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            self.downsample_layers.append(downsample_layer)     #构建剩余的下采样层，将特征维度减半

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = nn.LayerNorm(dims[i], eps=1e-6)
            self.norm_layers.append(norm_layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.stages[i](x)
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Decoder(nn.Module):
    '''Decoder'''
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()

        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            upsample_layer = nn.ConvTranspose2d(dims[-i - 1], dims[-i - 2], kernel_size=2, stride=2)
            self.upsample_layers.append(upsample_layer)

        stem = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, stride=2)
        self.upsample_layers.append(stem)

        self.steps = nn.ModuleList()
        for i in range(4):
            step = DFF(dims[-i - 1])
            self.steps.append(step)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[-i - 1], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(3):
            norm_layer = nn.LayerNorm(dims[-i - 2], eps=1e-6)
            self.norm_layers.append(norm_layer)

        norm_layer = nn.LayerNorm(dims[0], eps=1e-6)
        self.norm_layers.append(norm_layer)

    def forward(self, x, skips):
        for i in range(4):
            x = self.steps[i](x, skips[-i - 1])
            x = self.stages[i](x)
            x = self.upsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.stage = nn.Sequential(
            DLKBlock(dim=out_chans),
            DLKBlock(dim=out_chans)
        )

        self.downsample_layer = nn.Conv2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.upsample_layer = nn.ConvTranspose2d(out_chans, in_chans, kernel_size=2, stride=2)
        self.norm_layer1 = nn.LayerNorm(out_chans, eps=1e-6)
        self.norm_layer2 = nn.LayerNorm(in_chans, eps=1e-6)

    def forward(self, x):
        x = self.downsample_layer(x)
        x = channel_to_last(x)
        x = self.norm_layer1(x)
        x = channel_to_first(x)

        x = self.stage(x)

        x = self.upsample_layer(x)
        x = channel_to_last(x)
        x = self.norm_layer2(x)
        x = channel_to_first(x)
        return x

#动态特征融合 (DFF) 模块
class DFF(nn.Module):
    '''实现深度特征融合'''
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):   #CNN   transform
        output = torch.cat([x, skip], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att

        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output

class Convblock(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output

def channel_to_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)

def channel_to_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图

# 示例使用
if __name__ == '__main__':
    # model = Encoder(
    #     in_chans=3,
    #     depths=[2, 2, 2, 2],
    #     dims=[64, 128, 256, 512],
    #     drop_path_rate=0
    # )
    # # summary(model, (4, 3, 256, 256))
    model = DFF(256)
    summary(model, ((1, 256, 64, 64),(1, 256, 64, 64)))