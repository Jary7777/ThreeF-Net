#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2025/1/26 11:00
# File:ThreeF-Net.py
# ------------------------------
import torch
from utils import IntermediateLayerGetter
from _deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, SSLDeepLabPlus, SSLDeepLab, SSLDeepLabHead
from backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)
from torchinfo import summary


def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):
    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256  # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36]  # If follow paper trend, can put [24, 48, 72].

    if name == 'deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    '''deeplabv3plus', 'resnet101', output_stride=16 '''
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]  # True 选择空洞卷积以 增加感受野而不减小输出的空间分辨率
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)  #
    inplanes = 2048  # resnet最后的特征维度
    low_level_planes = 256

    if name == 'deeplabv3plus':
        '''
        layer1:[4, 48, 64, 64]
        layer2:[4, 512, 32, 32]
        layer2:[4, 1024, 16, 16]
        layer2:[4, 2048, 16, 16]
        '''
        return_layers = {'layer4': 'out', 'layer3': 'layer3', 'layer2': 'layer2',
                         'layer1': 'low_level'}  # DeepLabV3+返回最后一层（layer4）和低层特征（layer1）
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}  # DeepLabV3只返回最后一层
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 提取特定层

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = xception.xception(pretrained='imagenet' if pretrained_backbone else False,
                                 replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 128

    if name == 'deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == 'mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone == 'xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride,
                               pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model



def SSL_segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    '''deeplabv3plus', 'resnet101', 1, 16, Ture '''
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]  # True 选择空洞卷积以 增加感受野而不减小输出的空间分辨率
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)  #

    inplanes = 2048  # resnet最后的特征维度
    low_level_planes = 256  # 低维特征

    if name == 'deeplabv3plus':
        '''
        rsnet101的backbone下
        layer1:[4, 256, 64, 64]
        layer2:[4, 512, 32, 32]
        layer2:[4, 1024, 16, 16]
        layer2:[4, 2048, 16, 16]
        '''
        return_layers = {'layer4': 'out', 'layer3': 'layer3', 'layer2': 'layer2',
                         'layer1': 'layer1'}  # DeepLabV3+返回最后一层（layer4）和低层特征（layer1）
        classifier = SSLDeepLabPlus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}  # DeepLabV3只返回最后一层
        classifier = SSLDeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 提取特定层

    model = SSLDeepLab(backbone, classifier)
    return model


def ThreeFNet(num_classes=1, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return SSL_segm_resnet('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)  #


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_tensor = torch.randn(4, 3, 224, 224).to(device)
    model = ThreeFNet(num_classes=1, output_stride=16, pretrained_backbone=True).to(device)

    # summary(model, input_size=(4, 3, 224, 224))

    output = model(input_tensor)
    print(len(output))

    print("Output shape:", output[0].shape)
    print("Output shape:", output[1].shape)
    print("Output shape:", output[2].shape)
    print("Output shape:", output[3].shape)
