import torch
from torch import nn
from torch.nn import functional as F
import argparse
from utils import _SimpleSegmentationModel, SSL_SimpleSegmentationModel
from DNet_models.DNet_blocks_2D import DFF, DLK
from config import get_config
from swin_transform.swin_transformer import SwinTransformer



__all__ = ["DeepLabV3"]

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BUS', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ProstateX/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=int,    #交叉验证的折数
                    default=3, help='cross validation')
parser.add_argument('--max_iterations', type=int,
                    default=6500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--cross_val', type=bool,
                    default=True, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_ratio', type=int, default=10,
                    help='1/labeled_ratio data is provided mask')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)




class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    '''
        in_channels: 也就是aspp的输入通道     2048
        low_level_channels： 低特征通道数    256
        num_classes: 输出通道数
        aspp_dilate: 空洞卷积的扩张率
    '''

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )  # 由256--48的降通道

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )  # 48+256    处理low_level特征经过降维后与经过aspp模块的cat   3x3卷积  和Upsample
        self._init_weight()

    def forward(self, feature):
        # print(' layer5 ', feature['low_level'].shape)
        low_level_feature = self.project(feature['low_level'])  # [4, 48, 64, 64]  layer1
        output_feature = self.aspp(feature['out'])  # [4, 256, 16, 16]   layer4
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)  # [4, 256, 64, 64]
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    '''使用空洞卷积实现的aspp卷积快'''
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    '''全局平均池化部分'''
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    '''
        in_channels = 2048
        aspp_dilate = [6, 12, 18]
    '''
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))  #子模块1：1x1卷积 + BatchNorm2d + ReLU    由2048向256降维

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))  #子模块2：ASPPConv  rate1=6
        modules.append(ASPPConv(in_channels, out_channels, rate2))  #子模块3：ASPPConv  rate1=12
        modules.append(ASPPConv(in_channels, out_channels, rate3))  #子模块4：ASPPConv  rate1=18
        modules.append(ASPPPooling(in_channels, out_channels))      #子模块5：ASPPPooling  全局平均池化

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)      #Encoder最后的降维1/5

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)   #aspp之后的拼接特征
        return self.project(res)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module

#--------------------------------------------------------  SSL实验   ----------------------------------------------------
"""
    实验有效性分析
"""


class SSLDeepLab(SSL_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, C, H * W).permute(0, 2, 1)  # (N, H*W, C)
        qkv = self.qkv(x)  # (N, H*W, 3*embed_dim)
        qkv = qkv.view(N, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        attn_weights = attn_weights.softmax(dim=-1)

        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(N, H * W, -1)
        out = self.fc_out(out).permute(0, 2, 1).view(N, C, H, W)
        return out

class GGCA(nn.Module):  #(Global Grouped Coordinate Attention) 全局分组坐标注意力
    def __init__(self, channel, h, w, reduction=16, num_groups=4): #channel=640, h=56, w=56, reduction=16, num_groups=4
        super(GGCA, self).__init__()
        self.num_groups = num_groups  # 分组数4
        self.group_channels = channel // num_groups  # 每组的通道数    [4, 4, 160, 56, 56]
        self.h = h  # 高度方向的特定尺寸
        self.w = w  # 宽度方向的特定尺寸

        # 定义H方向的全局平均池化和最大池化
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))  # 输出大小为(h, 1)
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
        # 定义W方向的全局平均池化和最大池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))  # 输出大小为(1, w)
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))

        # 定义共享的卷积层，用于通道间的降维和恢复
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1))
        )
        # 定义sigmoid激活函数
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()   # ->  4, 640, 56, 56, reduction=16, num_groups=4
        # 确保通道数可以被分组数整除,一般分组数,要选择整数,不然不能被整除。而且是小一点.groups选择4挺好。
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        # 将输入特征图按通道数分组
        x = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        # 分别在H方向进行全局平均池化和最大池化
        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)

        # 分别在W方向进行全局平均池化和最大池化
        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用共享卷积层进行特征处理
        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, self.h, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, self.h, 1))

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, self.w))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, self.w))

        # 计算注意力权重
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, self.h, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用注意力权重
        out = x * att_h * att_w
        out = out.view(batch_size, channel, height, width)

        return out

class SSLDeepLabPlus(nn.Module):
    '''
        in_channels: 也就是aspp的输入通道     2048
        low_level_channels： 低特征通道数    256
        num_classes: 输出通道数
        aspp_dilate: 空洞卷积的扩张率
        layer1:[4, 256, 56, 56]
        layer2:[4, 512, 28, 28]
        layer3:[4, 1024, 14, 14]
        layer4:[4, 2048, 14, 14]
    '''

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(SSLDeepLabPlus, self).__init__()

        self.DLK_low_level_feature_1 = DLK(dim=256)
        self.DLK_low_level_feature_2 = DLK(dim=512)
        self.project_3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # 1024--48的降通道

        self.deproject = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        #这里
        self.transconv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.layer = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )  # low_level_channels--48的降通道
        self.classifier = nn.Sequential(
            nn.Conv2d( 640, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.classifier_CNN = nn.Sequential(
            nn.Conv2d(128+256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.classifier_Trans = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self.classifier_raw = nn.Sequential(
            nn.Conv2d(128+256+256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self.DFF = DFF(dim=256)


        self.swintransform = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=1,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=7,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT).cuda()

        self.Muattention = MultiHeadAttention(embed_dim=640, num_heads=8)
        self.GapAtt = GGCA(channel= 128+256+256, h=56, w=56, reduction=16, num_groups=4)
        self._init_weight()

    def forward(self, x, feature):
        ''' transform 特征 '''
        transform_feature = self.swintransform(x)  # [4, 768, 7, 7]

        ''' 处理低特征 '''
        #点1： 低维特征捕获多尺度信息
        #  DLK的消融
        low_level_feature = self.DLK_low_level_feature_1(feature['layer1'])  #[4, 256, 56, 56]  layer1
        low_level_feature_2 = self.DLK_low_level_feature_2(feature['layer2'])
        low_level_feature_2 = F.interpolate(low_level_feature_2, size=low_level_feature.shape[2:], mode='bilinear',
                                            align_corners=False)
        low_level_feature = self.deproject(
            torch.cat([low_level_feature, low_level_feature_2], dim=1))  # low_level_feature [4, 128, 56, 56]
        #----------------------------------------------------------------
        #
        # low_level_feature = self.layer( feature['layer1'] )  #[4, 48, 64, 64]  layer1

        #----------------------------------------------------------------

        output_feature = self.aspp(feature['out'])  # output_feature [4, 256, 14, 14]
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)  # [4, 256, 56, 56]
        CNN_feature = torch.cat([low_level_feature, output_feature], dim=1)  #[4, 384, 56, 56]
        #----------------------------------------------------------------
        # low_level_feature 低维特征
        # output_feature 编码器特征
        # transform_feature transform特征
        #点2：高低维信息的融合
        # CNN_feature = self.DFF(output_feature, low_level_feature)

        #transform_feature
        transform_feature = self.transconv(F.interpolate(transform_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False))     # [4, 256, 56, 56]
        # 02 加权
        # alpha = 0.5  # 可以通过训练或超参数调整确定
        # beta = 1.0 - alpha
        # fused_feature = alpha * CNN_feature + beta * transform_feature
        #
        # return self.classifier(fused_feature)

        # 03 注意力机制
        # cnn_attention = self.ca(CNN_feature)
        # transformer_attention = self.ca(transform_feature)
        #
        # fused_feature = cnn_attention * CNN_feature + transformer_attention * transform_feature
        # return self.classifier(fused_feature)

        # #04 逐像素相乘
        # fused_feature = CNN_feature * transform_feature
        # return self.classifier(fused_feature)

        # #01 直接拼接
        # return self.classifier(torch.cat([CNN_feature, transform_feature], dim=1))

        #05 自注意力机制Mu
        # 点3更加柔和的CNN-transform
        fused_feature_raw = torch.cat([CNN_feature, transform_feature], dim=1)   #[4, 640, 56, 56]
        fused_feature = self.GapAtt(fused_feature_raw)

        # fused_feature = self.Muattention(fused_feature_raw)
        # print(CNN_feature.shape, transform_feature.shape, fused_feature_raw.shape, fused_feature.shape)   # 384   256   640
        return (self.classifier_CNN(CNN_feature),  self.classifier_Trans(transform_feature),
                self.classifier_raw(fused_feature_raw), self.classifier(fused_feature))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SSLDeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(SSLDeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
