import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.resnet_dilation import resnet50, Bottleneck, conv1x1
from SDFM import (SDFM, DenseTransLayer,)
from MDEM import DFM
from models.BaseBlocks import BasicConv_PRelu
import torchvision
class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(32, 16) # 32 16
        self.up2_low = nn.Conv2d(256, 16, 1, stride=1, padding=0,bias=False)
        self.up2_bn2 = nn.BatchNorm2d(16)
        self.up2_act = nn.PReLU(16)
        self.refine=nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.PReLU())

    def forward(self, low_fea,high_fea):
        high_fea = self.up2_high(high_fea) # c 16
        low_fea = self.up2_bn2(self.up2_low(low_fea)) # c 16
        refine_feature = self.refine(self.up2_act(high_fea+low_fea)) # 卷积层
        return refine_feature

class DilatedParallelConvBlockD2(nn.Module): # 表面像是降通道的
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.)) # 向上取整数
        n2 = nOut - n # 这个不就是减去了一半
        #这里有个问题是既然是降低了，为什么还要按照通道分开，这里没有提到
        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False) # 降低了维度

        self.bn = nn.BatchNorm2d(nOut)
        #self.act = nn.PReLU(nOut)
        self.add = add
    # 在通道上进行不同的空洞操作类似于八度卷积吗
    def forward(self, input):
        in0 = self.conv0(input) # 先改通道
        in1, in2 = torch.chunk(in0, 2, dim=1) # 按照通道数分块
        b1 = self.conv1(in1) # 空洞率1
        b2 = self.conv2(in2) # 空洞率2
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)
        #output = self.act(output) # 为什么不加relu了

        return output

class DualFastnet(nn.Module):
    def __init__(self, channel=32):  # ,down_factor=4
        super(DualFastnet, self).__init__()
        # num_of_feat = 512
        # 这里是两个encoder
        self.Res50_depth = resnet50(pretrained=True, output_stride=16, input_channels=3)
        self.Res50_rgb = resnet50(pretrained=True, output_stride=16, input_channels=3)
        # 这是特征融合的层

        self.translayer = DenseTransLayer(32, 32)
        # 动态卷积融合

        self.selfdc = SDFM(32, 32, 32, 3, 4)
        self.decoder_plus_rgb = DFM()
        self.decoder_plus_depth = DFM()
        # transfor
        self.tranposelayer_rgb3 = BasicConv_PRelu(512,32,1)
        self.tranposelayer_rgb4 = BasicConv_PRelu(1024,32,1)
        self.tranposelayer_rgb5 = BasicConv_PRelu(2048,32,1)
        self.tranposelayer_depth3 = BasicConv_PRelu(512,32,1)
        self.tranposelayer_depth4 = BasicConv_PRelu(1024,32,1)
        self.tranposelayer_depth5 = BasicConv_PRelu(2048,32,1)

        self.combine=conbine_feature()
        # Drop这里有什么用
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1),nn.Conv2d(16, 1, kernel_size=1,bias=False))

        # # 这里是上采样
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)


    def forward(self,rgb,depth):
        # 两个encoder分别获得

        # rgb net
        block0 = self.Res50_rgb.conv1(rgb)
        block0 = self.Res50_rgb.bn1(block0)
        block0 = self.Res50_rgb.relu(block0)  # 256x256
        block0 = self.Res50_rgb.maxpool(block0)  # 128x128
        frist_rgb = self.Res50_rgb.layer1(block0)  # 64x64
        conv3_rgb = self.Res50_rgb.layer2(frist_rgb)  # 32x32
        conv4_rgb = self.Res50_rgb.layer3(conv3_rgb)  # 16x16
        conv5_rgb = self.Res50_rgb.layer4(conv4_rgb)  # 8x8

        # depth net
        block0_im = self.Res50_depth.conv1(depth)
        block0_im = self.Res50_depth.bn1(block0_im)
        block0_im = self.Res50_depth.relu(block0_im)
        block0_im = self.Res50_depth.maxpool(block0_im)
        frist_depth = self.Res50_depth.layer1(block0_im) # 256
        conv3_depth = self.Res50_depth.layer2(frist_depth)
        conv4_depth = self.Res50_depth.layer3(conv3_depth)
        conv5_depth = self.Res50_depth.layer4(conv4_depth)


        # transpose
        conv3_rgb = self.tranposelayer_rgb3(conv3_rgb)
        conv4_rgb = self.tranposelayer_rgb4(conv4_rgb)
        conv5_rgb = self.tranposelayer_rgb5(conv5_rgb)
        conv3_depth = self.tranposelayer_depth3(conv3_depth)
        conv4_depth = self.tranposelayer_depth4(conv4_depth)
        conv5_depth = self.tranposelayer_depth5(conv5_depth)


        # scale fuse
        rgb_final = self.decoder_plus_rgb(conv3_rgb, conv4_rgb,conv5_rgb)
        depth_final = self.decoder_plus_depth(conv3_depth,conv4_depth, conv5_depth) #1/8


        # DDPM
        trans_rgb = self.translayer(rgb_final,depth_final)


        rgb_high_feature_dy = self.selfdc(rgb_final,trans_rgb)+rgb_final

        # decoder
        # rgb decoder
        rgb_final = F.interpolate(rgb_high_feature_dy, size=(frist_rgb.shape[-2], frist_rgb.shape[-1]),
                                  mode="bilinear",
                                  align_corners=False)
        rgb_final = self.combine(frist_rgb,rgb_final) # 1/8

        rgb_final = F.interpolate(self.SegNIN(rgb_final), size=(rgb.shape[-2], rgb.shape[-1]), mode="bilinear",align_corners=False)

        return rgb_final

if __name__=="__main__":
    # from torchstat import stat
    a = torch.zeros(1, 3, 256, 256).cuda()
    b = torch.zeros(1, 3, 256, 256).cuda()

    mobile = DualFastnet().cuda()
    c = mobile(a, b)
    print(c.size())
    total_paramters = sum([np.prod(p.size()) for p in mobile.parameters()])
    print('Total network parameters: ' + str(total_paramters / 1e6) + "M")