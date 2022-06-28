import torch
from torch import nn
from models.BaseBlocks import BasicConv2d,BasicConv_PRelu
import torch.nn.functional as F
def Split(x):
    c = int(x.size()[1])  # x的通道数量
    c1 = round(c * 0.5)  # 大约为0.5
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


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
            self.denseblock.append(BasicConv_PRelu(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv_PRelu(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class DenseTransLayer(nn.Module):
    def __init__(self, in_C, out_C):
        super(DenseTransLayer, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BasicConv_PRelu(in_C*2, in_C, 3, 1, 1)
        #去掉denselayer会提升速度
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        # self.res_main = _OSA_module(in_C,in_C,in_C,4,True)

        self.fuse_main = BasicConv_PRelu(in_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(torch.cat([rgb,depth],dim=1))
        # feat =Channel_shuffle(feat,4) # 不知道这个通道洗牌有用吗，但是速度没怎么降低
        return self.fuse_main(self.res_main(feat)+feat) #self.res_main(feat)+
        # return self.fuse_main(feat) #self.res_main(feat)+


class SDFM(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, kernel_size=3, down_factor=4):
        """
        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            kernel_size (int): 指定的生成的卷积核的大小
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        #(32, 32, 32, 3, 4)
        super(SDFM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C # 这里没有缩减通道 =8
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC3x3_1(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.fuse = BasicConv_PRelu(2 * self.mid_c, out_C, 3, 1, 1)

    def forward(self, x, y):
        x = self.down_input(x) # channel 32 to 8
        result_1 = self.branch_1(x, y)
        # result_3 = self.branch_3(x, y)
        # result_5 = self.branch_5(x, y)
        # return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1))
        return self.fuse(torch.cat((x, result_1), dim=1))


class DepthDC3x3_1(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_1，利用nn.Unfold实现的动态卷积模块
        这里的x应该是被卷的，y是核
        Args:
            in_xC (int): 第一个输入的通道数 rgb
            in_yC (int): 第二个输入的通道数 kernel
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数

            这个版本改为是通道的和空间的平行，并且采用分组的方式，最后cat在一起然后洗牌
        """
        super(DepthDC3x3_1, self).__init__()

        self.kernel_size = 3
        mid_in_yC = in_yC//2
        mid_in_xC = in_xC//2
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel_spatial = nn.Sequential(
            nn.Conv2d(mid_in_yC, mid_in_yC, 3, 1, 1),
            # DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(mid_in_yC, self.kernel_size ** 2, 1),# in_xC
            #N C W H -> N k2 W H
        )
        self.gernerate_kernel_channel = nn.Sequential(
            # nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            # DenseLayer(in_yC, in_yC, k=down_factor),
            nn.AdaptiveAvgPool2d(self.kernel_size),
            nn.Conv2d(mid_in_yC, mid_in_xC, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.padding = 1
        self.dilation = 1
        self.stride = 1
        self.dynamic_bias = None

        # channel attention part
        self.avg_pool = nn.AdaptiveAvgPool2d((self.kernel_size, self.kernel_size))
        self.num_lat = int((self.kernel_size * self.kernel_size) / 2 + 1)
        self.ce = nn.Linear(self.kernel_size * self.kernel_size, self.num_lat, False)
        self.gd = nn.Linear(self.num_lat, self.kernel_size * self.kernel_size, False)
        self.ce_bn = nn.BatchNorm1d(mid_in_xC)
        # 激活层
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        # spatial attention part
        self.conv_sp_1 = nn.Conv2d(mid_in_xC,1,kernel_size=3,padding=1,bias=False)
        self.conv_sp = nn.Conv2d(2,1,kernel_size=3,padding=1,bias=False)
        self.unfold_sp = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.sig_sp = nn.Sigmoid()
    def forward(self, x, y):  # x : rgb y :kernel
        N, xC, xH, xW = x.size()

        # split
        x1,x2 =Split(x)
        y1,y2 = Split(y)
        # channel filter
        # --------------channel attention-------------------
        # 这里是用混合特征生成核
        N, yC, yH, yW = y1.size()
        gl = self.avg_pool(x1).view(N,yC, -1)  # N C k^2
        # # 实际实现过程就是一个se一样的
        out = self.ce(gl)  # N C numlat
        out = self.ce_bn(out)  # bn
        out = self.act(out)  # act
        out = self.gd(out)
        out = self.sig(out.view(-1,1,self.kernel_size,self.kernel_size))
        #
        kernel_channel = self.gernerate_kernel_channel(y1)
        #
        kernel_channel = kernel_channel.reshape(-1, 1, self.kernel_size, self.kernel_size)
        #kernel * filter
        kernel_channel_after = kernel_channel*out
        # -----------------------------------------------------
        # 1 NC k k
        x_input = x1.view(1, -1, x1.size()[2], x1.size()[3])
        channel_after = F.conv2d(x_input, weight=kernel_channel_after, bias=self.dynamic_bias, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=N * xC//2)
        channel_after = channel_after.reshape(N, -1, xH, xW)

        # spatial filter
        kernel = self.gernerate_kernel_spatial(y2)
        kernel = kernel.reshape([N, self.kernel_size ** 2, xH, xW, 1])
        # spatial attention
        # 这个是CBAM里的空间att
        kernel_sp = self.conv_sp_1(x2) # N 1 H W
        kernel_sp =self.unfold_sp(kernel_sp).reshape([N,-1,xH,xW,1]) # N k2 H W 1
        avg_out = torch.mean(kernel_sp, dim=1, keepdim=True) # N 1 H W 1
        max_out,_ = torch.max(kernel_sp, dim=1, keepdim=True) # N 1 H W 1
        x = torch.cat([avg_out, max_out], dim=1) # N 2 H W 1
        x = x.squeeze(4) # N 1 H W 1
        x = self.conv_sp(x)  # N 1 W H
        sp_x = self.sig_sp(x)  # N 1 W H 这里可以当作是一个att
        sp_x = sp_x.unsqueeze(4)
        sp_x =sp_x.permute(0,2,3,4,1).contiguous()  # N W H 1 C
        kernel = kernel.permute(0,2,3,4,1).contiguous()  # N W H 1 C

        kernel_after = kernel* sp_x

        kernel_after = kernel_after.permute(0,4,1,2,3) # N C W H 1
        # 这里就应该是kernel的部分
        kernel_after = kernel_after.permute(0, 2, 3, 1, 4).contiguous()  # N H W k2 1
        unfold_x = self.unfold(x2).reshape([N, xH, xW, xC//2, -1])  # N H W C k2
        # 这里就是两个矩阵的低维度相乘
        spatial_after = torch.matmul(unfold_x,kernel_after) #N H W k2 1×N H W C k2 = N H W C 1
        spatial_after = spatial_after.squeeze(4).permute(0,3,1,2)

        result = torch.cat([channel_after,spatial_after],dim=1)


        return self.fuse(result)