import torch
import torch.nn as nn
from models.BaseBlocks import BasicConv_PRelu,BasicConv2d
import torch.nn.functional as F


class decoder_plus(nn.Module):
    def __init__(self,code=512):
        super(decoder_plus,self).__init__()
        self.conv_lowfuse = BasicConv_PRelu(96,64,1,1,bias=False)
        self.conv_highcode = BasicConv_PRelu(96,code,1,1,bias=False)
        self.conv_highfuse = BasicConv_PRelu(96,64,1,1,bias=False)
        self.conv_transfor = nn.Conv2d(64,code,1,1,padding=0,bias=False)
        self.conv_fuse = BasicConv_PRelu(96,64,1,1,bias=False)
        self.softmax_weight = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, fea_down,fea_up): # up low \ down high
        # high
        n,c,hh,wh = fea_down.shape
        n,c,hl,wl = fea_up.shape
        fea_down_code = self.softmax_weight(self.conv_highcode(fea_down)) #n 512 h w

        fea_down_code =fea_down_code.view(n,-1,hh*wh)
        fea_down_code_t = fea_down_code.transpose(1, 2).contiguous()
        # base map
        fea_down_base = self.conv_highfuse(fea_down)  # n c h w
        fea_down_base_change = fea_down_base.view(n,-1,hh*wh)

        # pool
        fea_down_base_pool = self.gap(fea_down_base)
        fea_down_base_pool = F.interpolate(fea_down_base_pool,size=(fea_up.shape[-2],fea_up.shape[-1]),mode="bilinear",align_corners=False)
        codebook = torch.matmul(fea_down_base_change,fea_down_code_t) # n c code

        # low
        fea_up_fuse_base = self.conv_lowfuse(fea_up) # N C H W
        fea_up_fuse_trans = fea_up_fuse_base + fea_down_base_pool
        fea_up_fuse_trans = self.conv_transfor(fea_up_fuse_trans) # N code H W
        fea_up_fuse_trans = fea_up_fuse_trans.view(n,fea_up_fuse_trans.shape[1],-1)# N code H*W


        # multiply
        fea_new = torch.matmul(codebook,fea_up_fuse_trans) # N C H*W
        fea_new = fea_new.view(n,-1,hl,wl)

        # fuse
        final = self.conv_fuse(torch.cat((fea_new,fea_up_fuse_base),dim=1))
        return final
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

class DFM(nn.Module):
    def __init__(self):
        super(DFM,self).__init__()
        #计算改的通道数量
        # large = 32
        # middle = 16
        # small = 8
        large = 128
        middle = 64
        small = 32
        self.d_ls = int(large/small)  # 4 这个比例是扩张比例
        self.d_ms = int(middle/small) # 2
        self.kup = 3 # 这个是核的大小

        # 用于最后的特征融合
        # 这个融合的部分用3x3的还是1x1的呢
        self.conv_fuse = BasicConv_PRelu(192, 32, 1, 1, bias=False)
        # 用于改通道生成核
        # high
        self.conv_chchaneg_high = BasicConv_PRelu(32, self.kup ** 2, 1, 1) # 这里high的不用d的扩张比例
        # mid
        self.conv_chchaneg_mid = BasicConv_PRelu(32,self.d_ms**2*self.kup**2,1,1) # 这里该用3x3还是1x1
        # low
        self.conv_chchaneg_low = BasicConv_PRelu(32,self.d_ls**2*self.kup**2,1,1)

        #resize channel residual part
        self.conv_low_sp = BasicConv_PRelu(32,self.kup**2,1,1)
        self.conv_mid_sp = BasicConv_PRelu(32,self.kup**2,1,1)

        #-----------通道Dy的部分---------------------
        # 空洞率1
        # low
        self.resize_low = BasicConv_PRelu(32,32*self.d_ls**2,1,1) #变成 N C*d2 H W
        self.gernerate_kernel_channel = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 1),
            # DenseLayer(in_yC, in_yC, k=down_factor),
            nn.AdaptiveAvgPool2d(self.kup),
            #nn.Conv2d(32, 32, 1),
            BasicConv_PRelu(32,32,1),
        )
        self.padding = 1
        self.dilation = 1
        self.stride = 1
        self.dynamic_bias = None
        #mid
        self.resize_mid = BasicConv_PRelu(32,32*self.d_ms**2,1,1)
        self.gernerate_kernel_channel_mid = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 1),
            nn.AdaptiveAvgPool2d(self.kup),
            BasicConv_PRelu(32, 32, 1),
        )
        self.padding_mid = 1
        self.dilation_mid = 1
        self.stride_mid = 1
        self.dynamic_bias_mid = None

        self.gernerate_kernel_channel_high = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 1),
            nn.AdaptiveAvgPool2d(self.kup),
            BasicConv_PRelu(32, 32, 1),
        )
        self.padding_high = 1
        self.dilation_high = 1
        self.stride_high = 1
        self.dynamic_bias_high = None

        # fuse conv
        self.conv_fuse_high = BasicConv_PRelu(96,32,1)
        self.gap = nn.AdaptiveAvgPool2d(1)




    def forward(self,fea_low,fea_mid,fea_high):# up low up的尺寸大\ down high的尺寸小
        # 这里的特征是两个concat在一起的

        # 尺寸小
        nh, ch, hh, wh = fea_high.shape
        #中等尺寸的
        nm,cm,hm,wm = fea_mid.shape
        #尺寸大的
        nl, cl, hl, wl = fea_low.shape

        # resize to high
        fea_low_new = F.interpolate(fea_low,size=(fea_high.shape[-2],fea_high.shape[-1]),mode="bilinear",align_corners=False)
        fea_mid_new = F.interpolate(fea_mid,size=(fea_high.shape[-2],fea_high.shape[-1]),mode="bilinear",align_corners=False)


        #fuse high
        fea_high_fused = self.conv_fuse_high(torch.cat([fea_high,fea_mid_new,fea_low_new],dim=1))

        # pool
        fea_down_pool = self.gap(fea_high_fused)
        fea_down_pool_high = F.interpolate(fea_down_pool,size=(fea_high.shape[-2],fea_high.shape[-1]),mode="bilinear",align_corners=False)
        fea_down_pool_mid = F.interpolate(fea_down_pool,size=(fea_mid.shape[-2],fea_mid.shape[-1]),mode="bilinear",align_corners=False)
        fea_down_pool_low = F.interpolate(fea_down_pool,size=(fea_low.shape[-2],fea_low.shape[-1]),mode="bilinear",align_corners=False)



    #-----生成dy的核的部分--------
        # low
        kernel_tensor_low = self.conv_chchaneg_low(fea_high_fused) # N d^2*k^2 w h
        kernel_tensor_low = F.pixel_shuffle(kernel_tensor_low,self.d_ls) # N d^2*k^2 w h -> N k^2 dh dw = N k^2 H W
        # 这个正规化有没有用再试试
        # 添加特征low和high
        fea_low_d = self.conv_low_sp(fea_low) # 这里相当于去添加原来的特征。
        kernel_tensor_low += fea_low_d
        kernel_tensor_low = F.softmax(kernel_tensor_low,dim=1) # N k2 H W 这个对
        # reshape成k2在最后
        kernel_tensor_low = kernel_tensor_low.permute(0,2,3,1).contiguous() # N H W k^2
        # mid
        kernel_tensor_mid = self.conv_chchaneg_mid(fea_high_fused)
        kernel_tensor_mid = F.pixel_shuffle(kernel_tensor_mid,self.d_ms)
        fea_mid_d =self.conv_mid_sp(fea_mid)
        kernel_tensor_mid +=fea_mid_d
        kernel_tensor_mid = F.softmax(kernel_tensor_mid,dim=1)
        kernel_tensor_mid = kernel_tensor_mid.permute(0,2,3,1).contiguous()

        # high
        kernel_tensor_high = self.conv_chchaneg_high(fea_high_fused) # 这里少一步
        kernel_tensor_high = F.softmax(kernel_tensor_high,dim=1)
        kernel_tensor_high = kernel_tensor_high.permute(0,2,3,1).contiguous()

        #------生成特征的d维度--------
        # N C H+k W+k
        # low
        # New !!这里添加了各种pool
        fea_low_pad = F.pad(fea_low+fea_down_pool_low, pad=(self.kup // 2, self.kup // 2,self.kup // 2, self.kup // 2),mode='constant', value=0)
        fea_low_pad = fea_low_pad.unfold(dimension=2,size=self.kup,step=1) # N C H W+k k
        fea_low_pad = fea_low_pad.unfold(3,self.kup,step=1) # N C H W k k
        fea_low_pad = fea_low_pad.reshape(nl,cl,hl,wl,-1) # N C H W k^2
        fea_low_pad = fea_low_pad.permute(0,2,3,1,4).contiguous() # N H W C k^2
        # mid
        # New !!这里添加了各种pool
        fea_mid_pad = F.pad(fea_mid+fea_down_pool_mid, pad=(self.kup // 2, self.kup // 2, self.kup // 2, self.kup // 2), mode='constant',
                            value=0)
        fea_mid_pad = fea_mid_pad.unfold(dimension=2, size=self.kup, step=1)  # N C H W+k k
        fea_mid_pad = fea_mid_pad.unfold(3, self.kup, step=1)  # N C H W k k
        fea_mid_pad = fea_mid_pad.reshape(nm, cm, hm, wm, -1)  # N C H W k^2
        fea_mid_pad = fea_mid_pad.permute(0, 2, 3, 1, 4).contiguous()  # N H W C k^2
        # high
        # New !!这里添加了各种pool
        fea_high_pad = F.pad(fea_high+fea_down_pool_high, pad=(self.kup // 2, self.kup // 2, self.kup // 2, self.kup // 2), mode='constant',
                            value=0)
        fea_high_pad = fea_high_pad.unfold(dimension=2, size=self.kup, step=1)  # N C H W+k k
        fea_high_pad = fea_high_pad.unfold(3, self.kup, step=1)  # N C H W k k
        fea_high_pad = fea_high_pad.reshape(nh, ch, hh, wh, -1)  # N C H W k^2
        fea_high_pad = fea_high_pad.permute(0, 2, 3, 1, 4).contiguous()  # N H W C k^2

        #-------核的相乘----------
        # low
        # 首先扩张维度，将核的扩张成5维
        kernel_tensor_low = kernel_tensor_low.unsqueeze(4) # N H W k^2 1
        # N H W C k^2 * N H W k^2 1 = N H W C 1
        fea_low_new = torch.matmul(fea_low_pad,kernel_tensor_low)
        # 压缩通道
        fea_low_new = fea_low_new.squeeze(dim=4) # N H W C
        # 改变通道
        fea_low_new = fea_low_new.permute(0,3,1,2) # N C H W
        # mid
        # 首先扩张维度，将核的扩张成5维
        kernel_tensor_mid = kernel_tensor_mid.unsqueeze(4)  # N H W k^2 1
        # N H W C k^2 * N H W k^2 1 = N H W C 1
        fea_mid_new = torch.matmul(fea_mid_pad, kernel_tensor_mid)
        # 压缩通道
        fea_mid_new = fea_mid_new.squeeze(dim=4)  # N H W C
        # 改变通道
        fea_mid_new = fea_mid_new.permute(0, 3, 1, 2)  # N C H W
        # high
        # 首先扩张维度，将核的扩张成5维
        kernel_tensor_high = kernel_tensor_high.unsqueeze(4)  # N H W k^2 1
        # N H W C k^2 * N H W k^2 1 = N H W C 1
        fea_high_new = torch.matmul(fea_high_pad, kernel_tensor_high)
        # 压缩通道
        fea_high_new = fea_high_new.squeeze(dim=4)  # N H W C
        # 改变通道
        fea_high_new = fea_high_new.permute(0, 3, 1, 2)  # N C H W



        #--------通道Dy的部分----------
        #low
        #空洞率1
        # 生成核的部分
        fea_high_lowch = self.resize_low(fea_high_fused) # N C*d2 h w
        fea_high_lowch = F.pixel_shuffle(fea_high_lowch,self.d_ls) # N C h*d w*d
        kernel_tensor_ch_low =self.gernerate_kernel_channel(fea_high_lowch+fea_low).reshape(-1, 1, self.kup, self.kup) #ch 32,# NC 1 k k
        # 改变input的shape
        fea_low_re = fea_low +fea_down_pool_low
        fea_up_change_low = fea_low_re.reshape(1, -1, fea_low.size()[2], fea_low.size()[3])
        #卷积
        channel_after = F.conv2d(fea_up_change_low, weight=kernel_tensor_ch_low, bias=self.dynamic_bias, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation, groups=nl * cl)
        channel_after =channel_after.reshape(nl, -1, hl, wl)

        # mid
        # 空洞率1
        # 生成核的部分
        fea_high_midch = self.resize_mid(fea_high_fused) # N C*d2 h w
        fea_high_midch = F.pixel_shuffle(fea_high_midch,self.d_ms)
        kernel_tensor_ch_mid = self.gernerate_kernel_channel_mid(fea_high_midch+fea_mid).reshape(-1, 1, self.kup,
                                                                               self.kup)  # ch 32,# NC 1 k k
        # 改变input的shape
        fea_mid_re = fea_mid +fea_down_pool_mid
        fea_up_change_mid = fea_mid_re.reshape(1, -1, fea_mid.size()[2], fea_mid.size()[3])
        # 卷积
        channel_after_mid = F.conv2d(fea_up_change_mid, weight=kernel_tensor_ch_mid, bias=self.dynamic_bias_mid,
                                 stride=self.stride_mid,
                                 padding=self.padding_mid, dilation=self.dilation_mid, groups=nm * cm)
        channel_after_mid = channel_after_mid.reshape(nm, -1, hm, wm)

        # high
        # 空洞率1
        # 生成核的部分
        kernel_tensor_ch_high = self.gernerate_kernel_channel_high(fea_high_fused).reshape(-1, 1, self.kup,
                                                                               self.kup)  # ch 32,# NC 1 k k
        # 改变input的shape
        fea_high_re = fea_high +fea_down_pool_high
        fea_up_change_high = fea_high_re.reshape(1, -1, fea_high.size()[2], fea_high.size()[3])
        # 卷积
        channel_after_high = F.conv2d(fea_up_change_high, weight=kernel_tensor_ch_high, bias=self.dynamic_bias_high,
                                     stride=self.stride_high,
                                     padding=self.padding_high, dilation=self.dilation_high, groups=nh * ch)
        channel_after_high = channel_after_high.reshape(nh, -1, hh, wh)

        # 特征融合
        # 特征合并 concat在一起fea_up_af,

        # 融合还是要上采样
        fea_high_new = F.interpolate(fea_high_new,
                                                size=(fea_low_new.shape[-2], fea_low_new.shape[-1]),
                                                mode="bilinear", align_corners=False)
        fea_mid_new = F.interpolate(fea_mid_new,
                                                size=(fea_low_new.shape[-2], fea_low_new.shape[-1]),
                                                mode="bilinear", align_corners=False)

        # ch dy 上采样
        channel_after_mid = F.interpolate(channel_after_mid,
                                                size=(fea_low_new.shape[-2], fea_low_new.shape[-1]),
                                                mode="bilinear", align_corners=False)
        channel_after_high = F.interpolate(channel_after_high,
                                          size=(fea_low_new.shape[-2], fea_low_new.shape[-1]),
                                          mode="bilinear", align_corners=False)

        fea_final = torch.cat([fea_high_new,channel_after_high,fea_mid_new,channel_after_mid,fea_low_new,channel_after], dim=1)

        fea_final = self.conv_fuse(fea_final)

        # 最后返回特征
        return fea_final