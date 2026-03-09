import torch
import numpy as np
from torch import nn as nn
from torch.nn import init as init
import torch.distributed as dist
from collections import OrderedDict

from .fema_utils import NormLayer, ActLayer

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class SPAMBlock(nn.Module):
    def __init__(self, in_channel, out_channal, channel_expand=2):
        super().__init__()

        ### spatial attention module.
        ### Reference: https://github.com/cidautai/DarkIR/blob/main/archs/arch_model.py
        self.dw_channel = in_channel * channel_expand
        self.norm1 = nn.InstanceNorm2d(in_channel)
        # self.norm1 = LayerNorm2d(in_channel)


        self.conv1_1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, groups=in_channel, bias=True, dilation=1) 
        self.conv1_2 = nn.Conv2d(in_channel, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.conv1_3 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True, dilation=1)

        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )

        self.conv2 = nn.Conv2d(self.dw_channel//2, in_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        ### Gated Feed Forward Network.
        self.ffn_channel = in_channel * channel_expand
        self.norm2 = nn.InstanceNorm2d(in_channel)
        # self.norm2 = LayerNorm2d(in_channel)

        self.conv3 = nn.Conv2d(in_channel, self.ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(self.ffn_channel//2, out_channal, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, out_channal, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)

    def forward(self, x):
        ### Spatial Attention Module.
        x1 = self.norm1(x)
        x1 = self.conv1_3(self.conv1_2(self.conv1_1(x1)))

        x2 = self.sg1(x1)
        x3 = self.sca(x2) * x2
        x3 = self.conv2(x3)
        x3 = x + self.beta * x3

        ### Gated Feed Forward Network.
        x4 = self.conv3(self.norm2(x3))
        x5 = self.sg2(x4)
        x6 = self.conv4(x5)
        out = x3 + self.gamma * x6

        return out
    

class DCAMBlock(nn.Module):
    def __init__(self, in_channel, out_channal, channel_expand=2):
        super().__init__()

        ### spatial attention module.
        ### Reference: https://github.com/cidautai/DarkIR/blob/main/archs/arch_model.py
        self.dw_channel = in_channel * channel_expand
        self.norm1 = LayerNorm2d(in_channel)


        self.conv1_1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, groups=in_channel, bias=True, dilation=1) 
        self.conv1_2 = nn.Conv2d(in_channel, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # self.conv1_3 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True, dilation=1)

        dilation = [1,2,4]
        self.dconv1 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=dilation[0], stride=1, groups=self.dw_channel, 
                                    bias=True, dilation=dilation[0])
        self.dconv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=dilation[1], stride=1, groups=self.dw_channel, 
                                    bias=True, dilation=dilation[1])
        self.dconv3 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=dilation[2], stride=1, groups=self.dw_channel, 
                                    bias=True, dilation=dilation[2])

        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )

        self.conv2 = nn.Conv2d(self.dw_channel//2, in_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)

        ### Gated Feed Forward Network.
        self.ffn_channel = in_channel * channel_expand
        self.norm2 = LayerNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, self.ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(self.ffn_channel//2, out_channal, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, out_channal, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)

    def forward(self, x):
        ### Spatial Attention Module.
        x1 = self.norm1(x)
        # x1 = self.conv1_3(self.conv1_2(self.conv1_1(x1)))
        x1 = self.conv1_2(self.conv1_1(x1))
        x1 = self.dconv1(x1) + self.dconv2(x1) + self.dconv3(x1)

        x2 = self.sg1(x1)
        x3 = self.sca(x2) * x2
        x3 = self.conv2(x3)
        x3 = x + self.beta * x3

        ### Gated Feed Forward Network.
        x4 = self.conv3(self.norm2(x3))
        x5 = self.sg2(x4)
        x6 = self.conv4(x5)
        out = x3 + self.gamma * x6

        return out


class ChannelAttentionLayer(nn.Module):
    """ https://github.com/shiningZZ/GUPDM/blob/main/models/GUPDM.py """
    def __init__(self, channel, reduction=8, bias=True):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avg_pool(x)
        avg_y = self.conv_du(avg_y)
        max_y = self.max_pool(x)
        max_y = self.conv_du(max_y)
        y = self.sigmoid(avg_y + max_y)

        return x * y

class PixelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(PixelAttentionLayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
    
class AttnResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(AttnResBlock, self).__init__()

        self.residual = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            ChannelAttentionLayer(out_channel, reduction=16),
            PixelAttentionLayer(out_channel, reduction=16)
        )
    def forward(self, x):
        return self.residual(x) + x