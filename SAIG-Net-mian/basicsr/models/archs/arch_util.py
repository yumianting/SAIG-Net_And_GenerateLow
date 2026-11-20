import torch
from torch import nn as nn
from torch.nn import init as init

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        
        # 优化：使用 rsqrt()
        rsigma = (var + eps).rsqrt() 
        y = (x - mu) * rsigma
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(y, var, weight)
        
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        
        # 修复：使用 saved_tensors
        y, var, weight = ctx.saved_tensors
        
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        
        # 优化：使用 rsqrt()
        gx = (var + eps).rsqrt() * (g - y * mean_gy - mean_g)
        
        grad_weight = (grad_output * y).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.channels = channels
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class CustomSequential(nn.Module):  
    
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, *args, use_adapter=False, **kwargs):
        for module in self.modules_list:
            if hasattr(module, 'set_use_adapters'):
                module.set_use_adapters(use_adapter)
            x = module(x, *args, **kwargs)
        return x
