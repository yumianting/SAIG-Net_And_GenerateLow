import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_util import LayerNorm2d


def _reshape_to_heads(x, num_heads):
    b, c, h, w = x.shape
    if c % num_heads != 0:
        raise ValueError(f'channels={c} must be divisible by num_heads={num_heads}.')
    return x.reshape(b, num_heads, c // num_heads, h * w)

def _reshape_from_heads(x, h, w):
    b, num_heads, head_dim, hw = x.shape
    if hw != h * w:
        raise ValueError(f'Expected hw={h * w}, but got {hw}.')
    return x.reshape(b, num_heads * head_dim, h, w)

class SkipFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
        )
    def forward(self, x, skip):
        out = self.fuse(torch.cat([x, skip], dim=1))
        return x + out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
#--------------  编码器  ------------

class MSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head = dim // heads if dim_head is None else dim_head
        
        # 1. 联合生成 QKV 能够提高计算效率
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        
        # 2. Depthwise 卷积增强局部特征，增强平移不变性
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.k_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.v_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        # 温度系数，用于调节 softmax 后的分布
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        
        self.illu_k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.illu_v_proj = nn.Conv2d(dim, 1, 1, bias=False) 
        # 3. 补全：特征聚合的输出投影层
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, illu_map, illu_scale):
        # x, illu_map: [B, C, H, W]
        B, C, H, W = x.shape
    
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1) # 切分后每个 shape 为 [B, C, H, W]
        
        q = self.q_dw(q)
        k = self.k_dw(k)
        v = self.v_dw(v)
        
        illu_k = torch.sigmoid(self.illu_k_proj(illu_map))
        k = k * (1 + illu_k)
        
        gate = torch.sigmoid(self.illu_v_proj(illu_scale))
        v = v * (2.0 - gate)
        # -----------------------------------------------------------
        
        q = _reshape_to_heads(q, self.heads)
        k = _reshape_to_heads(k, self.heads)
        v = _reshape_to_heads(v, self.heads)
        
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        
        attn = (q @ k.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        
        out = _reshape_from_heads(out, H, W)
        
        out = self.proj(out)
        
        return out


class ENFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 2
        self.project_in = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.conv(x)
        x = self.sg(x)
        x = x * self.sca(x)  
        x = self.project_out(x)
        return x 


class EnBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=None,
            heads=8,
    ):
        super().__init__()
        self.laynorm2d1 = LayerNorm2d(dim)
        self.laynorm2d2 = LayerNorm2d(dim)
        self.msa = MSA(dim=dim, dim_head=dim_head, heads=heads)
        self.ffn = ENFFN(dim)
        
    def forward(self, input, illu_map, illu_scale):
        
        x = self.laynorm2d1(input)
        x = self.msa(x, illu_map, illu_scale)
        mid = input + x

        x1 = self.laynorm2d2(mid)
        out = mid + self.ffn(x1)
        return out

# ------------- 解码器 --------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """
    沿 channel 维度做 LayerNorm。
    小通道时 eps 适当调大（1e-4）以稳定方差估计。
    """
    def __init__(self, num_channels: int, eps: float = 1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x    = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class IlluminationAwareCoordAtt(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        prior_channels: int = -1,   # -1 表示与 channels 相同
    ):
        super().__init__()

        prior_channels = channels if prior_channels < 1 else prior_channels
        mip = min(channels, max(8, channels // reduction))

        # ---- 局部方向平滑 ----
        self.local_smooth_h = nn.AvgPool2d(kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.local_smooth_w = nn.AvgPool2d(kernel_size=(1, 7), stride=1, padding=(0, 3))

        # ---- 主干坐标注意力 ----
        self.conv1  = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.norm   = nn.GroupNorm(1, mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

        # ---- 全局/局部融合系数（sigmoid 约束，[-∞,+∞] → [0,1]）----
        self.gamma_h = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5 初始等权
        self.gamma_w = nn.Parameter(torch.zeros(1))

        # ---- 光照感知分支 ----
        # illu_avg 的全局均值池化结果后面会复用，不单独放在 nn.Sequential 里
        self.illum_max = nn.AdaptiveMaxPool2d(1)
        self.illum_fc  = nn.Sequential(
            nn.Conv2d(channels * 2, mip, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ---- 双路融合权重（softmax 归一化，和为 1）----
        self.route_w = nn.Parameter(torch.zeros(2))  # softmax → [w_coord, w_illum]

        # ---- prior 分支 ----
        self.prior_proj = nn.Conv2d(prior_channels, channels, kernel_size=1, bias=False)
        self.prior_norm = LayerNorm2d(channels)       # 消除尺度差异
        self.prior_dw   = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.prior_scale_h = nn.Parameter(torch.zeros(1))  # tanh 约束
        self.prior_scale_w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # ------ 1. 方向特征：全局 + 局部融合 ------
        # 全局
        x_h_global = x.mean(dim=3, keepdim=True)                          # [N,C,H,1]
        x_w_global = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)     # [N,C,W,1]

        # 局部：先做 7×1 / 1×7 局部平滑，再沿互补方向全局压缩
        x_h_local = self.local_smooth_h(x).mean(dim=3, keepdim=True)      # [N,C,H,1]
        x_w_local = self.local_smooth_w(x).mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        gamma_h = torch.sigmoid(self.gamma_h)
        gamma_w = torch.sigmoid(self.gamma_w)
        x_h = x_h_global + gamma_h * x_h_local
        x_w = x_w_global + gamma_w * x_w_local

        # ------ 2. prior 注入（带归一化）------
        prior = self.prior_norm(self.prior_proj(prior))   # 消除尺度差异
        prior = self.prior_dw(prior)

        p_h = prior.mean(dim=3, keepdim=True)
        p_w = prior.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        s_h = torch.tanh(self.prior_scale_h)
        s_w = torch.tanh(self.prior_scale_w)
        x_h = x_h + s_h * p_h
        x_w = x_w + s_w * p_w

        # ------ 3. 坐标注意力 ------
        y = torch.cat([x_h, x_w], dim=2)         # [N,C,H+W,1]
        y = self.act(self.norm(self.conv1(y)))

        x_h_att, x_w_att = torch.split(y, [h, w], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)

        a_h  = torch.sigmoid(self.conv_h(x_h_att))   # [N,C,H,1]
        a_w  = torch.sigmoid(self.conv_w(x_w_att))   # [N,C,1,W]
        gate = a_h * a_w                              # [N,C,H,W]

        coord_delta = identity * gate

        # ------ 4. 光照感知分支 ------
        # 全局均值池化复用（避免重复计算）
        illum_avg_feat = x.mean(dim=[2, 3], keepdim=True)          # [N,C,1,1]
        illum_max_feat = self.illum_max(x)                         # [N,C,1,1]
        illum_scale = self.illum_fc(
            torch.cat([illum_avg_feat, illum_max_feat], dim=1)
        )                                                          # [N,C,1,1]
        illum_delta = identity * illum_scale

        # ------ 5. 双路 softmax 融合 ------
        w = torch.softmax(self.route_w, dim=0)
        delta = w[0] * coord_delta + w[1] * illum_delta
        
        return delta


class DEFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = dim * 2

        self.project_in  = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.dwconv      = nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=1, padding=1,
            groups=hidden_dim, bias=False,
        )
        self.conv        = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.sg          = SimpleGate()

        # SimpleGate 后通道减半 → dim
        self.sca         = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)    # [B, 2dim, H, W]
        x = self.dwconv(x)
        x = self.conv(x)
        x = self.sg(x)            # [B, dim, H, W]
        x = x * self.sca(x)
        x = self.project_out(x)
        return x


class DeBlock(nn.Module):
    def __init__(self, dim: int, prior_channels: int = -1):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = IlluminationAwareCoordAtt(
            dim, reduction=16, prior_channels=prior_channels
        )
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = DEFFN(dim)

        self.attn_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.ffn_scale  = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(
        self, x: torch.Tensor, illu_map: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn_scale * self.attn(self.norm1(x), illu_map)
        x = x + self.ffn_scale  * self.ffn(self.norm2(x))
        return x