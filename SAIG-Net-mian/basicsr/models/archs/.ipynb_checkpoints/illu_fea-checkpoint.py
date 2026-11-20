import math
import torch
from torch import nn
import torch.nn.functional as F

class ChannelAwareModulation(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels, 4)  # 小通道数保底

        # 全局 + 跨通道：一个 SE 搞定
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B,C,1,1)
            nn.Conv2d(in_channels, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid(),
        )

        # 局部空间：depthwise 3×3（极轻量，无跨通道参数）
        self.local_dw = nn.Conv2d(
            in_channels, in_channels, 3,
            padding=1, groups=in_channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.global_gate(x)        # 全局通道权重
        local = torch.sigmoid(self.local_dw(x))  # 局部空间权重
        return x * gate * local           # 两级门控，语义统一

class SharedMultiScaleBackbone(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int):
        super().__init__()
        assert feat_dim % 2 == 0  # 只需两路，放宽限制
        bd = feat_dim // 2

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 3, padding=1),
            nn.GELU(),
        )

        self.rf_d1 = nn.Conv2d(feat_dim, bd, 3, padding=1, dilation=1)
        self.rf_d4 = nn.Conv2d(feat_dim, bd, 3, padding=4, dilation=4)

        self.glb_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, max(feat_dim // 2, 4), 1),
            nn.GELU(),
            nn.Conv2d(max(feat_dim // 2, 4), feat_dim, 1),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),  # 空间混合
            nn.Conv2d(feat_dim, feat_dim, 1),                              # 通道混合
            nn.GELU(),
        )

        self.register_buffer('log11', torch.log(torch.tensor(11.0)))
        self.brightness_gate = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, 1),
            nn.Sigmoid(),
        )

        self.modulation_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, in_channels, 1),
        )
        self.channel_modulation = ChannelAwareModulation(in_channels)

    def forward(self, x: torch.Tensor):
        f = self.stage1(x)

        # 两路多尺度 + 全局 SE 门控
        f_ms = torch.cat([self.rf_d1(f), self.rf_d4(f)], dim=1)  # (B, feat_dim, H, W)
        f_ms = f_ms * self.glb_se(f)   # 全局上下文以门控形式注入，无梯度累加问题

        feat = self.fusion(f_ms)

        # Log 域亮度门控
        x_log = torch.log1p(x.clamp(min=0) * 10) / self.log11
        feat = feat * self.brightness_gate(x_log)

        modulation = self.channel_modulation(self.modulation_proj(feat))
        return feat, modulation

class RotatedRegionSampler(nn.Module):
    def __init__(self, in_channels, feat_dim, n_pts=9):
        super().__init__()
        s = int(n_pts ** 0.5)
        self.n_pts = n_pts
        self.in_channels = in_channels

        self.affine_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 1),   # ← 插入逐点卷积，打通通道
            nn.GELU(),
            nn.Conv2d(feat_dim, 5, 1),
        )

        self.weight_head = nn.Conv2d(feat_dim, n_pts * in_channels, 1)
      
        # 固定基础网格
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, s),
            torch.linspace(-1, 1, s), indexing='ij')
        self.register_buffer('base_pts',
            torch.stack([gx.flatten(), gy.flatten()], dim=1))

        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def forward(self, x, feat):
        B, C, H, W = x.shape
        N = self.n_pts

        # 仿射参数（通道间已交流）
        params = self.affine_head(feat)
        sx    = torch.sigmoid(params[:, 0]) * 0.5 + 0.01
        sy    = torch.sigmoid(params[:, 1]) * 0.5 + 0.01
        theta = params[:, 2] * math.pi
        dcx   = params[:, 3].tanh() * 0.3
        dcy   = params[:, 4].tanh() * 0.3

        cos_t = theta.cos().unsqueeze(-1)
        sin_t = theta.sin().unsqueeze(-1)
        sx = sx.unsqueeze(-1)
        sy = sy.unsqueeze(-1)

        bx = self.base_pts[:, 0].view(1, 1, 1, N)
        by = self.base_pts[:, 1].view(1, 1, 1, N)

        delta_x = sx * cos_t * bx - sy * sin_t * by
        delta_y = sx * sin_t * bx + sy * cos_t * by

        c_y = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
        c_x = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
        grid_y, grid_x = torch.meshgrid(c_y, c_x, indexing='ij')

        pts_x = grid_x.unsqueeze(0).unsqueeze(-1) + dcx.unsqueeze(-1) + delta_x
        pts_y = grid_y.unsqueeze(0).unsqueeze(-1) + dcy.unsqueeze(-1) + delta_y

        sample_grid = torch.stack([pts_x, pts_y], dim=-1)
        sample_grid = sample_grid.view(B, H, W * N, 2)

        sampled = F.grid_sample(
            x, sample_grid,
            align_corners=False, padding_mode='border', mode='bilinear')
        sampled = sampled.view(B, C, H, W, N)

        weights = self.weight_head(feat)         
        weights = weights.view(B, C, N, H, W)       # (B, C, N, H, W)
        weights = torch.softmax(weights, dim=2)     # 对 N 维度做 softmax
        weights = weights.permute(0, 1, 3, 4, 2)   # (B, C, H, W, N)

        return (sampled * weights).sum(dim=-1)      # (B, C, H, W)                   

class SAI2E(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 train_patch: int | list = 256,
                 eps: float = 1e-3,
                 feat_dim: int = None,
                 n_pts: int = 9):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if feat_dim is None:
            feat_dim = max(in_channels * 1, 16)
        feat_dim = ((feat_dim + 1) // 2) * 2 
        self.feat_dim = feat_dim

        if not isinstance(train_patch, list):
            self.train_patch = [train_patch, train_patch]
        else:
            self.train_patch = train_patch
            
        self.backbone = SharedMultiScaleBackbone(in_channels, feat_dim)

        self.rotated_sampler = RotatedRegionSampler(in_channels, feat_dim, n_pts)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, scale = self.backbone(x)

        # scale 保护（保留，但承认只防数值稳定）
        if self.eps != 0:
            safe_sign = torch.where(scale >= 0,
                                    torch.ones_like(scale),
                                    torch.full_like(scale, -1.0))
            scale = torch.where(scale.abs() < self.eps,
                                safe_sign * self.eps, scale)

        raw_avg = self.rotated_sampler(x, feat)
        
        res = raw_avg * scale
        
        return res , scale
