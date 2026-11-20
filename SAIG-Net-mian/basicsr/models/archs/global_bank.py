import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device
            )
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing='ij'
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def split_heads(x, heads):
    b, n, c = x.shape
    return x.view(b, n, heads, c // heads).transpose(1, 2)

def merge_heads(x):
    b, heads, n, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, heads * d)

def LinearPositionEmbeddingSine(x, dim=128, NORMALIZE_FACOR=1/200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim//4-1, dim//4).to(x.device)
    return torch.cat([torch.sin(3.14*x[..., -2:-1]*freq_bands*NORMALIZE_FACOR), torch.cos(3.14*x[..., -2:-1]*freq_bands*NORMALIZE_FACOR), torch.sin(3.14*x[..., -1:]*freq_bands*NORMALIZE_FACOR), torch.cos(3.14*x[..., -1:]*freq_bands*NORMALIZE_FACOR)], dim=-1)

def ExpPositionEmbeddingSine(x, dim=128, NORMALIZE_FACOR=1/200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim//4-1, dim//4).to(x.device)
    return torch.cat([torch.sin(x[..., -2:-1]*(NORMALIZE_FACOR * 2 ** freq_bands)), torch.cos(x[..., -2:-1]*(NORMALIZE_FACOR * 2 ** freq_bands)), torch.sin(x[..., -1:]*(NORMALIZE_FACOR * 2 ** freq_bands)), torch.cos(x[..., -1:]*(NORMALIZE_FACOR * 2 ** freq_bands))], dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, embed_dim=64, pe='linear'):
        super().__init__()
        self.patch_size = patch_size
        self.dim = embed_dim
        self.pe = pe

        # assert patch_size == 8
        if patch_size == 8:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//2, embed_dim, kernel_size=6, stride=2, padding=2),
            )
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim, kernel_size=6, stride=2, padding=2),
            )
        else:
            print(f"patch size = {patch_size} is unacceptable.")

        self.ffn_with_coord = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1)
        )
        self.norm = nn.LayerNorm(embed_dim*2)

    def forward(self, x):
        B, C, H, W = x.shape    # C == 1

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        x = self.proj(x)
        out_size = x.shape[2:] 

        patch_coord = coords_grid(B, out_size[0], out_size[1], x.device) * self.patch_size + self.patch_size/2 # in feature coordinate space
        patch_coord = patch_coord.view(B, 2, -1).permute(0, 2, 1)
        if self.pe == 'linear':
            patch_coord_enc = LinearPositionEmbeddingSine(patch_coord, dim=self.dim)
        elif self.pe == 'exp':
            patch_coord_enc = ExpPositionEmbeddingSine(patch_coord, dim=self.dim)
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(B, -1, out_size[0], out_size[1])

        x_pe = torch.cat([x, patch_coord_enc], dim=1)
        x =  self.ffn_with_coord(x_pe)
        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, out_size

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = split_heads(Q, self.heads)
        K = split_heads(K, self.heads)

        dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)

        V = split_heads(V, self.heads)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = merge_heads(out)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(SelfAttentionLayer, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = split_heads(Q, self.heads)
        K = split_heads(K, self.heads)

        dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale
        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)

        V = split_heads(V, self.heads)
        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = merge_heads(out)

        return out

class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)

        x = self.multi_head_attn(q, k, v)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

class GlobalSceneBank(nn.Module):
    def __init__(
        self,
        in_dim3: int,
        in_dim4: int,
        patch_size: int = 4,
        patch_embed_dim: int = 64,   # PatchEmbed 内部 embed_dim
        bank_token_num: int = 8,     # Nb
        bank_dim: int = 128,         # Cb
        num_heads: int = 8,
        refine_depth: int = 1,
        pe: str = "linear",
        dropout: float = 0.0,
    ):
        super().__init__()

        assert patch_size in [4, 8], "当前 PatchEmbed 代码只支持 patch_size=4 或 8"

        token_dim = patch_embed_dim * 2

        self.patch_embed3 = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_dim3,
            embed_dim=patch_embed_dim,
            pe=pe
        )
        self.patch_embed4 = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_dim4,
            embed_dim=patch_embed_dim,
            pe=pe
        )

        # 用于区分来自哪一个尺度
        self.level_embed3 = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.level_embed4 = nn.Parameter(torch.zeros(1, 1, token_dim))

        # learnable latent tokens = bank 槽位
        self.latent_tokens = nn.Parameter(
            torch.randn(1, bank_token_num, bank_dim)
        )

        # 写库：latent query 去读 multi-scale tokens
        self.write_attn = CrossAttentionLayer(
            qk_dim=bank_dim,
            v_dim=bank_dim,
            query_token_dim=bank_dim,
            tgt_token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 写完之后再 refine 一下 bank
        self.refine_blocks = nn.ModuleList([
            SelfAttentionLayer(
                dim=bank_dim,
                cfg=None,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(refine_depth)
        ])

        self.norm = nn.LayerNorm(bank_dim)

    def forward(self, feat3, feat4):
        # t3, t4: [B, N, token_dim]
        t3, _ = self.patch_embed3(feat3)
        t4, _ = self.patch_embed4(feat4)

        t3 = t3 + self.level_embed3
        t4 = t4 + self.level_embed4

        tokens = torch.cat([t3, t4], dim=1)  # [B, N3+N4, token_dim]

        # expand latent tokens 到 batch 维
        latent = self.latent_tokens.expand(tokens.size(0), -1, -1)  # [B, Nb, Cb]

        # 写库
        bank = self.write_attn(latent, tokens)  # [B, Nb, Cb]

        # refine bank
        for blk in self.refine_blocks:
            bank = blk(bank)

        bank = self.norm(bank)
        return bank

class BankQueryBlock(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        bank_dim: int = 128,
        num_heads: int = 8,
        query_downsample: int = 1,   # D2 可以设成 2，减轻计算
        dropout: float = 0.0,
    ):
        super().__init__()

        self.query_downsample = query_downsample

        self.q_proj = nn.Conv2d(feat_dim, bank_dim, kernel_size=1)

        self.read_attn = CrossAttentionLayer(
            qk_dim=bank_dim,
            v_dim=bank_dim,
            query_token_dim=bank_dim,
            tgt_token_dim=bank_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim + bank_dim, feat_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, dec_feat, bank):
        identity = dec_feat

        if self.query_downsample > 1:
            q_map = F.avg_pool2d(
                dec_feat,
                kernel_size=self.query_downsample,
                stride=self.query_downsample
            )
        else:
            q_map = dec_feat

        B, _, Hq, Wq = q_map.shape

        # query tokens: [B, Hq*Wq, Cb]
        q = self.q_proj(q_map).flatten(2).transpose(1, 2)

        # 从 bank 中读
        g = self.read_attn(q, bank)  # [B, Hq*Wq, Cb]
        g = g.transpose(1, 2).reshape(B, -1, Hq, Wq)

        if self.query_downsample > 1:
            g = F.interpolate(
                g,
                size=identity.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        fused = self.fuse(torch.cat([identity, g], dim=1))
        out = identity + self.gamma * fused
        return out
