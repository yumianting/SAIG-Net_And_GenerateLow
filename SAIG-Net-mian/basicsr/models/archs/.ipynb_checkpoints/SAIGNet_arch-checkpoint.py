import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_model import EnBlock, DeBlock, SkipFuse
from .arch_util import CustomSequential, Downsample ,Upsample
from .global_bank import BankQueryBlock, GlobalSceneBank
from .illu_fea import SAI2E

class SAIGNet(nn.Module):
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=1,
                 middle_blk_num_dec=1, 
                 enc_blk_nums=[1, 1, 1], 
                 dec_blk_nums=[1, 1, 1],
                 train_patch=256,
                 eps=0.01,
                 use_global_bank=True,
                 bank_token_num=8,
                 bank_dim=128,
                 bank_patch_size=4,
                 bank_query_downsample=1):
        super(SAIGNet, self).__init__()

        if use_global_bank and len(enc_blk_nums) < 2:
            raise ValueError('Global bank needs at least two encoder stages.')
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.down_illumap = nn.ModuleList()
        self.down_illuscalse = nn.ModuleList()
        
        self.illu_map = SAI2E(in_channels=width, train_patch=train_patch, eps=eps)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.skipfuses = nn.ModuleList()
        self.use_global_bank = use_global_bank
        self.bank_query_blocks = nn.ModuleDict()
        
        chan = width
        encoder_channels = []
        for num in enc_blk_nums:
            encoder_channels.append(chan)
            self.encoders.append(
                CustomSequential(
                    *[EnBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                Downsample(chan)
            )
            self.down_illumap.append(
                Downsample(chan)
            )
            self.down_illuscalse.append(
                Downsample(chan)
            )
            
            chan = chan * 2
            
        if self.use_global_bank:
            self.global_bank = GlobalSceneBank(
                in_dim3=encoder_channels[-1],
                in_dim4=chan,
                patch_size=bank_patch_size,
                patch_embed_dim=bank_dim // 2,
                bank_token_num=bank_token_num,
                bank_dim=bank_dim,
            )
            bank_query_dims = {encoder_channels[-2], encoder_channels[-1]}
        else:
            self.global_bank = None
            bank_query_dims = set()

        self.middle_blks_enc = \
            CustomSequential(
                *[EnBlock(chan) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DeBlock(chan) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                Upsample(chan)
            )
            chan = chan // 2
            self.skipfuses.append(SkipFuse(chan))
            self.decoders.append(
                CustomSequential(
                    *[DeBlock(chan) for _ in range(num)]
                )
            )
            dec_idx = len(self.decoders) - 1
            if chan in bank_query_dims:
                self.bank_query_blocks[str(dec_idx)] = BankQueryBlock(
                    feat_dim=chan,
                    bank_dim=bank_dim,
                    query_downsample=bank_query_downsample,
                )
        
    def forward(self, input):

        _, _, H, W = input.shape

        x = self.intro(input)
        illu_map,illu_scale = self.illu_map(x) 
        
        skips = []
        illumap_skips = []
        for encoder, down, downillu, downscale in zip(self.encoders, self.downs,self.down_illumap,self.down_illuscalse):
            x = encoder(x,illu_map,illu_scale)
            skips.append(x)
            illumap_skips.append(illu_map)
            x = down(x)
            illu_map = downillu(illu_map)
            illu_scale = downscale(illu_scale)
            
        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x, illu_map, illu_scale)

        bank = None
        if self.use_global_bank:
            bank = self.global_bank(skips[-1], x_light)
        
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light, illu_map)
        
        x = x + x_light

        for dec_idx, (decoder, up, skip, skipfuse, illuskip) in enumerate(zip(
                self.decoders, self.ups, skips[::-1], self.skipfuses,
                illumap_skips[::-1])):
            x = up(x)
            x = skipfuse(x, skip)
            bank_query_key = str(dec_idx)
            if bank is not None and bank_query_key in self.bank_query_blocks:
                x = self.bank_query_blocks[bank_query_key](x, bank)
            x = decoder(x, illuskip)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W] # we recover the original size of the image

        return out
      


    
    




