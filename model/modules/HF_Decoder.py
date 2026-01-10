import torch
import torch.nn as nn
import torch.nn.functional as F

from .GFM import GFM_Module
from .DGG import DGG_Module
from .ISF import ISF_Module


class MLP(nn.Module):
    """Simple MLP for decoder"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class HiF_Decoder(nn.Module):
    """Hierarchical Factorized Decoder"""
    def __init__(
        self,
        encoder_channels=[64, 128, 320, 512],
        decoder_channels=256,
    ):
        super().__init__()
        
        # MLP layers to unify channel dimensions
        self.linear_c4 = MLP(input_dim=encoder_channels[3], embed_dim=decoder_channels)
        self.linear_c3 = MLP(input_dim=encoder_channels[2], embed_dim=decoder_channels)
        self.linear_c2 = MLP(input_dim=encoder_channels[1], embed_dim=decoder_channels)
        self.linear_c1 = MLP(input_dim=encoder_channels[0], embed_dim=decoder_channels)

        self.dropout = nn.Dropout2d(0.1)

        self.gfm_c4_1 = GFM_Module(decoder_channels, decoder_channels//2)
        self.gfm_c3_1 = GFM_Module(decoder_channels, decoder_channels//2)
        self.gfm_c2_1 = GFM_Module(decoder_channels, decoder_channels//2)
        self.gfm_c1_1 = GFM_Module(decoder_channels, decoder_channels//2)

        self.gfm_c_o_1 = GFM_Module(decoder_channels, decoder_channels//2)
        self.gfm_c_e_1 = GFM_Module(decoder_channels, decoder_channels//2)

        self.gfm_c_o_2 = GFM_Module(decoder_channels//2, decoder_channels//4)
        self.gfm_c_e_2 = GFM_Module(decoder_channels//2, decoder_channels//4)

        self.gfm_c_o_3 = GFM_Module(decoder_channels//4, decoder_channels//8)
        self.gfm_c_e_3 = GFM_Module(decoder_channels//4, decoder_channels//8)

        self.cyclic_shuffle_enhancer_o = ISF_Module(channels=decoder_channels, groups=4, kernel_size=3, cyclic_percent=0.0)
        self.cyclic_shuffle_enhancer_e = ISF_Module(channels=decoder_channels, groups=4, kernel_size=3, cyclic_percent=0.0)

        self.gatefuser = DGG_Module(channels=decoder_channels//4, groups=4)

    def forward(self, encoder_features):
        # Encoder features: [c1, c2, c3, c4] with shapes [H/4, H/8, H/16, H/32]
        c1, c2, c3, c4 = encoder_features

        # Get target size (H/4, W/4) - same as c1
        n, _, h, w = c1.shape

        # Transform each feature and upsample to H/4
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # c1 is already at the target size, no need to interpolate

        # Concatenate and fuse
        # print(_c4.shape, _c3.shape, _c2.shape, _c1.shape)

        # First Stage Ghost
        # 4*256=1024 -> 8*64=512
        _c4_g1_o, _c4_g2_e = self.gfm_c4_1(_c4)
        _c3_g1_o, _c3_g2_e = self.gfm_c3_1(_c3)
        _c2_g1_o, _c2_g2_e = self.gfm_c2_1(_c2)
        _c1_g1_o, _c1_g2_e = self.gfm_c1_1(_c1)
        # 2*4*64 -> 2*256=512 -> 4*64=256
        _c_o_1 = torch.cat([_c4_g1_o, _c3_g1_o, _c2_g1_o, _c1_g1_o], dim=1) # B, 256, H, W
        _c_e_1 = torch.cat([_c4_g2_e, _c3_g2_e, _c2_g2_e, _c1_g2_e], dim=1) # B, 256, H, W
        _c_o_1_f = self.cyclic_shuffle_enhancer_o(_c_o_1) # fused _c_o_1 feature
        _c_e_1_f = self.cyclic_shuffle_enhancer_e(_c_e_1) # fused _c_e_1 feature

        _c_o_1_o, _c_o_1_e = self.gfm_c_o_1(_c_o_1_f)
        _c_e_1_o, _c_e_1_e = self.gfm_c_e_1(_c_e_1_f)
        
        # Second Stage Ghost
        # 2*2*64=256 -> 2*128 -> 4*32=128
        _c_o_2 = torch.cat([_c_o_1_o, _c_e_1_o], dim=1)   # (B, 128, H, W)
        _c_e_2 = torch.cat([_c_o_1_e, _c_e_1_e], dim=1)   # (B, 128, H, W)
        _c_o_2_o, _c_o_2_e = self.gfm_c_o_2(_c_o_2)    # (B, 32 H, W), (B, 32, H, W)
        _c_e_2_o, _c_e_2_e = self.gfm_c_e_2(_c_e_2)    # (B, 32 H, W), (B, 32, H, W)
        
        # Third Stage Ghost
        # 2*2*32=128 -> 2*64 -> 4*16=64
        _c_o_3 = torch.cat([_c_o_2_o, _c_e_2_o], dim=1)   # (B, 64, H, W)
        _c_e_3 = torch.cat([_c_o_2_e, _c_e_2_e], dim=1)   # (B, 64, H, W)
        _c_o_3_o, _c_o_3_e = self.gfm_c_o_3(_c_o_3)    # (B, 16 H, W), (B, 16, H, W)
        _c_e_3_o, _c_e_3_e = self.gfm_c_e_3(_c_e_3)    # (B, 16 H, W), (B, 16, H, W)

        x = torch.cat([_c_o_3_o, _c_e_3_o, _c_o_3_e, _c_e_3_e], dim=1) # (B, 64, H, W)
        x_f = self.gatefuser(x)
        x = x + x_f
        x = self.dropout(x)
        return x
