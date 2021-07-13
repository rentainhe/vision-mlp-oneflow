from random import randrange
import oneflow
import oneflow.F as F
import oneflow.experimental as flow
from oneflow.experimental import nn
import random
# 开启oneflow的eager动态图模式
flow.enable_eager_execution()
import numpy as np
from functools import partial

# functions
def exists(val):
    return val is not None

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        """Single head self-attention used in g-MLP
        """
        super().__init__()
        self.scale = dim_inner ** -0.5

        self.to_q = nn.Linear(dim_in, dim_inner, bias=False)
        self.to_k = nn.Linear(dim_in, dim_inner, bias=False)
        self.to_v = nn.Linear(dim_in, dim_inner, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)
    
    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        dots = flow.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = dots.softmax(dim=-1)
        out = flow.matmul(attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim=None):
        super().__init__()

        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = Attention(dim * 2, dim, attn_dim) if exists(attn_dim) else None
        # initialization mentioned in paper
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        # chunk method has some problem
#         res, gate = x.chunk(2, dim=-1)
        b, _, c = x.shape
        res = x[:, :, :c//2]
        gate = x[:, :, c//2:]
        gate = self.norm(gate)
        gate = self.proj(gate)

        if exists(self.attn):
            gate += self.attn(x)

        return gate * res

class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_patches, attn_dim):
        super(gMLPBlock, self).__init__()
        self.gmlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            SpatialGatingUnit(ffn_dim, num_patches, attn_dim),
            nn.Linear(ffn_dim // 2, hidden_dim)
        )

    def forward(self, x):
        return x + self.gmlp(x)

class gMLPVision(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, ffn_dim, attn_dim=None,
                 image_size=224):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image size must be divisible by the patch size'

        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.gmlp_layers = nn.ModuleList([gMLPBlock(hidden_dim, ffn_dim, num_patches, attn_dim) for _ in range(num_blocks)])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        layers = self.gmlp_layers
        x = nn.Sequential(*layers)(x)
        x = self.layer_norm(x)
        x = x.transpose(-1, -2).mean(dim=-1)
        return self.proj(x)