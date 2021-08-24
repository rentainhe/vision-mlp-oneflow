import random
import numpy as np

import oneflow
import oneflow as flow
import oneflow.nn as nn
import oneflow.F as F


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.token_layernorm = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.channel_layernorm = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)
    
    def forward(self, x):
        out = self.token_layernorm(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1,2)
        out = self.channel_layernorm(x)
        x = x + self.channel_mix(out)
        return x

class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=224):
        super(MlpMixer, self).__init__()
        assert (image_size % patch_size) == 0, 'image size must be divisible by the patch size'

        num_tokens = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1,2)
        x = self.mlp(x)
        x = self.layernorm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    test_data = flow.ones((1, 3, 224, 224))
    model = MlpMixer(num_classes=1000,
                     num_blocks=12,
                     patch_size=16,
                     hidden_dim=768,
                     tokens_mlp_dim=384,
                     channels_mlp_dim=3072,
                     image_size=224)
    print(model(test_data).shape)