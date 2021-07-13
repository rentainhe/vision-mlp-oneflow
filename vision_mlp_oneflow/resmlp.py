import oneflow
import oneflow.F as F
import oneflow.experimental as flow
from oneflow.experimental import nn
import random
# 开启oneflow的eager动态图模式
flow.enable_eager_execution()
import numpy as np

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(flow.ones((1, 1, dim)), requires_grad=True)
        self.bias = nn.Parameter(flow.zeros((1, 1, dim)), requires_grad=True)
    
    def forward(self, x):
        return x * self.gamma + self.bias

class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth < 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = flow.tensor(np.zeros((1, 1, dim)), dtype=flow.float32).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim=dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.affine(x), **kwargs) * self.scale + x

class ResMLP(nn.Module):
    def __init__(self, *, dim, depth, num_classes, expansion_factor=4, patch_size=16, image_size=224):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.res_mlp_layers = nn.Sequential(*[nn.Sequential(wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
                                                            wrapper(i, nn.Sequential(
                                                                nn.Linear(dim, dim * expansion_factor),
                                                                nn.GELU(),
                                                                nn.Linear(dim * expansion_factor, dim)
                                                            ))
                                                            ) for i in range(depth)])

        self.affine = Affine(dim)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1,2)
        x = self.res_mlp_layers(x)
        x = self.affine(x)
        x = x.transpose(1,2).mean(dim=-1)
        x = self.to_logits(x)
        return x

if __name__ == "__main__":
    x = flow.tensor(np.random.randn(1, 3, 224, 224), dtype=flow.float32)
    net = ResMLP(dim=384, depth=3, num_classes=100)
    print(x.shape)
    print(net(x).shape)
