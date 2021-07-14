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
import math
import warnings

# oneflow version
def truncated_normal_(tensor, mean, std=0.09):
    with flow.no_grad():
        size = tuple(tensor.shape) + (4, )
        tmp = flow.tensor(np.random.randn(*size), dtype=flow.float32).normal_()
        valid = (tmp < 2) and (tmp > -2)
        ind = valid.max(-1, keepdim=True)
        # 注意index参数必须为long类型
        # 必须为numpy.ndarray才可以deep copy
        tensor.data.copy_(tmp.gather(dim = 2, index = ind.long()).squeeze(-1).numpy())
        tensor.data.copy_(flow.mul(tensor.data, std).add_(mean).numpy())
        return tensor

# torch version
# def truncated_normal(tensor, mean=0, std=0.09):
#     with torch.no_grad():
#         size = tuple(tensor.size())
#         tmp = tensor.new_empty(size+(4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.data.mul_(std).add_(mean)
#         return tensor