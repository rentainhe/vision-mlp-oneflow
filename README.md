# vision-mlp-oneflow
Vision MLP Models Based on OneFlow

## TODO LIST
- [x] [Mlp-Mixer]()
- [x] [ResMLP]()
- [x] [gMLP]()
- [ ] [S2-MLP]()
- [ ] [Vision-Permutator]()

## 复现过程中遇到的问题
- `tensor.chunk()`方法没有和`torch.tensor.chunk()`对齐, 在输入的dim参数为负数的时候未作判断, 并且整体功能还变成了repeat操作
- 缺少`tensor.floor_()`方法, 只能用`tensor = tensor.floor()`替代
- 缺少`tensor.erfinv()`方法及其inplace版本, 无法复现`trunc_norm_`
- 缺少`tensor.new_empty()`方法
- `flow.shape`返回的是`oneflow._oneflow_internal.Size` 无法和 `tuple`相加, 按以下方法使用会报错:
```python
# oneflow里报错
x = flow.tensor(np.random.randn(1, 16, 512), dtype=flow.float32)
x.shape + (4, )
x.size() + (4, )
>>>TypeError: unsupported operand type(s) for +: 'oneflow._oneflow_internal.Size' and 'tuple'

# torch里不报错
x = torch.randn(1, 16, 512)
x.shape + (4, )
x.size() + (4, )
>>> torch.Size([1, 16, 512, 4])
```
- `tensor.gather()`未具体判定输入的dim为负数的情况