from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pyramidsplitattention import PyramidSplitAttention

class _LayerNorm(nn.Module):
    def __init__(self, num_features, img_size):
        """
        Normalizes over the entire image and scales + weights for each feature
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            (num_features, img_size, img_size), elementwise_affine=False, eps=1e-12
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )

    def forward(self, x):
        out = self.layer_norm(x)
        out = out * self.weight + self.bias
        return out
class _SamePad(nn.Module):
    """
    Pads equivalent to the behavior of tensorflow "SAME"
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 2 and x.shape[2] % 2 == 0:
            return F.pad(x, (0, 1, 0, 1)) # 对输入矩阵的后两个维度进行扩充
        return F.pad(x, (1, 1, 1, 1))
def _conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    out_size=None,
    activate=True,
    dropout=0.0,
):
    layers = OrderedDict()
    layers["pad"] = _SamePad(stride)
    layers["conv"] = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    if activate:
        if out_size is None:
            raise ValueError("Must provide out_size if activate is True")
        layers["relu"] = nn.LeakyReLU(0.2)
        layers["norm"] = _LayerNorm(out_channels, out_size)

    if dropout > 0.0:
        layers["dropout"] = nn.Dropout(dropout)
    return nn.Sequential(layers)

# 无PSA版本
class _EncoderBlock_0(nn.Module):
    # 无PSA版本
    def __init__(
        self,
        pre_channels, #
        in_channels,
        out_channels,
        num_layers,
        out_size,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pre_conv = _conv2d(
            in_channels=pre_channels, # pre层将通道数调节为pre
            out_channels=pre_channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )

        self.conv0 = _conv2d( # 第1层
            in_channels=in_channels + pre_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            out_size=out_size,
        )
        total_channels = in_channels + out_channels

        for i in range(1, num_layers): # 循环加卷积层
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=total_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    out_size=out_size,
                ),
            )
            total_channels += out_channels # 通道数在增加
        self.add_module(
            "conv%d" % num_layers, # 第num_layers+1层
            _conv2d(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                out_size=(out_size + 1) // 2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)
        out = self.conv0(torch.cat([x, pre_input], 1)) # 和pre层拼接

        all_outputs = [x, out]
        for i in range(1, self.num_layers + 1):
            input_features = torch.cat(
                [all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], 1 # 拼接
            )
            module = self._modules["conv%d" % i]
            out = module(input_features) # 逐层forward
            all_outputs.append(out) # 记录输出到all_outputs
        return all_outputs[-2], all_outputs[-1]

# PSA版本
class _EncoderBlock_1(nn.Module):
    # PSA版本
    def __init__(
        self,
        pre_channels, #
        in_channels,
        out_channels,
        num_layers,
        out_size,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pre_conv = _conv2d(
            in_channels=pre_channels, # pre层将通道数调节为pre
            out_channels=pre_channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )

        self.conv0 = _conv2d( # 第1层
            in_channels=in_channels + pre_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            out_size=out_size,
        )
        total_channels = in_channels + out_channels
        self.conv1 = PyramidSplitAttention( # 插入金字塔模块
            in_channels=total_channels,
            out_channels=out_channels,
        )
        total_channels += out_channels  # 通道数在增加

        for i in range(2, num_layers): # 循环加卷积层
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=total_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    out_size=out_size,
                ),
            )
            total_channels += out_channels # 通道数在增加
        self.add_module(
            "conv%d" % int(num_layers), # 第num_layers+1层
            _conv2d(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                out_size=(out_size + 1) // 2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        # print("1111")
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)

        out = self.conv0(torch.cat([x, pre_input], 1)) # 和pre层拼接

        all_outputs = [x, out]
        for i in range(1, self.num_layers + 1):
            input_features = torch.cat(
                [all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], 1 # 拼接
            )
            module = self._modules["conv%d" % i]
            out = module(input_features) # 逐层forward
            all_outputs.append(out) # 记录输出到all_outputs
        return all_outputs[-2], all_outputs[-1]

# PSA2版本
class _EncoderBlock_2(nn.Module):
    # PSA2版本
    def __init__(
        self,
        pre_channels, #
        in_channels,
        out_channels,
        num_layers,
        out_size,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pre_conv = _conv2d(
            in_channels=pre_channels, # pre层将通道数调节为pre
            out_channels=pre_channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )

        self.conv0 = _conv2d( # 第1层
            in_channels=in_channels + pre_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            out_size=out_size,
        )
        total_channels = in_channels + out_channels

        self.conv1 = PyramidSplitAttention( # 插入金字塔模块
            in_channels=total_channels,
            out_channels=out_channels,
        )
        total_channels += out_channels  # 通道数在增加

        for i in range(2, num_layers): # 循环加卷积层
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=total_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    out_size=out_size,
                ),
            )
            total_channels += out_channels # 通道数在增加
        self.add_module(
            "conv%d" % int(num_layers),
            PyramidSplitAttention(
                in_channels=total_channels,
                out_channels=out_channels,
            ),
        )
        total_channels += out_channels  # 增加通道数

        self.add_module(
            "conv%d" % (num_layers+1), # 第num_layers+1层 ？transition?
            _conv2d(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                out_size=(out_size + 1) // 2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        # print("222")
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)

        out = self.conv0(torch.cat([x, pre_input], 1)) # 和pre层拼接

        all_outputs = [x, out]
        for i in range(1, self.num_layers + 2): # 2个PSA
            input_features = torch.cat(
                [all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], 1 # 将多个特征图组成list,进行cat拼接
            )
            module = self._modules["conv%d" % i]
            out = module(input_features) # 逐层forward
            all_outputs.append(out) # 记录输出到all_outputs
        return all_outputs[-2], all_outputs[-1]

# PSA3版本
class _EncoderBlock_3(nn.Module):
    # PSA3版本
    def __init__(
        self,
        pre_channels, #
        in_channels,
        out_channels,
        num_layers,
        out_size,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pre_conv = _conv2d(
            in_channels=pre_channels, # pre层将通道数调节为pre
            out_channels=pre_channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )

        self.conv0 = _conv2d( # 第1层
            in_channels=in_channels + pre_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            out_size=out_size,
        )
        total_channels = in_channels + out_channels

        self.conv1 = PyramidSplitAttention( # 插入金字塔模块
            in_channels=total_channels,
            out_channels=out_channels,
        )
        total_channels += out_channels  # 通道数在增加

        for i in range(2, num_layers,2): # 循环加卷积层
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=total_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    out_size=out_size,
                ),
            )
            total_channels += out_channels # 通道数在增加

            self.add_module(
                "conv%d" % int(i+1), # 每步后面都加PSA
                PyramidSplitAttention(
                    in_channels=total_channels,
                    out_channels=out_channels,
                ),
            )
            total_channels += out_channels # 通道数在增加


        self.add_module(
            "conv%d" % (num_layers+1), # 第num_layers+1层
            _conv2d(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                out_size=(out_size + 1) // 2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        # print("222")
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)

        out = self.conv0(torch.cat([x, pre_input], 1)) # 和pre层拼接

        all_outputs = [x, out]
        for i in range(1, self.num_layers + 2): # 2个PSA
            input_features = torch.cat(
                [all_outputs[-1], all_outputs[-2]]
                + all_outputs[:-2], 1 #
            )
            module = self._modules["conv%d" % i]
            out = module(input_features) # 逐层forward
            all_outputs.append(out) # 记录输出到all_outputs
        return all_outputs[-2], all_outputs[-1]


class Discriminator(nn.Module):
    def __init__(self, dim, channels, dropout_rate=0.0, z_dim=100):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.channels = channels
        self.layer_sizes = [64, 64, 128, 128]
        self.num_inner_layers = 5 # 为什么模块内有5个卷积？
        block = _EncoderBlock_2 # 设定psa模块

        # Number of times dimension is halved 尺寸减半的次数
        self.depth = len(self.layer_sizes)

        # 每个级别的维度
        self.dim_arr = [dim]
        for i in range(self.depth):
            self.dim_arr.append((self.dim_arr[-1] + 1) // 2)

        # Encoders
        self.encode0 = _conv2d(
            in_channels=self.channels,
            out_channels=self.layer_sizes[0],
            kernel_size=3,
            stride=2,
            out_size=self.dim_arr[1],
        )
        for i in range(1, self.depth): # 这里要解释设置个数 只有1-3个
            self.add_module(
                "encode%d" % i,
                block( # 设定的psa模块
                    pre_channels=self.channels if i == 1 else self.layer_sizes[i - 1],
                    in_channels=self.layer_sizes[i - 1], # 上一层的输出
                    out_channels=self.layer_sizes[i], # 设定的本层输出
                    num_layers=self.num_inner_layers,
                    out_size=self.dim_arr[i],
                    dropout_rate=dropout_rate,
                ),
            )
        self.dense1 = nn.Linear(self.layer_sizes[-1], 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(self.layer_sizes[-1] * self.dim_arr[-1] ** 2 + 1024, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1) # 拼接
        out = [x, self.encode0(x)]
        for i in range(1, len(self.layer_sizes)):
            out = self._modules["encode%d" % i](out)
        out = out[1]

        out_mean = out.mean([2, 3])
        out_flat = torch.flatten(out, 1)

        out = self.dense1(out_mean)
        out = self.leaky_relu(out)
        out = self.dense2(torch.cat([out, out_flat], 1))

        return out

if __name__ == '__main__':
    model = Discriminator(dim=128, channels=1 * 2, dropout_rate=0.5)
    print(model)
    a = torch.randn([16, 1, 128, 128])
    b = torch.randn([16, 1, 128, 128])
    y = model(a,b)
    # print(y.shape)

    # print(y)
