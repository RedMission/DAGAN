'''
MC+SC版本
MC版本
两者都无版本
'''
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

class _SamePad(nn.Module):
    """
    Pads equivalent to the behavior of tensorflow "SAME"
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 2 and x.shape[2] % 2 == 0:
            return F.pad(x, (0, 1, 0, 1))
        return F.pad(x, (1, 1, 1, 1))

def _conv2d(in_channels, out_channels, kernel_size, stride, activate=True, dropout=0.0):
    layers = OrderedDict()
    layers["pad"] = _SamePad(stride)
    layers["conv"] = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    if activate:
        layers["relu"] = nn.LeakyReLU(0.2)
        layers["batchnorm"] = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    if dropout > 0.0:
        layers["dropout"] = nn.Dropout(dropout)
    return nn.Sequential(layers)

def _conv2d_transpose(
    in_channels, out_channels, kernel_size, upscale_size, activate=True, dropout=0.0
):
    layers = OrderedDict()
    layers["upsample"] = nn.Upsample(upscale_size)
    layers["conv"] = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=1, padding=1
    )
    if activate:
        layers["relu"] = nn.LeakyReLU(0.2)
        layers["batchnorm"] = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    if dropout > 0.0:
        layers["dropout"] = nn.Dropout(dropout)
    return nn.Sequential(layers)

# ##############################  MC+SC版本  ###################################
# class _EncoderBlock(nn.Module):
#     # MC+SC
#     def __init__(
#         self, pre_channels, in_channels, out_channels, num_layers, dropout_rate=0.0
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.pre_conv = _conv2d(
#             in_channels=pre_channels,
#             out_channels=pre_channels,
#             kernel_size=3,
#             stride=2,
#             activate=False,
#         )
#
#         self.conv0 = _conv2d(
#             in_channels=in_channels + pre_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#         )
#         total_channels = in_channels + out_channels
#         for i in range(1, num_layers):
#             self.add_module(
#                 "conv%d" % i,
#                 _conv2d(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#             total_channels += out_channels
#         self.add_module(
#             "conv%d" % num_layers,
#             _conv2d(
#                 in_channels=total_channels,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=2,
#                 dropout=dropout_rate,
#             ),
#         )
#
#     def forward(self, inp):
#         pre_input, x = inp
#         pre_input = self.pre_conv(pre_input) # 前项输出的处理
#         out = self.conv0(torch.cat([x, pre_input], 1)) # 密集连接
#
#         all_outputs = [x, out]
#         for i in range(1, self.num_layers + 1):
#             input_features = torch.cat(all_outputs, 1)
#             module = self._modules["conv%d" % i]
#             out = module(input_features)
#             all_outputs.append(out)
#         return all_outputs[-2], all_outputs[-1]
#
# class _DecoderBlock(nn.Module):
#     # MC+SC
#     def __init__(
#         self,
#         pre_channels,
#         in_channels,
#         out_channels,
#         num_layers,
#         curr_size,
#         upscale_size=None,
#         dropout_rate=0.0,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.should_upscale = upscale_size is not None
#         self.should_pre_conv = pre_channels > 0
#
#         total_channels = pre_channels + in_channels
#         for i in range(num_layers):
#             if self.should_pre_conv: # 判断是否需要pre层
#                 self.add_module(
#                     "pre_conv_t%d" % i,
#                     _conv2d_transpose(
#                         in_channels=pre_channels,
#                         out_channels=pre_channels,
#                         kernel_size=3,
#                         upscale_size=curr_size,
#                         activate=False,
#                     ),
#                 )
#             self.add_module(
#                 "conv%d" % i,
#                 _conv2d(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#             total_channels += out_channels
#
#         if self.should_upscale: # 判断是否需要上采样
#             total_channels -= pre_channels
#             self.add_module(
#                 "conv_t%d" % num_layers,
#                 _conv2d_transpose(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     upscale_size=upscale_size,
#                     dropout=dropout_rate,
#                 ),
#             )
#
#     def forward(self, inp):
#         pre_input, x = inp
#         all_outputs = [x]
#         for i in range(self.num_layers):
#             curr_input = all_outputs[-1]
#             if self.should_pre_conv:
#                 pre_conv_output = self._modules["pre_conv_t%d" % i](pre_input)
#                 curr_input = torch.cat([curr_input, pre_conv_output], 1)
#             input_features = torch.cat([curr_input] + all_outputs[:-1], 1)
#             module = self._modules["conv%d" % i]
#             out = module(input_features)
#             all_outputs.append(out)
#
#         if self.should_upscale:
#             module = self._modules["conv_t%d" % self.num_layers]
#             input_features = torch.cat(all_outputs, 1)
#             out = module(input_features)
#             all_outputs.append(out)
#         return all_outputs[-2], all_outputs[-1]
#
# class Generator(nn.Module):
#     # MC+SC
#     def __init__(self, dim, channels, dropout_rate=0.0, z_dim=100):
#         super().__init__()
#         self.dim = dim
#         self.z_dim = dim
#         self.channels = channels
#         self.layer_sizes = [64, 64, 128, 128]
#         self.num_inner_layers = 3
#
#         # Number of times dimension is halved
#         self.U_depth = len(self.layer_sizes)
#
#         # dimension at each level of U-net
#         self.dim_arr = [dim]
#         for i in range(self.U_depth):
#             self.dim_arr.append((self.dim_arr[-1] + 1) // 2)
#
#         # Encoders
#         self.encode0 = _conv2d(
#             in_channels=1, out_channels=self.layer_sizes[0], kernel_size=3, stride=2
#         )
#         for i in range(1, self.U_depth):
#             self.add_module(
#                 "encode%d" % i,
#                 _EncoderBlock(
#                     pre_channels=self.channels if i == 1 else self.layer_sizes[i - 1],
#                     in_channels=self.layer_sizes[i - 1],
#                     out_channels=self.layer_sizes[i],
#                     num_layers=self.num_inner_layers,
#                     dropout_rate=dropout_rate,
#                 ),
#             )
#
#         # Noise encoders
#         self.noise_encoders = 3
#         num_noise_filters = 8
#         self.z_channels = []
#         for i in range(self.noise_encoders):
#             curr_dim = self.dim_arr[-1 - i]  # Iterate dim from back
#             self.add_module(
#                 "z_reshape%d" % i,
#                 nn.Linear(self.z_dim, curr_dim * curr_dim * num_noise_filters),
#             )
#             self.z_channels.append(num_noise_filters)
#             num_noise_filters //= 2 # 噪声滤波器数量减半
#
#         # Decoders
#         for i in range(self.U_depth + 1):
#             # Input from previous decoder
#             in_channels = 0 if i == 0 else self.layer_sizes[-i]
#             # Input from encoder across the "U"
#
#             # in_channels += (
#             #     self.channels-1 if i == self.U_depth else self.layer_sizes[-i - 1]
#             # )
#             # 密集连接的修改：增加其他层D输出的连接
#             in_channels += (
#                     sum(self.layer_sizes[: -i]) + self.channels if i > 0 else self.layer_sizes[-i - 1]
#             )
#
#
#             # Input from injected noise
#             if i < self.noise_encoders:
#                 in_channels += self.z_channels[i]
#
#             self.add_module(
#                 "decode%d" % i,
#                 _DecoderBlock(
#                     pre_channels=0 if i == 0 else self.layer_sizes[-i],
#                     in_channels=in_channels,
#                     out_channels=self.layer_sizes[0]
#                     if i == self.U_depth
#                     else self.layer_sizes[-i - 1],
#                     num_layers=self.num_inner_layers,
#                     curr_size=self.dim_arr[-i - 1],
#                     upscale_size=None if i == self.U_depth else self.dim_arr[-i - 2],
#                     dropout_rate=dropout_rate,
#                 ),
#             )
#
#         # Final conv
#         self.num_final_conv = 3
#         for i in range(self.num_final_conv - 1):
#             self.add_module(
#                 "final_conv%d" % i,
#                 _conv2d(
#                     in_channels=self.layer_sizes[0],
#                     out_channels=self.layer_sizes[0],
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#         self.add_module(
#             "final_conv%d" % (self.num_final_conv - 1),
#             _conv2d(
#                 in_channels=self.layer_sizes[0],
#                 out_channels=self.channels,
#                 kernel_size=3,
#                 stride=1,
#                 activate=False,
#             ),
#         )
#         self.tanh = nn.Tanh()
#
#     def forward(self, x, z):
#         # Final output of every encoding block
#         all_outputs = [x, self.encode0(x)] # 已有两项
#
#         # Last 2 layer outputs
#         out = [x, self.encode0(x)]
#         for i in range(1, len(self.layer_sizes)):
#             out = self._modules["encode%d" % i](out) # 对外层out更新了吧
#             all_outputs.append(out[1])
#             # all_outputs长度变化范围3，4，5
#
#
#         pre_input, curr_input = None, out[1]
#         for i in range(self.U_depth + 1):
#             if i > 0:
#                 # 拼接解码器的输出
#                 curr_input = torch.cat([curr_input, all_outputs[-i - 1]], 1) # 按照通道（channel）维度拼接
#                 sf = 1
#                 for layer in range(-i-2, -len(all_outputs)-1, -1):
#                     sf /= 2
#                     cat_output = nn.functional.interpolate(all_outputs[layer], scale_factor=sf, mode='bilinear', align_corners=True,
#                                                      recompute_scale_factor=True)
#                     curr_input = torch.cat([curr_input, cat_output], 1)  # 按照通道（channel）维度拼接
#             if i < self.noise_encoders:
#                 z_out = self._modules["z_reshape%d" % i](z)
#
#                 curr_dim = self.dim_arr[-i - 1]
#                 z_out = z_out.view(-1, self.z_channels[i], curr_dim, curr_dim)
#                 curr_input = torch.cat([z_out, curr_input], 1) # 与噪声拼接
#
#             pre_input, curr_input = self._modules["decode%d" % i](
#                 [pre_input, curr_input]
#             )
#
#         for i in range(self.num_final_conv):
#             curr_input = self._modules["final_conv%d" % i](curr_input)
#         return self.tanh(curr_input)
# # # # # # # # # # # # # # # #   MC版本  # # # # # # # # # # # # # # # # # # #
# class _EncoderBlock(nn.Module):
#     # MC版
#     def __init__(
#         self, pre_channels, in_channels, out_channels, num_layers, dropout_rate=0.0
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.pre_conv = _conv2d(
#             in_channels=pre_channels,
#             out_channels=pre_channels,
#             kernel_size=3,
#             stride=2,
#             activate=False,
#         )
#
#         self.conv0 = _conv2d(
#             in_channels=in_channels + pre_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#         )
#         total_channels = in_channels + out_channels
#         for i in range(1, num_layers):
#             self.add_module(
#                 "conv%d" % i,
#                 _conv2d(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#             total_channels += out_channels
#         self.add_module(
#             "conv%d" % num_layers,
#             _conv2d(
#                 in_channels=total_channels,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=2,
#                 dropout=dropout_rate,
#             ),
#         )
#
#     def forward(self, inp):
#         pre_input, x = inp
#         pre_input = self.pre_conv(pre_input) # 前项输出的处理
#         out = self.conv0(torch.cat([x, pre_input], 1)) # 密集连接
#
#         all_outputs = [x, out]
#         for i in range(1, self.num_layers + 1):
#             input_features = torch.cat(all_outputs, 1)
#             module = self._modules["conv%d" % i]
#             out = module(input_features)
#             all_outputs.append(out)
#         return all_outputs[-2], all_outputs[-1]
#
#
# class _DecoderBlock(nn.Module):
#     # MC版
#     def __init__(
#         self,
#         pre_channels,
#         in_channels,
#         out_channels,
#         num_layers,
#         curr_size,
#         upscale_size=None,
#         dropout_rate=0.0,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.should_upscale = upscale_size is not None
#         self.should_pre_conv = pre_channels > 0
#
#         total_channels = pre_channels + in_channels
#         for i in range(num_layers):
#             if self.should_pre_conv: # 判断是否需要pre层
#                 self.add_module(
#                     "pre_conv_t%d" % i,
#                     _conv2d_transpose(
#                         in_channels=pre_channels,
#                         out_channels=pre_channels,
#                         kernel_size=3,
#                         upscale_size=curr_size,
#                         activate=False,
#                     ),
#                 )
#             self.add_module(
#                 "conv%d" % i,
#                 _conv2d(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#             total_channels += out_channels
#
#         if self.should_upscale: # 判断是否需要上采样
#             total_channels -= pre_channels
#             self.add_module(
#                 "conv_t%d" % num_layers,
#                 _conv2d_transpose(
#                     in_channels=total_channels,
#                     out_channels=out_channels,
#                     kernel_size=3,
#                     upscale_size=upscale_size,
#                     dropout=dropout_rate,
#                 ),
#             )
#
#     def forward(self, inp):
#         pre_input, x = inp
#         all_outputs = [x]
#         for i in range(self.num_layers):
#             curr_input = all_outputs[-1]
#             if self.should_pre_conv:
#                 pre_conv_output = self._modules["pre_conv_t%d" % i](pre_input)
#                 curr_input = torch.cat([curr_input, pre_conv_output], 1)
#             input_features = torch.cat([curr_input] + all_outputs[:-1], 1)
#             module = self._modules["conv%d" % i]
#             out = module(input_features)
#             all_outputs.append(out)
#
#         if self.should_upscale:
#             module = self._modules["conv_t%d" % self.num_layers]
#             input_features = torch.cat(all_outputs, 1)
#             out = module(input_features)
#             all_outputs.append(out)
#         return all_outputs[-2], all_outputs[-1]
#
#
# class Generator(nn.Module):
#     # MC版
#     def __init__(self, dim, channels, dropout_rate=0.0, z_dim=100):
#         super().__init__()
#         self.dim = dim
#         self.z_dim = dim
#         self.channels = channels
#         self.layer_sizes = [64, 64, 128, 128]
#         self.num_inner_layers = 3
#
#         # Number of times dimension is halved
#         self.U_depth = len(self.layer_sizes)
#
#         # dimension at each level of U-net
#         self.dim_arr = [dim]
#         for i in range(self.U_depth):
#             self.dim_arr.append((self.dim_arr[-1] + 1) // 2)
#
#         # Encoders
#         self.encode0 = _conv2d(
#             in_channels=1, out_channels=self.layer_sizes[0], kernel_size=3, stride=2
#         )
#         for i in range(1, self.U_depth):
#             self.add_module(
#                 "encode%d" % i,
#                 _EncoderBlock(
#                     pre_channels=self.channels if i == 1 else self.layer_sizes[i - 1],
#                     in_channels=self.layer_sizes[i - 1],
#                     out_channels=self.layer_sizes[i],
#                     num_layers=self.num_inner_layers,
#                     dropout_rate=dropout_rate,
#                 ),
#             )
#
#         # Noise encoders
#         self.noise_encoders = 3
#         num_noise_filters = 8
#         self.z_channels = []
#         for i in range(self.noise_encoders):
#             curr_dim = self.dim_arr[-1 - i]  # Iterate dim from back
#             self.add_module(
#                 "z_reshape%d" % i,
#                 nn.Linear(self.z_dim, curr_dim * curr_dim * num_noise_filters),
#             )
#             self.z_channels.append(num_noise_filters)
#             num_noise_filters //= 2 # 噪声滤波器数量减半
#
#         # Decoders
#         for i in range(self.U_depth + 1):
#             # Input from previous decoder
#             in_channels = 0 if i == 0 else self.layer_sizes[-i]
#             # Input from encoder across the "U"
#             in_channels += (
#                 self.channels if i == self.U_depth else self.layer_sizes[-i - 1]
#             )
#             # Input from injected noise
#             if i < self.noise_encoders:
#                 in_channels += self.z_channels[i]
#
#             self.add_module(
#                 "decode%d" % i,
#                 _DecoderBlock(
#                     pre_channels=0 if i == 0 else self.layer_sizes[-i],
#                     in_channels=in_channels,
#                     out_channels=self.layer_sizes[0]
#                     if i == self.U_depth
#                     else self.layer_sizes[-i - 1],
#                     num_layers=self.num_inner_layers,
#                     curr_size=self.dim_arr[-i - 1],
#                     upscale_size=None if i == self.U_depth else self.dim_arr[-i - 2],
#                     dropout_rate=dropout_rate,
#                 ),
#             )
#
#         # Final conv
#         self.num_final_conv = 3
#         for i in range(self.num_final_conv - 1):
#             self.add_module(
#                 "final_conv%d" % i,
#                 _conv2d(
#                     in_channels=self.layer_sizes[0],
#                     out_channels=self.layer_sizes[0],
#                     kernel_size=3,
#                     stride=1,
#                 ),
#             )
#         self.add_module(
#             "final_conv%d" % (self.num_final_conv - 1),
#             _conv2d(
#                 in_channels=self.layer_sizes[0],
#                 out_channels=self.channels,
#                 kernel_size=3,
#                 stride=1,
#                 activate=False,
#             ),
#         )
#         self.tanh = nn.Tanh()
#
#     def forward(self, x, z):
#         # Final output of every encoding block
#         all_outputs = [x, self.encode0(x)] # 已有两项
#
#         # Last 2 layer outputs
#         out = [x, self.encode0(x)]
#         for i in range(1, len(self.layer_sizes)):
#             out = self._modules["encode%d" % i](out) # 对外层out更新了吧
#             all_outputs.append(out[1])
#             # all_outputs长度变化范围3，4，5
#
#         pre_input, curr_input = None, out[1]
#         for i in range(self.U_depth + 1):
#             if i > 0:
#                 # 拼接解码器的输出
#                 curr_input = torch.cat([curr_input, all_outputs[-i - 1]], 1) # 按照通道（channel）维度拼接
#
#             if i < self.noise_encoders:
#                 z_out = self._modules["z_reshape%d" % i](z)
#
#                 curr_dim = self.dim_arr[-i - 1]
#                 z_out = z_out.view(-1, self.z_channels[i], curr_dim, curr_dim)
#                 curr_input = torch.cat([z_out, curr_input], 1) # 与噪声拼接
#
#             pre_input, curr_input = self._modules["decode%d" % i](
#                 [pre_input, curr_input]
#             )
#
#         for i in range(self.num_final_conv):
#             curr_input = self._modules["final_conv%d" % i](curr_input)
#         return self.tanh(curr_input)
# # # # # # # # # # # # # # # # # # #

# ###############################两者都无版本######################################
class _EncoderBlock(nn.Module):
    # 无密集连接版本
    def __init__(
        self, in_channels, out_channels, num_layers, dropout_rate=0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.conv0 = _conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )
        for i in range(1, num_layers):
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                ),
            )
        self.add_module(
            "conv%d" % num_layers,
            _conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        out = self.conv0(inp) # 密集连接
        all_outputs = [out]
        for i in range(1, self.num_layers + 1):
            input_features = all_outputs[-1]
            module = self._modules["conv%d" % i]
            out = module(input_features)
            all_outputs.append(out)
        return all_outputs[-1]
class _DecoderBlock(nn.Module):
    # 无密集连接版本
    def __init__(
        self,
        pre_channels,
        in_channels,
        out_channels,
        num_layers,
        curr_size,
        upscale_size=None,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.should_upscale = upscale_size is not None
        self.should_pre_conv = pre_channels > 0

        for i in range(num_layers):
            if self.should_pre_conv: # 判断是否需要pre层
                self.add_module(
                    "pre_conv_t%d" % i,
                    _conv2d_transpose(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        upscale_size=curr_size,
                        activate=False,
                    ),
                )
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                ),
            )

        if self.should_upscale: # 判断是否需要上采样
            self.add_module(
                "conv_t%d" % num_layers,
                _conv2d_transpose(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    upscale_size=upscale_size,
                    dropout=dropout_rate,
                ),
            )

    def forward(self, inp):
        curr_input = self._modules["pre_conv_t%d" % 0](inp)
        # print("decode修改后：",curr_input.shape)
        outputs = [curr_input]
        for i in range(self.num_layers):
            curr_input = outputs[-1]
            #     curr_input = torch.cat([curr_input], 1)
            # input_features = torch.cat([curr_input] + all_outputs[:-1], 1)
            module = self._modules["conv%d" % i]
            # 尺寸问题
            out = module(curr_input)
            outputs.append(out)

        if self.should_upscale:
            module = self._modules["conv_t%d" % self.num_layers]
            out = module(outputs[-1])
        return out
class Generator(nn.Module):
    # 无密集连接MC版本
    def __init__(self, dim, channels, dropout_rate=0.0, z_dim=100):
        super().__init__()
        self.dim = dim
        self.z_dim = dim
        self.channels = channels
        self.layer_sizes = [64, 64, 128, 128]
        self.num_inner_layers = 3

        # Number of times dimension is halved
        self.U_depth = len(self.layer_sizes)

        # dimension at each level of U-net
        self.dim_arr = [dim]
        for i in range(self.U_depth):
            self.dim_arr.append((self.dim_arr[-1] + 1) // 2)

        # Encoders
        self.encode0 = _conv2d(
            in_channels=1, out_channels=self.layer_sizes[0], kernel_size=3, stride=2
        )
        for i in range(1, self.U_depth):
            self.add_module(
                "encode%d" % i,
                _EncoderBlock(
                    in_channels=self.layer_sizes[i - 1],
                    out_channels=self.layer_sizes[i],
                    num_layers=self.num_inner_layers,
                    dropout_rate=dropout_rate,
                ),
            )

        # Noise encoders
        self.noise_encoders = 3
        num_noise_filters = 8
        self.z_channels = []
        for i in range(self.noise_encoders):
            curr_dim = self.dim_arr[-1 - i]  # Iterate dim from back
            self.add_module(
                "z_reshape%d" % i,
                nn.Linear(self.z_dim, curr_dim * curr_dim * num_noise_filters),
            )
            self.z_channels.append(num_noise_filters)
            num_noise_filters //= 2 # 噪声滤波器数量减半

        # Decoders
        for i in range(self.U_depth):
            in_channels = self.layer_sizes[-i-1]

            if i < self.noise_encoders:
                in_channels += self.z_channels[i]

            self.add_module(
                "decode%d" % i,
                _DecoderBlock(
                    pre_channels = self.layer_sizes[-i],
                    in_channels=in_channels,
                    out_channels=self.layer_sizes[-i-2] if i<self.U_depth-1 else self.layer_sizes[0]
                    if i == self.U_depth
                    else self.layer_sizes[-i - 1],
                    num_layers=self.num_inner_layers,
                    curr_size=self.dim_arr[-i - 1],
                    upscale_size=None if i == self.U_depth else self.dim_arr[-i - 2],
                    dropout_rate=dropout_rate,
                ),
            )

        # Final conv
        self.num_final_conv = 3
        for i in range(self.num_final_conv - 1):
            self.add_module(
                "final_conv%d" % i,
                _conv2d(
                    in_channels=self.layer_sizes[0],
                    out_channels=self.layer_sizes[0],
                    kernel_size=3,
                    stride=1,
                ),
            )
        self.add_module(
            "final_conv%d" % (self.num_final_conv - 1),
            _conv2d(
                in_channels=self.layer_sizes[0],
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                activate=False,
            ),
        )
        self.tanh = nn.Tanh()

    def forward(self, x, z):
        # Final output of every encoding block
        all_outputs = [x, self.encode0(x)] # 已有两项
        # Last 2 layer outputs
        for i in range(1, len(self.layer_sizes)):
            out = self._modules["encode%d" % i](all_outputs[-1]) # 对外层out更新
            all_outputs.append(out)
            # all_outputs最后尺寸：torch.Size([10, 128, 8, 8])

        curr_input = all_outputs[-1]
        for i in range(self.U_depth):
            if i < self.noise_encoders:
                z_out = self._modules["z_reshape%d" % i](z)
                curr_dim = self.dim_arr[-i - 1]
                z_out = z_out.view(-1, self.z_channels[i], curr_dim, curr_dim)
                curr_input = torch.cat([z_out, curr_input], 1) # 与噪声拼接
            # print("生成器的解码：", i)
            # print("in形状：",curr_input.shape)
            curr_input = self._modules["decode%d" % i](curr_input)
            # print("out形状：",curr_input.shape)


        for i in range(self.num_final_conv):
            curr_input = self._modules["final_conv%d" % i](curr_input)
        return self.tanh(curr_input)

if __name__ == '__main__':
    # 测试是否可以运行
    model = Generator(dim=128, channels=1, dropout_rate=0.5)
    a = torch.randn([10,1,128,128])
    z = torch.randn([10,128])
    y = model(a,z)
    print(y.shape)