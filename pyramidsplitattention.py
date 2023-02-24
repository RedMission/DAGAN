import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidSplitAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, scales=[3, 6]):
        super(PyramidSplitAttention, self).__init__()
        self.out_channels = out_channels
        self.scales = scales
        self.num_scales = len(scales)

        if out_channels is None:
            self.out_channels = in_channels // 4

        self.query_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.key_conv = nn.ModuleList([nn.Conv2d(in_channels, self.out_channels, kernel_size=1) for _ in range(self.num_scales)])
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, self.out_channels, kernel_size=1) for _ in range(self.num_scales)])
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv_concat = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate query, key, value
        query = self.query_conv(x)
        keys, values = [], []
        for i in range(self.num_scales):
            scale_factor = self.scales[i]
            h, w = height // scale_factor, width // scale_factor
            x_level = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
            key = self.key_conv[i](x_level)
            value = self.value_conv[i](x_level)
            keys.append(key.view(batch_size, -1, h * w))
            values.append(value.view(batch_size, -1, h * w))

        # Concatenate keys and values across scales
        keys = torch.cat(keys, dim=2)
        values = torch.cat(values, dim=2)

        # Calculate attention map and weighted values
        energy = torch.bmm(query.view(batch_size, -1, height * width).permute(0, 2, 1), keys)
        attention = F.softmax(energy, dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdim=True))
        weighted_values = torch.bmm(values, attention.permute(0, 2, 1))
        weighted_values = weighted_values.view(batch_size, -1, height, width)

        # Apply transformations and concatenate across scales
        x = self.conv(x)
        out = self.gamma * weighted_values + x
        out = self.conv_concat(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

if __name__ == '__main__':
    # test PyramidAttentionModule with random input
    batch_size = 16
    in_channels = 128
    out_channels = 64
    height = 64
    width = 64
    num_levels = 3

    input_tensor = torch.randn(batch_size, in_channels, height, width)
    pam = PyramidSplitAttention(in_channels, out_channels)
    output_tensor = pam(input_tensor)

    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)
