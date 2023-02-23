import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=3, conv_kernel_size=1, pool_kernel_size=3):
        super(PAM, self).__init__()
        self.num_levels = num_levels
        self.conv_channels = out_channels // 2

        self.query_conv = nn.Conv2d(in_channels, self.conv_channels, kernel_size=conv_kernel_size)
        self.key_conv = nn.Conv2d(in_channels, self.conv_channels, kernel_size=conv_kernel_size)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size)

        self.max_pool = nn.MaxPool2d(pool_kernel_size, stride=2, padding=1)

        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 计算金字塔切分的大小和位置
        sizes = [(height // (2 ** i), width // (2 ** i)) for i in range(self.num_levels)]
        centers = [(height // (2 ** i) // 2, width // (2 ** i) // 2) for i in range(self.num_levels)]

        # 生成金字塔切分
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        # F.interpolate 实现上 / 下采样操作
        queries = [F.interpolate(query, size=s, mode='bilinear', align_corners=True) for s in sizes]
        keys = [F.interpolate(key, size=s, mode='bilinear', align_corners=True) for s in sizes]
        values = [F.interpolate(value, size=s, mode='bilinear', align_corners=True) for s in sizes]

        # 计算注意力权重
        attention_maps = []
        for i in range(self.num_levels):
            q = queries[i]
            k = keys[i]
            v = values[i]

            score = torch.matmul(q.view(batch_size, self.conv_channels, -1).permute(0, 2, 1), k.view(batch_size, self.conv_channels, -1))
            attention = F.softmax(score, dim=2)

            attention_maps.append(attention)

        # 将注意力权重应用于金字塔切分
        pyramid = []
        for i in range(self.num_levels):
            v = values[i]
            a = attention_maps[i]

            w = torch.matmul(a, v.view(batch_size, self.conv_channels, -1).permute(0, 2, 1))
            w = w.permute(0, 2, 1).contiguous().view(batch_size, self.conv_channels, sizes[i][0], sizes[i][1])

            pyramid.append(w)

        # 级联金字塔切分
        merged = torch.cat(pyramid, dim=1)

        # 应用卷积层
        merged = self.conv_bn(merged)
        return merged

if __name__ == '__main__':
    test = PAM(1, 16)
    x = torch.randn(1,1,32,32)
    print(x)
    y = test(x)
    print("-------------")
    print(y)