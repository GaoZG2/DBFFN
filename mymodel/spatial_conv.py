import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.relu_depthwise = nn.ReLU(inplace=True)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.relu_pointwise = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu_depthwise(x)
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = self.relu_pointwise(x)
        return x

class SpatialConv(nn.Module):
    def __init__(self, kernel_widthNheight=7, width=25, height=25, processed_data_depth=200):
        super(SpatialConv, self).__init__()
        padding2D = (kernel_widthNheight // 2, kernel_widthNheight // 2)
        self.width=width
        self.height=height
        self.conv2D = nn.Sequential(
            nn.Conv2d(processed_data_depth, processed_data_depth, kernel_size=(kernel_widthNheight, kernel_widthNheight),stride=1,padding=padding2D),
            nn.BatchNorm2d(processed_data_depth),
            nn.ReLU(inplace=True),
            nn.Conv2d(processed_data_depth, 64, kernel_size=(kernel_widthNheight, kernel_widthNheight), stride=1, padding=padding2D),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, kernel_size=(kernel_widthNheight, kernel_widthNheight), stride=1, padding=padding2D),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        assert self.width == x.shape[2], "Spatial module: x.shape[2] != self.width"
        assert self.height == x.shape[3], "Spatial module: x.shape[3] != self.height"
        x = x.permute(0, 1, 4, 2, 3).view(x.shape[0], -1, self.width, self.height)
        spatial_conv = self.conv2D(x)
        spatial_conv = spatial_conv.permute(0, 2, 3, 1).contiguous().view(x.shape[0], 1, self.width, self.height, -1)
        return spatial_conv