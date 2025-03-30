import torch
import torch.nn as nn
import config

class SpectralAttention(nn.Module):
    def __init__(self, width, height, processed_data_depth):
        super(SpectralAttention, self).__init__()
        self.width = width
        self.height = height

        self.center_layers = nn.Sequential(
            # 第一个卷积层
            nn.Linear(processed_data_depth, processed_data_depth // config.se_ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(inplace=True),
            nn.Linear(processed_data_depth // config.se_ratio, processed_data_depth, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        self.fusion_se = FusionSE(processed_data_depth, config.se_ratio)

    def forward(self, x):
        """
        前向传播函数，按照给定步骤处理输入特征图
        """
        batch_size, _, width, height, depth = x.size()
        assert width == self.width, "输入张量的 width 维度与模块初始化的 width 不一致"
        assert height == self.height, "输入张量的 height 维度与模块初始化的 height 不一致"
        input_tensor = x.permute(0, 1, 4, 2, 3).contiguous()
        # 调整后的形状变为 (batch_size, channels * depth, width, height)，这里将通道维度和depth维度合并为通道维度
        input_tensor = input_tensor.view(batch_size, -1, width, height)
        se_out = self.fusion_se(input_tensor).permute(0, 2, 3, 1).view(batch_size, 1, width, height, depth)
        return x + se_out
    
class GetCenter(nn.Module):
    def __init__(self):
        super(GetCenter, self).__init__()

    def forward(self, x):
        # 提取中心元素（hwidth/2, height/2）
        b, c, width, height = x.size()
        center_element = x[:, :, width//2, height//2]  # 直接提取中心点 (b,c)
        return center_element
      
class FusionSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FusionSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.center_pixel = GetCenter()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        mean_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        center_out = self.center_pixel(x)
        concat_out = torch.cat([mean_out, max_out, center_out], dim=1)
        b, c, _, _ = x.size()
        y = self.fc(concat_out).view(b, c, 1, 1)
        return x * y.expand_as(x)