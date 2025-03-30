import torch
import torch.nn as nn

class DistanceSuppressionModule(nn.Module):
    def __init__(self, batch_size, width, height, depth, device):
        super(DistanceSuppressionModule, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.bn = nn.BatchNorm2d(depth)

        # 定义可学习的参数a和b
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.sigma = nn.Parameter(torch.tensor(5.0))

        # 计算每个位置到中心位置的相对坐标
        grid_x, grid_y = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='ij')
        center_x = width // 2
        center_y = height // 2
        x_relative = grid_x.float() - center_x
        y_relative = grid_y.float() - center_y

        # 将相对坐标扩展到与输入数据维度匹配
        x_relative = x_relative.view(1, 1, width, height, 1).expand(batch_size, 1, width, height, depth).to(device)
        y_relative = y_relative.view(1, 1, width, height, 1).expand(batch_size, 1, width, height, depth).to(device)

        # 计算距离，这里使用欧几里得距离
        self.distance = torch.sqrt(x_relative ** 2 + y_relative ** 2)

    def forward(self, x):
        """
        :param x: 输入数据，形状为(batchsize, 1, width, height, depth)
        :return: 经过距离抑制后的结果，形状与输入数据相同
        """
        batch_size, _, width, height, depth = x.size()
        assert width == self.width, "输入张量的 width 维度与模块初始化的 width 不一致"
        assert height == self.height, "输入张量的 height 维度与模块初始化的 height 不一致"
        assert depth == self.depth, "输入张量的 depth 维度与模块初始化的 depth 不一致"

        # 根据给定的公式计算权重
        weight = torch.exp(-self.distance ** 2 / (2 * self.sigma ** 2))

        # 将权重与输入数据对应元素相乘，实现距离抑制效果
        return x * weight[:batch_size]