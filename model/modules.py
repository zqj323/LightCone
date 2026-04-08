import torch
import torch.nn as nn

# 动态阈值调节模块
class DynamicThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresh_low = 0.6
        self.thresh_high = 0.95

    def forward(self, x):
        return torch.clamp(x, self.thresh_low, self.thresh_high)

# 边界软约束模块
class SoftBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.8

    def forward(self, x):
        return x * self.lam

# 自洽校验模块
class SelfConsistency(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
