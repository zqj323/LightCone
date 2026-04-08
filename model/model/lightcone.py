import torch
import torch.nn as nn
from .modules import DynamicThreshold, SoftBoundary, SelfConsistency

class LightCone(nn.Module):
    def __init__(self):
        super().__init__()
        # 推理网络 RN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        # 逻辑分类头 LC
        self.head = nn.Linear(128, 10)
        
        # 三大创新模块
        self.dynamic_thresh = DynamicThreshold()
        self.soft_boundary = SoftBoundary()
        self.self_consistency = SelfConsistency()

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.soft_boundary(feat)
        logits = self.head(feat)
        logits = self.dynamic_thresh(logits)
        result = self.self_consistency(logits)
        return result
