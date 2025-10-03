import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDataDrivenBranch(nn.Module):
    """简化的数据驱动分支 - 基础版本"""
    
    def __init__(self, channels=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1)
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        attn = self.attention(enc)
        enc = enc * attn  # 应用注意力权重
        return self.decoder(enc)