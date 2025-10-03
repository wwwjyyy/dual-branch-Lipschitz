# models/fusion.py
import torch
import torch.nn as nn

class SimpleLearnableFusion(nn.Module):
    """第一步：最简单的可学习标量权重融合"""
    
    def __init__(self, init_alpha=0.1):
        super().__init__()
        # 初始化权重为0.5，让两个分支平等参与
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, data_output, model_output):
        # 使用sigmoid确保权重在[0,1]范围内
        alpha = torch.sigmoid(self.alpha)
        fused_output = alpha * data_output + (1 - alpha) * model_output
        return fused_output