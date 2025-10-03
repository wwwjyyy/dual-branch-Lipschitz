# models/dual_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
# import cv2
from .data_driven import SimpleDataDrivenBranch
from .model_driven import SimpleModelDriven
from .fusion import SimpleLearnableFusion

class DualBranchDenoise(nn.Module):
    """第一步：简化双分支去噪网络"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 初始化配置参数
        self.config = config or {}
        model_config = self.config.get('model', {})
        
        # 初始化两个分支
        self.data_branch = SimpleDataDrivenBranch(
            # num_layers=model_config.get('data_layers', 5),
            # num_channels=model_config.get('data_channels', 64)
        )
        
        self.model_branch = SimpleModelDriven(
            in_channels=model_config.get('in_channels', 1)
        )
        
        # 使用最简单的融合模块
        self.fusion = SimpleLearnableFusion()
        
        # 添加可配置的参数
        self.in_channels = model_config.get('in_channels', 1)
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        前向传播
        输入: 
            x - 噪声图像 [batch, channels, height, width]
            target - 目标图像（可选，用于训练时计算损失）
        输出: 融合后的去噪结果和损失字典
        """
        data_out = self.data_branch(x)
        model_out, tv_loss = self.model_branch(x)
        fused_output = self.fusion(data_out, model_out)
        
        # 简化的损失字典（可根据需要扩展）
        losses_dict = {}
        if target is not None:
            # 主损失 - 融合输出
            losses_dict['fusion_loss'] = F.mse_loss(data_out, target)
            
            # 分支独立损失 - 用较小的权重
            losses_dict['data_branch_loss'] = 0.1 * F.mse_loss(data_out, target)
            losses_dict['model_branch_loss'] = 0.1 * F.mse_loss(model_out, target)

            losses_dict['tv_loss'] = self.model_branch.tv_weight * tv_loss
            
            # 总损失
            losses_dict['total_loss'] = (
                losses_dict['fusion_loss'] 
                + losses_dict['data_branch_loss'] 
                + losses_dict['model_branch_loss']
                + losses_dict['tv_loss']
            )

        return fused_output, losses_dict
    
    def get_fusion_weight(self) -> float:
        """获取当前融合权重（用于监控）"""
        return torch.sigmoid(self.fusion.alpha).item()
    
    def get_lipschitz_estimate(self) -> torch.Tensor:
        """简化的Lipschitz常数估计（返回固定值）"""
        return torch.tensor(1.0, device=next(self.parameters()).device)
    
    def get_branch_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各分支的输出（用于分析或可视化）"""
        with torch.no_grad():
            data_out = self.data_branch(x)
            model_out = self.model_branch(x)
            fused_out = self.fusion(data_out, model_out)
            
            return {
                'data_branch': data_out,
                'model_branch': model_out,
                'fused_output': fused_out
            }
