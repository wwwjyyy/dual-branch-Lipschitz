import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def tv_regularization(x: torch.Tensor, 
                     weight: float = 0.05) -> torch.Tensor:
    """
    全变分正则化 (对应式2.3中的TV项)
    
    Args:
        x: 输入张量 [B, C, H, W] 或 [C, H, W]
        weight: 正则化权重
        
    Returns:
        loss: TV正则化损失
    """
    # 确保输入在合理范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 处理不同维度的输入
    if x_clamped.dim() == 3:
        # 如果是三维张量 [C, H, W]，添加批次维度
        x_clamped = x_clamped.unsqueeze(0)
    
    batch_size = x_clamped.size(0)
    h_tv = torch.abs(x_clamped[:, :, 1:, :] - x_clamped[:, :, :-1, :]).sum()
    w_tv = torch.abs(x_clamped[:, :, :, 1:] - x_clamped[:, :, :, :-1]).sum()
    tv_loss = weight * (h_tv + w_tv) / batch_size
    
    return tv_loss

def sparsity_regularization(x: torch.Tensor,
                          weight: float = 0.1,
                          p: float = 1.0) -> torch.Tensor:
    """
    稀疏性正则化 (对应式2.3中的稀疏项)
    
    Args:
        x: 输入张量 [B, C, H, W] 或 [C, H, W]
        weight: 正则化权重
        p: Lp范数的p值 (1.0为L1正则化)
        
    Returns:
        loss: 稀疏性正则化损失
    """
    # 确保输入在合理范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 处理不同维度的输入
    if x_clamped.dim() == 3:
        # 如果是三维张量 [C, H, W]，添加批次维度
        x_clamped = x_clamped.unsqueeze(0)
    
    # 确保weight是张量
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight, device=x_clamped.device)
    
    sparsity_loss = weight * torch.norm(x_clamped, p=p) / x_clamped.numel()
    return sparsity_loss

def lipschitz_regularization(model: nn.Module,
                           target_constant: float = 2.5,
                           penalty_lambda: float = 0.1,
                           method: str = "power_iteration") -> torch.Tensor:
    """
    Lipschitz正则化项 (对应式5.2)
    
    Args:
        model: 神经网络模型
        target_constant: 目标Lipschitz常数 K_target
        penalty_lambda: 惩罚系数 λ
        method: Lipschitz常数估计方法
        
    Returns:
        loss: Lipschitz正则化损失
    """
    # 获取模型参数所在的设备
    device = next(model.parameters()).device
    
    # 确保参数是张量
    if not isinstance(penalty_lambda, torch.Tensor):
        penalty_lambda = torch.tensor(penalty_lambda, device=device)
    
    # 初始化 upper_bound 为张量
    upper_bound = torch.tensor(1.0, device=device)
    
    # 估计Lipschitz常数上界
    if method == "power_iteration":
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # 简单估计每层的谱范数
                sigma = torch.svd(param).S.max()
                upper_bound = upper_bound * sigma
    else:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                if weight.dim() > 1:
                    sigma = torch.svd(weight).S.max()
                    upper_bound = upper_bound * sigma
    
    # 将 target_constant 转换为张量
    target_constant_tensor = torch.tensor(target_constant, device=device)
    
    # 计算正则化损失 (式5.2)
    lip_loss = penalty_lambda * torch.relu(upper_bound - target_constant_tensor)
    
    return lip_loss