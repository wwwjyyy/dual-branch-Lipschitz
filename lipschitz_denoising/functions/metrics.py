import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from typing import Tuple, Optional

def psnr(denoised: torch.Tensor, 
        target: torch.Tensor) -> torch.Tensor:
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        denoised: 去噪后的图像 [B, C, H, W]
        target: 原始目标图像 [B, C, H, W]
        data_range: 数据范围，如果为None则自动计算
        
    Returns:
        psnr_value: PSNR值
    """
    mse = torch.mean((denoised - target) ** 2)
    
    # 对于[-1,1]范围，MAX_I = 2（因为1 - (-1) = 2）
    psnr_value = 10 * torch.log10(4.0 / mse)
    return psnr_value

def ssim(denoised: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        denoised: 去噪后的图像 [B, C, H, W]
        target: 原始目标图像 [B, C, H, W]
        data_range: 数据范围，如果为None则自动计算
        
    Returns:
        ssim_value: SSIM值
    """    
    return structural_similarity_index_measure(denoised, target, data_range=2.0)

def lipschitz_sensitivity_ratio(model: torch.nn.Module,
                               test_dataset: torch.utils.data.Dataset,
                               num_samples: int = 100) -> float:
    """
    计算Lipschitz敏感度比 (LSR, 对应式3.1)
    
    Args:
        model: 神经网络模型
        test_dataset: 测试数据集
        num_samples: 使用的样本数量
        
    Returns:
        lsr: Lipschitz敏感度比
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 随机选择样本
    indices = torch.randperm(len(test_dataset))[:num_samples]
    subset = torch.utils.data.Subset(test_dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    sensitivities = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
                
            x = x.to(device).requires_grad_(True)
            
            # 计算Jacobian范数
            jacobian_norm = torch.autograd.functional.jacobian(
                lambda x: model(x).mean(), x, create_graph=False
            )
            jacobian_norm = jacobian_norm.norm()
            
            sensitivities.append(jacobian_norm.item())
    
    sensitivities = torch.tensor(sensitivities)
    lsr = sensitivities.max() / sensitivities.min()
    
    return lsr.item()

def mixed_noise_robustness(clean_psnr: float,
                          noisy_psnr: float) -> float:
    """
    计算混合噪声鲁棒性 (MNR, 对应式3.2)
    
    Args:
        clean_psnr: 干净图像的PSNR
        noisy_psnr: 噪声图像的PSNR
        
    Returns:
        mnr: 混合噪声鲁棒性指标
    """
    return (clean_psnr - noisy_psnr) / clean_psnr

def domain_difference_decay(source_psnr: float,
                           target_psnr: float) -> float:
    """
    计算域差异衰减率 (DDD, 对应式3.3)
    
    Args:
        source_psnr: 源域PSNR
        target_psnr: 目标域PSNR
        
    Returns:
        ddd: 域差异衰减率
    """
    return (source_psnr - target_psnr) / source_psnr * 100.0