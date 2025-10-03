import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

def _ensure_4d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """确保输入为4维张量，返回处理后的张量和是否添加了批次维度的标志"""
    if x.dim() == 3:
        # 单张图像 [C, H, W]，添加批次维度
        return x.unsqueeze(0), True
    elif x.dim() == 4:
        # 已经是批次图像 [B, C, H, W]
        return x, False
    else:
        raise ValueError(f"输入张量必须是3维或4维，但得到了 {x.dim()} 维")

def _restore_dims(x: torch.Tensor, was_3d: bool) -> torch.Tensor:
    """如果需要，恢复原始维度"""
    if was_3d:
        return x.squeeze(0)
    return x

def add_gaussian_noise(x: torch.Tensor, 
                      sigma: float = 25.0,
                      mean: float = 0.0) -> torch.Tensor:
    """
    添加高斯噪声
    
    Args:
        x: 输入图像张量 [B, C, H, W] 或 [C, H, W]
        sigma: 噪声标准差
        mean: 噪声均值
        
    Returns:
        noisy_x: 添加高斯噪声后的图像
    """
    # 确保输入在[0,1]范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 确保输入为4维
    x_4d, was_3d = _ensure_4d(x_clamped)
    
    # 生成噪声
    noise = torch.randn_like(x_4d) * sigma / 255.0 + mean
    noisy_x = x_4d + noise
    
    # 确保输出在[0,1]范围内并恢复原始维度
    noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
    return _restore_dims(noisy_x, was_3d)

def add_poisson_noise(x: torch.Tensor, 
                     lam: float = 30.0) -> torch.Tensor:
    """
    添加泊松噪声
    
    Args:
        x: 输入图像张量 [B, C, H, W] 或 [C, H, W]
        lam: 泊松分布参数
        
    Returns:
        noisy_x: 添加泊松噪声后的图像
    """
    # 确保输入在[0,1]范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 确保输入为4维
    x_4d, was_3d = _ensure_4d(x_clamped)
    
    # 将图像转换到[0, 255]范围
    x_255 = x_4d * 255.0
    
    # 确保lambda参数为正
    lam = max(lam, 1e-6)

    # 生成泊松噪声
    noisy_x_255 = torch.poisson(x_255 * lam / 255.0) * (255.0 / lam)
    
    # 转换回[0, 1]范围并恢复原始维度
    noisy_x = noisy_x_255 / 255.0
    noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
    return _restore_dims(noisy_x, was_3d)

def add_impulse_noise(x: torch.Tensor, 
                     density: float = 0.1,
                     salt_vs_pepper: float = 0.5) -> torch.Tensor:
    """
    添加脉冲噪声 (椒盐噪声)
    
    Args:
        x: 输入图像张量 [B, C, H, W] 或 [C, H, W]
        density: 噪声密度
        salt_vs_pepper: 盐噪声与椒噪声的比例
        
    Returns:
        noisy_x: 添加脉冲噪声后的图像
    """
    # 确保输入在[0,1]范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 确保输入为4维
    x_4d, was_3d = _ensure_4d(x_clamped)
    
    batch_size, channels, height, width = x_4d.shape
    noisy_x = x_4d.clone()
    
    # 生成随机掩码
    mask = torch.rand(batch_size, channels, height, width, device=x_4d.device)
    
    # 添加盐噪声
    salt_mask = mask < (density * salt_vs_pepper)
    noisy_x[salt_mask] = 1.0
    
    # 添加椒噪声
    pepper_mask = (mask >= (density * salt_vs_pepper)) & (mask < density)
    noisy_x[pepper_mask] = 0.0
    
    # 恢复原始维度
    return _restore_dims(noisy_x, was_3d)

def add_mixed_noise(x: torch.Tensor,
                   noise_config: List[Dict[str, Union[float, str]]]) -> torch.Tensor:
    """
    添加混合噪声 (对应开题报告3.4.1节)
    
    Args:
        x: 输入图像张量 [B, C, H, W] 或 [C, H, W]
        noise_config: 噪声配置列表，每个元素为包含噪声类型和参数的字典
                    例如: [{"type": "gaussian", "sigma": 25, "ratio": 0.5},
                          {"type": "poisson", "lambda": 30, "ratio": 0.3},
                          {"type": "impulse", "density": 0.1, "ratio": 0.2}]
                          
    Returns:
        noisy_x: 添加混合噪声后的图像
    """
    # 确保输入在[0,1]范围内
    x_clamped = torch.clamp(x, 0.0, 1.0)
    
    # 确保输入为4维
    x_4d, was_3d = _ensure_4d(x_clamped)
    
    noisy_x = torch.zeros_like(x_4d)
    total_ratio = sum(config.get("ratio", 1.0) for config in noise_config)
    
    # 确保比例总和为1
    if abs(total_ratio - 1.0) > 1e-6:
        # 归一化比例
        for config in noise_config:
            if "ratio" in config:
                config["ratio"] = config["ratio"] / total_ratio
    
    # 为每种噪声类型生成噪声图像
    noise_images = []
    for config in noise_config:
        noise_type = config.get("type", "gaussian")
        ratio = config.get("ratio", 1.0)
        
        if noise_type == "gaussian":
            sigma = config.get("sigma", 25.0)
            noisy_img = add_gaussian_noise(x_4d, sigma=sigma)
        elif noise_type == "poisson":
            lam = config.get("lambda", 30.0)
            noisy_img = add_poisson_noise(x_4d, lam=lam)
        elif noise_type == "impulse":
            density = config.get("density", 0.1)
            salt_vs_pepper = config.get("salt_vs_pepper", 0.5)
            noisy_img = add_impulse_noise(x_4d, density=density, salt_vs_pepper=salt_vs_pepper)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noise_images.append(noisy_img * ratio)
    
    # 混合噪声图像
    for img in noise_images:
        noisy_x += img
    
    # 确保输出在[0,1]范围内并恢复原始维度
    noisy_x = torch.clamp(noisy_x, 0.0, 1.0)
    return _restore_dims(noisy_x, was_3d)