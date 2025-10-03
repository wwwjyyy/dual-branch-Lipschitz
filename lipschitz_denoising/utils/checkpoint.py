import os
import torch
import glob
import re
from datetime import datetime
from lipschitz_denoising.functions import psnr, ssim  # 添加评估指标导入

def save_checkpoint(state, filename, is_best=False, best_filename='best_model.pth'):
    """
    保存训练检查点
    
    参数:
        state: 包含模型状态、优化器状态等的字典
        filename: 检查点文件名
        is_best: 是否是最佳模型
        best_filename: 最佳模型文件名
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 保存检查点
    torch.save(state, filename)
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), best_filename)
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, logger=None):
    """
    加载训练检查点
    
    参数:
        checkpoint_path: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 要加载状态的优化器（可选）
        scheduler: 要加载状态的学习率调度器（可选）
        logger: 日志记录器（可选）
        
    返回:
        start_epoch: 起始epoch
        best_metric: 最佳指标值
    """
    if not os.path.isfile(checkpoint_path):
        if logger:
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
        return 0, 0.0
    
    if logger:
        logger.info(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # 获取起始epoch和最佳指标
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    if logger:
        logger.info(f"从epoch {start_epoch}恢复训练，最佳指标: {best_metric:.4f}")
    
    return start_epoch, best_metric

def find_latest_checkpoint(checkpoint_dir, pattern='checkpoint_epoch_*.pth'):
    """
    查找最新的检查点文件
    
    参数:
        checkpoint_dir: 检查点目录
        pattern: 检查点文件模式
        
    返回:
        latest_checkpoint: 最新检查点文件路径，如果没有找到则返回None
    """
    # 获取所有匹配的检查点文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # 从文件名中提取epoch数字
    def extract_epoch(f):
        match = re.search(r'checkpoint_epoch_(\d+).pth', f)
        return int(match.group(1)) if match else -1
    
    # 按epoch数字排序
    checkpoint_files.sort(key=extract_epoch)
    
    # 返回最新的检查点
    return checkpoint_files[-1]

def evaluate_checkpoint(model, checkpoint_path, test_loader, device, logger=None):
    """
    评估检查点模型的性能
    
    参数:
        model: 模型架构
        checkpoint_path: 检查点路径
        test_loader: 测试数据加载器
        device: 计算设备
        logger: 日志记录器（可选）
        
    返回:
        metrics: 包含PSNR和SSIM的字典
    """
    # 加载检查点
    load_checkpoint(checkpoint_path, model, logger=logger)
    model.to(device)
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs, _ in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 前向传播
            denoised_imgs = model(noisy_imgs)
            
            # 计算指标
            batch_psnr = psnr(denoised_imgs, clean_imgs)
            batch_ssim = ssim(denoised_imgs, clean_imgs)
            
            total_psnr += batch_psnr * noisy_imgs.size(0)
            total_ssim += batch_ssim * noisy_imgs.size(0)
            num_samples += noisy_imgs.size(0)
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    if logger:
        logger.info(f"检查点评估结果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    return {'psnr': avg_psnr, 'ssim': avg_ssim}