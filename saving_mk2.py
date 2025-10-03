# scripts/train_dual_branch.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse

# 添加项目根目录到Python路径（保持与train_hybrid.py一致）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lipschitz_denoising.models.dual_branch import DualBranchDenoise
from lipschitz_denoising.utils.data_loader import create_data_loaders
from lipschitz_denoising.utils.logger import setup_logger
from lipschitz_denoising.utils.checkpoint import save_checkpoint, load_checkpoint
from lipschitz_denoising.functions.metrics import psnr, ssim

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练双分支去噪网络')
    parser.add_argument('--config', type=str, default='lipschitz_denoising/configs/hybrid.yaml', 
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称 (如: bsd68)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='实验目录路径 (如: experiments/dual_branch_v1)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备 (默认: 自动检测)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式 (小批量数据)')
    return parser.parse_args()

def load_config(config_path, dataset_name=None):
    """加载并合并配置文件，支持数据集特定配置（与train_hybrid.py保持一致）"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理配置继承
    if '_base_' in config:
        base_configs = config['_base_'] if isinstance(config['_base_'], list) else [config['_base_']]
        base_config = {}
        
        for base_config_path in base_configs:
            full_base_path = os.path.join(os.path.dirname(config_path), base_config_path)
            current_base_config = load_config(full_base_path)
            
            # 深度合并配置
            for key, value in current_base_config.items():
                if key != '_base_':
                    if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                        base_config[key].update(value)
                    else:
                        base_config[key] = value
        
        # 合并当前配置
        for key, value in config.items():
            if key != '_base_':
                if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                    base_config[key].update(value)
                else:
                    base_config[key] = value
                    
        config = base_config
    
    # 如果指定了数据集名称，加载数据集特定配置
    if dataset_name:
        dataset_config_path = os.path.join(os.path.dirname(config_path), 'datasets', f"{dataset_name}.yaml")
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # 合并数据集配置到主配置
            for key, value in dataset_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    return config

def setup_experiment_dirs(experiment_dir):
    """创建实验所需的目录结构（与train_hybrid.py保持一致）"""
    dirs = {
        'root': experiment_dir,
        'logs': os.path.join(experiment_dir, 'logs'),
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'tensorboard': os.path.join(experiment_dir, 'tensorboard')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def setup_tensorboard(log_dir):
    """设置TensorBoard（保持原有功能）"""
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=log_dir)

def train_epoch(model, dataloader, criterion, optimizer, device, logger, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = len(dataloader)
    
    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy, clean = noisy.to(device), clean.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        denoised = model(noisy)
        
        # 计算损失
        loss = criterion(denoised, clean)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            denoised_clamped = torch.clamp(denoised, 0, 1)
            batch_psnr = psnr(denoised_clamped, clean)
            batch_ssim = ssim(denoised_clamped, clean)
            
            total_loss += loss.item()
            total_psnr += batch_psnr
            total_ssim += batch_ssim
        
        if batch_idx % 10 == 0:
            fusion_weight = model.get_fusion_weight()
            logger.info(f'Epoch [{epoch}], Batch [{batch_idx}/{num_batches}], '
                       f'Loss: {loss.item():.6f}, PSNR: {batch_psnr:.4f}, '
                       f'SSIM: {batch_ssim:.4f}, Fusion Weight: {fusion_weight:.4f}')
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    return avg_loss, avg_psnr, avg_ssim

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            
            denoised = model(noisy)
            denoised_clamped = torch.clamp(denoised, 0, 1)
            
            loss = criterion(denoised_clamped, clean)
            batch_psnr = psnr(denoised_clamped, clean)
            batch_ssim = ssim(denoised_clamped, clean)
            
            total_loss += loss.item()
            total_psnr += batch_psnr
            total_ssim += batch_ssim
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    return avg_loss, avg_psnr, avg_ssim

def main():
    # 解析参数
    args = parse_args()
    
    # 设置实验目录（使用与train_hybrid.py相同的函数）
    dirs = setup_experiment_dirs(args.experiment_dir)
    
    # 保存当前配置到实验目录
    config = load_config(args.config, args.dataset)
    config_save_path = os.path.join(dirs['root'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 设置日志（使用与train_hybrid.py相同的函数）
    logger = setup_logger(dirs['logs'])
    writer = setup_tensorboard(dirs['tensorboard'])
    
    logger.info(f"开始训练实验: {args.experiment_dir}")
    logger.info(f"数据集: {args.dataset}")
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")
    
    # 创建数据加载器（使用与train_hybrid.py相同的函数和配置结构）
    dataset_config = config['dataset']
    train_loader, val_loader, test_loader = create_data_loaders(config, dataset_config)
    logger.info(f"数据集加载完成: {args.dataset}")
    logger.info(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    
    # 调试模式: 使用小批量数据
    if args.debug:
        logger.warning("启用调试模式 - 使用小批量数据")
        from torch.utils.data import Subset
        subset_indices = list(range(min(32, len(train_loader.dataset))))
        train_loader.dataset = Subset(train_loader.dataset, subset_indices)
        val_loader.dataset = Subset(val_loader.dataset, subset_indices[:16])
    
    # 创建模型
    model_config = config['model']
    model = DualBranchDenoise(model_config).to(device)
    logger.info("双分支去噪模型创建完成")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['lr'],
                          weight_decay=config['training']['weight_decay'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config['training']['lr_step_size'],
                                         gamma=config['training']['lr_gamma'])
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler, logger)
        logger.info(f"从检查点恢复训练: {args.resume}, 起始轮次: {start_epoch}")
    
    # 训练循环
    logger.info("开始训练...")
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            # 训练
            train_loss, train_psnr, train_ssim = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch
            )
            
            # 验证
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('PSNR/Train', train_psnr, epoch)
            writer.add_scalar('SSIM/Train', train_ssim, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('PSNR/Val', val_psnr, epoch)
            writer.add_scalar('SSIM/Val', val_ssim, epoch)
            writer.add_scalar('Fusion/Weight', model.get_fusion_weight(), epoch)
            
            # 更新学习率
            scheduler.step()
            
            # 记录日志
            logger.info(f'Epoch [{epoch}/{config["training"]["epochs"]}]: '
                       f'Train Loss: {train_loss:.6f}, PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f} | '
                       f'Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f} | '
                       f'Fusion Weight: {model.get_fusion_weight():.4f}')
            
            # 保存检查点
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'config': config
            }
            
            checkpoint_path = os.path.join(dirs['checkpoints'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(checkpoint_state, checkpoint_path, is_best)
            
            if epoch % config['training']['save_interval'] == 0:
                logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 最终测试
        logger.info("训练完成，开始最终测试...")
        test_loss, test_psnr, test_ssim = validate(model, test_loader, criterion, device)
        logger.info(f"最终测试结果 - Loss: {test_loss:.6f}, PSNR: {test_psnr:.4f}, SSIM: {test_ssim:.4f}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise e
    finally:
        writer.close()
    
    # 保存最终模型
    final_model_path = os.path.join(dirs['checkpoints'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")
    logger.info("训练完成！")

if __name__ == '__main__':
    main()