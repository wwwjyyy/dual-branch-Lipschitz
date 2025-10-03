#!/usr/bin/env python3
"""
混合驱动去噪模型训练脚本
基于配置文件启动混合模型训练流程
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from lipschitz_denoising.trainers.hybrid_trainer import create_hybrid_trainer
from lipschitz_denoising.utils.data_loader import create_data_loaders
from lipschitz_denoising.utils.logger import setup_logger

def load_config(config_path, dataset_name=None):
    """加载并合并配置文件，支持数据集特定配置"""
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
    """创建实验所需的目录结构"""
    dirs = {
        'root': experiment_dir,
        'logs': os.path.join(experiment_dir, 'logs'),
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'tensorboard': os.path.join(experiment_dir, 'tensorboard')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def create_comprehensive_data_loaders(config, dataset_config):
    """创建覆盖整个数据集的数据加载器"""
    # 首先尝试使用原有的数据加载器
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config, dataset_config)
        
        # 检查数据集大小
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        if hasattr(test_loader, 'dataset'):
            total_samples += len(test_loader.dataset)
        
        print(f"当前数据分割: 训练集={len(train_loader.dataset)}, 验证集={len(val_loader.dataset)}, 测试集={len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 'N/A'}")
        print(f"总样本数: {total_samples}")
        
        # 如果数据集大小不合理，重新创建数据加载器
        if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
            print("检测到空的数据集，将重新配置数据加载器...")
            raise ValueError("空数据集")
            
    except Exception as e:
        print(f"数据加载器创建失败: {e}，将使用备用方案...")
        # 备用方案：手动创建数据加载器
        from torch.utils.data import DataLoader, random_split
        from lipschitz_denoising.datasets import create_dataset
        
        # 创建完整数据集
        full_dataset = create_dataset(dataset_config, mode='train')
        
        # 计算分割比例
        train_ratio = dataset_config.get('train_ratio', 0.8)
        val_ratio = dataset_config.get('val_ratio', 0.1)
        test_ratio = dataset_config.get('test_ratio', 0.1)
        
        # 确保比例总和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio != 1.0:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # 计算各集大小
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # 随机分割数据集
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='训练混合驱动去噪模型')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称 (如: bsd68)')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='实验目录路径 (如: experiments/hybrid_v1)')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练 (检查点路径)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式 (小批量数据)')
    parser.add_argument('--use_full_dataset', action='store_true',
                       help='强制使用完整数据集')
    args = parser.parse_args()
    
    # 加载配置，包括基础配置和数据集特定配置
    config = load_config("lipschitz_denoising/configs/hybrid.yaml", args.dataset)
    
    # 设置实验目录
    dirs = setup_experiment_dirs(args.experiment_dir)
    
    # 保存当前配置到实验目录
    config_save_path = os.path.join(dirs['root'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 设置日志
    logger = setup_logger(dirs['logs'], name='train_hybrid')
    logger.info(f"开始训练实验: {args.experiment_dir}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"设备: {config['experiment'].get('device', '自动检测')}")
    
    # 设置随机种子
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    # 从配置中获取数据集配置
    dataset_config = config['dataset']
    
    # 根据参数选择数据加载器创建方式
    if args.use_full_dataset:
        logger.info("使用完整数据集模式")
        train_loader, val_loader, test_loader = create_comprehensive_data_loaders(config, dataset_config)
    else:
        train_loader, val_loader, test_loader = create_data_loaders(config, dataset_config)
    
    # 检查数据加载器
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    if hasattr(test_loader, 'dataset'):
        logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    total_samples = len(train_loader.dataset) + len(val_loader.dataset)
    if hasattr(test_loader, 'dataset'):
        total_samples += len(test_loader.dataset)
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"训练集比例: {len(train_loader.dataset)/total_samples:.2%}")
    logger.info(f"验证集比例: {len(val_loader.dataset)/total_samples:.2%}")
    if hasattr(test_loader, 'dataset'):
        logger.info(f"测试集比例: {len(test_loader.dataset)/total_samples:.2%}")

    # 获取一个样本并检查
    sample_noisy, sample_clean, _ = next(iter(train_loader))
    logger.info(f"样本噪声图像形状: {sample_noisy.shape}")
    logger.info(f"样本干净图像形状: {sample_clean.shape}")
    logger.info(f"样本噪声图像范围: [{sample_noisy.min().item():.3f}, {sample_noisy.max().item():.3f}]")
    logger.info(f"样本干净图像范围: [{sample_clean.min().item():.3f}, {sample_clean.max().item():.3f}]")

    # 调试模式: 使用小批量数据
    if args.debug:
        logger.warning("启用调试模式 - 使用小批量数据")
        # 创建小批量数据子集
        from torch.utils.data import Subset
        subset_indices = list(range(min(32, len(train_loader.dataset))))
        train_loader.dataset = Subset(train_loader.dataset, subset_indices)
        val_loader.dataset = Subset(val_loader.dataset, subset_indices[:16])
    
    # 创建训练器
    logger.info("初始化训练器...")
    trainer = create_hybrid_trainer(config, dirs['root'])
    
    # 从检查点恢复 (如果提供)
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    logger.info("开始训练...")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("训练完成!")
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise e
    
    # 保存最终模型
    final_model_path = os.path.join(dirs['checkpoints'], 'final_model.pth')
    torch.save(trainer.model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")

if __name__ == "__main__":
    main()