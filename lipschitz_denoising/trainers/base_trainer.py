# lipschitz_denoising/trainers/base_trainer.py

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from ..utils.logger import setup_logger, setup_tensorboard
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.data_loader import create_data_loaders
from ..functions.metrics import psnr, ssim

class BaseTrainer(ABC):
    """基础训练器抽象类，提供通用的训练框架"""
    
    def __init__(self, config, model, optimizer, scheduler=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 设置设备
        self.device = torch.device(config['experiment']['device'] 
                                  if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 设置日志和检查点
        self.log_dir = config['logging']['log_dir']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        self.logger = setup_logger(self.log_dir, name='trainer')
        self.writer = setup_tensorboard(self.log_dir, config)
        
        # 训练参数
        self.batch_size = config['training']['batch_size']
        self.num_epochs = config['training']['num_epochs']
        self.start_epoch = 0
        self.best_metric = 0
        self.current_epoch = 0
        
        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
    def train(self):
        """完整的训练流程"""
        self.logger.info("开始训练...")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 训练阶段
            train_loss = self.train_epoch()
            
            # 验证阶段
            if (epoch + 1) % self.config['validation']['interval'] == 0:
                val_metrics = self.validate()
                
                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step(val_metrics['psnr'])
                
                # 保存检查点
                is_best = val_metrics['psnr'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['psnr']
                
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # 记录到TensorBoard
                self.log_to_tensorboard(train_loss, val_metrics, epoch)
        
        self.logger.info("训练完成!")
        
    @abstractmethod
    def train_epoch(self):
        """单个epoch的训练逻辑，由子类实现"""
        pass
        
    def validate(self):
        """验证模型性能"""
        self.model.eval()
        val_loss = 0
        psnr_values = []
        ssim_values = []
        
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(tqdm(self.val_loader, desc="验证")):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # 前向传播
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                # 计算指标
                val_loss += loss.item()
                psnr_values.append(psnr(output, clean))
                ssim_values.append(ssim(output, clean))
        
        avg_loss = val_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        metrics = {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        
        self.logger.info(f"验证结果 - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, Loss: {avg_loss:.4f}")
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metrics': metrics
        }
        
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(state, filename, is_best)
        
    def log_to_tensorboard(self, train_loss, val_metrics, epoch):
        """记录训练信息到TensorBoard"""
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        self.writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
        self.writer.add_scalar('Metrics/SSIM', val_metrics['ssim'], epoch)
        
        # 记录学习率
        if self.scheduler is not None:
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.start_epoch, self.best_metric = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.logger
        )