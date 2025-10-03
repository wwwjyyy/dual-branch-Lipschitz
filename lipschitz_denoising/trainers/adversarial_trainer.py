# lipschitz_denoising/trainers/adversarial_trainer.py

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .hybrid_trainer import HybridTrainer

class AdversarialTrainer(HybridTrainer):
    """对抗训练器，实现PGD攻击和对抗训练"""
    
    def __init__(self, config, model, optimizer, scheduler=None):
        super().__init__(config, model, optimizer, scheduler)
        
        # 对抗训练配置
        self.adv_config = config.get('adversarial', {})
        self.adv_ratio = self.adv_config.get('adv_ratio', 0.5)
        
    def train_epoch(self):
        """对抗训练逻辑"""
        self.model.train()
        train_loss = 0
        
        # 确定当前训练阶段和可训练分支
        self._update_trainable_branches()
        
        # 训练循环
        for batch_idx, (noisy, clean) in enumerate(tqdm(self.train_loader, desc="对抗训练")):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 生成对抗样本
            if torch.rand(1).item() < self.adv_ratio:
                adv_noisy = self._generate_adversarial_examples(noisy, clean)
            else:
                adv_noisy = noisy
            
            # 前向传播
            self.optimizer.zero_grad()
            output, losses = self.model(adv_noisy, clean)
            
            # 计算总损失
            total_loss = self._compute_total_loss(output, clean, losses)
            
            # 反向传播和优化
            total_loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                
            self.optimizer.step()
            
            train_loss += total_loss.item()
            
            # 定期更新Lipschitz估计
            if (batch_idx + 1) % self.lipschitz_config.get('update_interval', 10) == 0:
                self._update_lipschitz_estimate()
        
        avg_loss = train_loss / len(self.train_loader)
        self.logger.info(f"对抗训练损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def _generate_adversarial_examples(self, noisy, clean):
        """生成PGD对抗样本"""
        if not self.adv_config.get('enabled', False):
            return noisy
            
        # 获取攻击参数
        attack_type = self.adv_config.get('attack_type', 'pgd')
        epsilon = self.adv_config.get('epsilon', 0.03137)
        alpha = self.adv_config.get('alpha', 0.00784)
        iterations = self.adv_config.get('iterations', 10)
        random_start = self.adv_config.get('random_start', True)
        
        # 创建对抗样本
        adv_noisy = noisy.clone().detach().requires_grad_(True)
        
        # 随机起始点
        if random_start:
            adv_noisy = adv_noisy + torch.empty_like(adv_noisy).uniform_(-epsilon, epsilon)
            adv_noisy = torch.clamp(adv_noisy, 0, 1)
        
        # PGD攻击迭代
        for _ in range(iterations):
            # 前向传播
            output, _ = self.model(adv_noisy)
            loss = self.criterion(output, clean)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 生成对抗扰动
            perturbation = alpha * adv_noisy.grad.sign()
            adv_noisy = adv_noisy.detach() + perturbation
            adv_noisy = torch.min(torch.max(adv_noisy, noisy - epsilon), noisy + epsilon)
            adv_noisy = torch.clamp(adv_noisy, 0, 1)
            adv_noisy.requires_grad_(True)
        
        return adv_noisy.detach()