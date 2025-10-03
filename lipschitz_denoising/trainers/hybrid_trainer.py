# trainers/hybrid_trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from typing import Dict, List, Tuple, Optional, Callable
import time
import numpy as np
from tqdm import tqdm

from lipschitz_denoising.utils.logger import setup_logger, setup_tensorboard
from lipschitz_denoising.utils.checkpoint import save_checkpoint, load_checkpoint
from lipschitz_denoising.utils.data_loader import DenoisingDataset, create_data_loaders
from lipschitz_denoising.functions import psnr, ssim

# 导入新的模型组件
from lipschitz_denoising.models import SimpleLearnableFusion, SimpleModelDriven, DualBranchDenoise

# 导入Lipschitz估计函数
try:
    from lipschitz_denoising.assistance.Lipschitz import compute_lipschitz_upper_bound_per_layer, compute_upper_bound
    LIPSCHITZ_AVAILABLE = True
except ImportError:
    LIPSCHITZ_AVAILABLE = False
    print("警告: 无法导入Lipschitz模块，将使用备用估计方法")

class BaseTrainer:
    """基础训练器类，提供通用的训练功能"""
    
    def __init__(self, config: Dict, experiment_dir: str):
        """
        初始化基础训练器
        
        Args:
            config: 训练配置字典
            experiment_dir: 实验目录路径
        """
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device(config['experiment'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设置日志和TensorBoard
        self.logger = setup_logger(os.path.join(experiment_dir, 'logs'))
        self.tb_writer = setup_tensorboard(os.path.join(experiment_dir, 'tensorboard'), config)
        
        # 训练状态
        self.current_epoch = 0
        self.current_stage = 0
        self.best_metric = float('inf')  # 越低越好
        self.train_losses = []
        self.val_metrics = []
        
        # Lipschitz 相关配置
        self.lipschitz_enabled = config['training'].get('lipschitz_enabled', True)
        self.lipschitz_estimation_interval = config['training'].get('lipschitz_estimation_interval', 5)
        self.lipschitz_target = config['training'].get('lipschitz_target', 2.5)
        self.lipschitz_lower_bounds = []
        self.lipschitz_upper_bounds = []
        
        # 初始化模型、优化器和学习率调度器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_metrics = []
        
        # 添加以下属性用于可视化
        self.fig, self.axes = None, None
        self.live_plot_enabled = config['logging'].get('live_plot', False)
        
    def setup_model(self) -> nn.Module:
        """初始化模型"""
        raise NotImplementedError("子类必须实现setup_model方法")
    
    def setup_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """初始化优化器和学习率调度器"""
        raise NotImplementedError("子类必须实现setup_optimizer方法")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        raise NotImplementedError("子类必须实现train_epoch方法")

    def estimate_model_lipschitz(self, dataloader: DataLoader) -> Tuple[float, float]:
        if not self.lipschitz_enabled:
            return 0.0, 0.0
            
        if not LIPSCHITZ_AVAILABLE:
            self.logger.warning("Lipschitz模块不可用，使用备用估计方法")
            return self.safe_estimate_lipschitz(dataloader)
        
        try:
            # 获取样本数据
            sample_batch = next(iter(dataloader))
            if isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch
                
            input_shape = list(sample_input.shape[1:])  # 去掉batch维度
            
            # 使用改进的Lipschitz估计
            lipschitz_per_layer = compute_lipschitz_upper_bound_per_layer(
                layer=self.model,
                layer_input_shape=input_shape,  # 直接传递单个形状
                dtype=torch.float32,
                ord=2
            )
            
            # 计算整体上界
            upper_bound = compute_upper_bound(lipschitz_per_layer, dtype=torch.float32)
            
            # 改进的下界估计：使用更合理的比例
            if upper_bound > 1.0:
                lower_bound = upper_bound * 0.7  # 对于较大的上界，使用更保守的下界
            else:
                lower_bound = upper_bound * 0.5
                
            return lower_bound.item(), upper_bound.item()
            
        except Exception as e:
            self.logger.warning(f"Lipschitz估计失败: {str(e)}")
            return self.safe_estimate_lipschitz(dataloader)
        
    def safe_estimate_lipschitz(self, dataloader: DataLoader) -> Tuple[float, float]:
        """安全的Lipschitz估计方法，避免解包错误"""
        if not self.lipschitz_enabled:
            return 0.0, 0.0
            
        try:
            # 首先尝试使用模型的get_lipschitz_estimate方法
            if hasattr(self.model, 'get_lipschitz_estimate'):
                lip_estimate = self.model.get_lipschitz_estimate()
                if isinstance(lip_estimate, (tuple, list)) and len(lip_estimate) >= 2:
                    # 如果返回元组，取前两个值
                    lower, upper = lip_estimate[0], lip_estimate[1]
                elif isinstance(lip_estimate, torch.Tensor):
                    lip_value = lip_estimate.item()
                    lower, upper = lip_value * 0.8, lip_value * 1.2
                else:
                    lip_value = float(lip_estimate)
                    lower, upper = lip_value * 0.8, lip_value * 1.2
                return lower, upper
            
            # 备用方案：基于模型参数计算
            upper_bound = 1.0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() in [2, 4]:
                    try:
                        if param.dim() == 4:  # 卷积层
                            weight_mat = param.data.view(param.shape[0], -1)
                            sigma = torch.linalg.matrix_norm(weight_mat, ord=2).item()
                        else:  # 全连接层
                            sigma = torch.linalg.matrix_norm(param.data, ord=2).item()
                        upper_bound *= sigma
                    except Exception as e:
                        self.logger.warning(f"计算参数 {name} 的谱范数时出错: {e}")
                        continue
            
            # 估计下界为上界的一半
            lower_bound = upper_bound * 0.5
            
            return lower_bound, upper_bound
            
        except Exception as e:
            self.logger.warning(f"安全Lipschitz估计失败: {str(e)}")
            return 0.0, 1.0  # 返回默认值

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """在验证集上评估模型"""
        self.model.eval()
        val_psnr = 0
        val_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            for noisy_imgs, clean_imgs, _ in val_loader:
                noisy_imgs = noisy_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                # 前向传播 - 新模型接口
                outputs, _ = self.model(noisy_imgs)
                
                # 计算指标
                batch_size = noisy_imgs.size(0)
                num_samples += batch_size
                
                # 计算PSNR和SSIM
                for i in range(batch_size):
                    # 直接使用张量，不要转换为numpy
                    output_img = outputs[i]  # 形状为 [C, H, W]
                    target_img = clean_imgs[i]  # 形状为 [C, H, W]
                    
                    # 确保有批次维度
                    output_img = output_img.unsqueeze(0)  # 形状变为 [1, C, H, W]
                    target_img = target_img.unsqueeze(0)  # 形状变为 [1, C, H, W]
                    
                    val_psnr += psnr(output_img, target_img).item()
                    val_ssim += ssim(output_img, target_img).item()
        
        # 计算平均指标
        avg_psnr = val_psnr / num_samples
        avg_ssim = val_ssim / num_samples
        
        # 计算Lipschitz敏感度比（如果启用）
        lsr = 0
        lip_lower, lip_upper = 0, 0
        
        if self.lipschitz_enabled and (self.current_epoch % self.lipschitz_estimation_interval == 0):
            try:
                lip_lower, lip_upper = self.estimate_model_lipschitz(val_loader)
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    self.logger.warning("使用安全Lipschitz估计方法")
                    lip_lower, lip_upper = self.safe_estimate_lipschitz(val_loader)
                else:
                    raise e
                    
            self.lipschitz_lower_bounds.append(lip_lower)
            self.lipschitz_upper_bounds.append(lip_upper)
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lsr': lsr,
            'lip_lower': lip_lower,
            'lip_upper': lip_upper
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """完整的训练循环"""
        self.model.to(self.device)

        # 设置实时绘图
        if self.live_plot_enabled:
            self.setup_live_plot()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            self.model.train()
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_metrics = self.validate(val_loader)
            self.val_metrics.append(val_metrics)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录日志
            log_message = (f"Epoch {epoch+1}/{num_epochs} - "
                         f"Train Loss: {train_loss:.6f} - "
                         f"Val PSNR: {val_metrics['psnr']:.4f} - "
                         f"Val SSIM: {val_metrics['ssim']:.4f}")
            
            # 添加Lipschitz信息（如果可用）
            if self.lipschitz_enabled and val_metrics['lip_upper'] > 0:
                log_message += (f" - Lipschitz: [{val_metrics['lip_lower']:.4f}, {val_metrics['lip_upper']:.4f}]"
                              f" - LSR: {val_metrics['lsr']:.4f}")
            
            self.logger.info(log_message)
            
            # TensorBoard记录
            self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
            self.tb_writer.add_scalar('Metrics/val_psnr', val_metrics['psnr'], epoch)
            self.tb_writer.add_scalar('Metrics/val_ssim', val_metrics['ssim'], epoch)
            
            # 记录Lipschitz相关信息
            if self.lipschitz_enabled:
                self.tb_writer.add_scalar('Lipschitz/lower_bound', val_metrics['lip_lower'], epoch)
                self.tb_writer.add_scalar('Lipschitz/upper_bound', val_metrics['lip_upper'], epoch)
                self.tb_writer.add_scalar('Lipschitz/sensitivity_ratio', val_metrics['lsr'], epoch)
            
            # 保存检查点
            is_best = val_metrics['psnr'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['psnr']
                
            checkpoint = {
                'epoch': epoch,
                'stage': self.current_stage,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_metric,
                'train_losses': self.train_losses,
                'val_metrics': self.val_metrics,
                'lipschitz_lower_bounds': self.lipschitz_lower_bounds,
                'lipschitz_upper_bounds': self.lipschitz_upper_bounds
            }
            
            checkpoint_path = os.path.join(self.experiment_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(checkpoint, checkpoint_path, is_best)
            
            # 定期保存
            if (epoch + 1) % self.config['logging'].get('save_interval', 10) == 0:
                latest_path = os.path.join(self.experiment_dir, 'checkpoints', 'latest.pth')
                save_checkpoint(checkpoint, latest_path, False)

            # 在每个epoch结束后更新图表
            if self.live_plot_enabled:
                self.update_live_plot()
        
        # 训练结束后保持图表显示
        if self.live_plot_enabled:
            import matplotlib.pyplot as plt
            plt.ioff()  # 关闭交互模式
            plt.show()  # 保持图表显示

        self.tb_writer.close()
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点恢复训练"""
        checkpoint = load_checkpoint(
            checkpoint_path, 
            self.model, 
            self.optimizer, 
            self.scheduler if self.scheduler else None,
            self.logger
        )
        
        self.current_epoch = checkpoint['epoch']
        self.current_stage = checkpoint.get('stage', 0)
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        
        # 恢复Lipschitz边界历史
        if 'lipschitz_lower_bounds' in checkpoint:
            self.lipschitz_lower_bounds = checkpoint['lipschitz_lower_bounds']
        if 'lipschitz_upper_bounds' in checkpoint:
            self.lipschitz_upper_bounds = checkpoint['lipschitz_upper_bounds']
        
        return checkpoint
    
    def setup_live_plot(self):
        """设置实时绘图"""
        if not self.live_plot_enabled:
            return
            
        import matplotlib.pyplot as plt
        plt.ion()  # 开启交互模式
        
        # 创建四个子图，垂直排列
        self.fig, self.axes = plt.subplots(4, 1, figsize=(10, 10))
        self.fig.suptitle('Training Progress')
        
        # 初始化损失图表
        self.axes[0].set_title('Training Loss')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.loss_line, = self.axes[0].plot([], [], 'b-')
        
        # 初始化PSNR图表
        self.axes[1].set_title('Validation PSNR')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('PSNR (dB)')
        self.psnr_line, = self.axes[1].plot([], [], 'r-')
        
        # 初始化SSIM图表
        self.axes[2].set_title('Validation SSIM')
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('SSIM')
        self.ssim_line, = self.axes[2].plot([], [], 'g-')
        
        # 初始化Lipschitz边界图表
        self.axes[3].set_title('Lipschitz Bounds')
        self.axes[3].set_xlabel('Epoch')
        self.axes[3].set_ylabel('Lipschitz Constant')
        self.lip_lower_line, = self.axes[3].plot([], [], 'c-', label='Lower Bound')
        self.lip_upper_line, = self.axes[3].plot([], [], 'm-', label='Upper Bound')
        self.axes[3].legend()
        
        plt.tight_layout()

    def update_live_plot(self):
        """更新实时绘图"""
        if not self.live_plot_enabled or self.fig is None:
            return
            
        epochs = range(1, len(self.train_losses) + 1)
        
        # 更新损失图表
        self.loss_line.set_data(epochs, self.train_losses)
        self.axes[0].relim()
        self.axes[0].autoscale_view()
        
        # 更新PSNR图表
        psnr_values = [m['psnr'] for m in self.val_metrics]
        self.psnr_line.set_data(epochs, psnr_values)
        self.axes[1].relim()
        self.axes[1].autoscale_view()
        
        # 更新SSIM图表
        ssim_values = [m['ssim'] for m in self.val_metrics]
        self.ssim_line.set_data(epochs, ssim_values)
        self.axes[2].relim()
        self.axes[2].autoscale_view()
        
        # 更新Lipschitz边界图表
        if self.lipschitz_enabled and len(self.lipschitz_upper_bounds) > 0:
            # 只显示有Lipschitz估计的epoch
            lip_epochs = [(i+1) * self.lipschitz_estimation_interval 
                         for i in range(len(self.lipschitz_upper_bounds))]
            
            self.lip_lower_line.set_data(lip_epochs, self.lipschitz_lower_bounds)
            self.lip_upper_line.set_data(lip_epochs, self.lipschitz_upper_bounds)
            self.axes[3].relim()
            self.axes[3].autoscale_view()
        
        # 强制重绘
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class SimpleHybridTrainer(BaseTrainer):
    """支持分阶段训练的混合模型训练器"""
    
    def __init__(self, config: Dict, experiment_dir: str):
        super().__init__(config, experiment_dir)
        
        # 训练配置
        self.training_config = config.get('training', {})
        self.stages = self.training_config.get('stages', [])
        
        if not self.stages:
            self.logger.warning("未找到阶段配置，使用默认训练")
            self.stages = [{
                'name': 'default',
                'epochs': self.training_config.get('epochs', 300),
                'trainable_branches': ['both'],
                'freeze_other': False,
                'lr': 1e-4
            }]
        
        # 初始化模型和优化器
        self.model = self.setup_model()
        self.setup_stage_optimizer(0)  # 初始化第一个阶段的优化器

        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"模型总参数: {total_params}")
        self.logger.info(f"可训练参数: {trainable_params}")
        
        # Lipschitz配置信息
        if self.lipschitz_enabled:
            self.logger.info(f"Lipschitz估计已启用，间隔: {self.lipschitz_estimation_interval} epoch")
            self.logger.info(f"Lipschitz目标值: {self.lipschitz_target}")
        
    def setup_model(self) -> nn.Module:
        """初始化简化混合模型"""
        model = DualBranchDenoise(self.config)
        return model
    
    def setup_stage_optimizer(self, stage_idx: int) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """为特定阶段设置优化器和学习率调度器"""
        stage_config = self.stages[stage_idx]
        
        # 设置参数的可训练性
        self.set_parameter_trainability(stage_config)
        
        # 分离参数
        data_params = []
        model_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'data_branch' in name:
                    data_params.append(param)
                elif 'model_branch' in name:
                    model_params.append(param)
                else:
                    fusion_params.append(param)
        
        # 获取阶段特定的学习率
        base_lr = stage_config.get('lr', 1e-4)
        fusion_lr = stage_config.get('fusion_lr', base_lr)
        
        # 创建参数组
        param_groups = []
        
        if data_params:
            param_groups.append({
                'params': data_params, 
                'lr': base_lr, 
                'weight_decay': 1e-5,
                'name': 'data_branch'
            })
        
        if model_params:
            param_groups.append({
                'params': model_params, 
                'lr': base_lr, 
                'weight_decay': 2e-5,
                'name': 'model_branch'
            })
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params, 
                'lr': fusion_lr, 
                'weight_decay': 1e-5,
                'name': 'fusion'
            })
        
        # 创建优化器
        if param_groups:
            optimizer = Adam(param_groups)
        else:
            self.logger.warning("当前阶段没有可训练参数，创建空优化器")
            optimizer = Adam([{'params': [], 'lr': base_lr}])
        
        # 学习率调度器
        scheduler = StepLR(
            optimizer,
            step_size=self.config['scheduler'].get('step_size', 50),
            gamma=self.config['scheduler'].get('gamma', 0.8)
        )
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        return optimizer, scheduler
    
    def set_parameter_trainability(self, stage_config: Dict):
        """根据阶段配置设置参数的可训练性"""
        trainable_branches = stage_config.get('trainable_branches', ['both'])
        freeze_other = stage_config.get('freeze_other', True)
        
        self.logger.info(f"设置阶段参数可训练性: {trainable_branches}, freeze_other: {freeze_other}")
        
        for name, param in self.model.named_parameters():
            # 确定参数属于哪个分支
            if 'data_branch' in name:
                branch_type = 'data'
            elif 'model_branch' in name:
                branch_type = 'model'
            else:  # 融合模块
                branch_type = 'fusion'
            
            # 判断参数是否应该训练
            should_train = (
                branch_type in trainable_branches or 
                'both' in trainable_branches or
                (branch_type == 'fusion' and not freeze_other)
            )
            
            param.requires_grad = should_train
        
        # 记录可训练参数数量
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"可训练参数: {trainable_count}/{total_count} ({trainable_count/total_count*100:.1f}%)")
    
    def compute_loss_0(self, output: torch.Tensor, target: torch.Tensor, 
                       losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """改进的损失计算，使用范数计算稳定性"""
        recon_loss = self.criterion(output, target)
        return recon_loss
    
    def compute_loss_1(self, output: torch.Tensor, target: torch.Tensor, 
                       losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """改进的损失计算，使用范数计算稳定性"""
    
        # 基础重构损失
        recon_loss = self.criterion(output, target)
        
        # 获取各分支输出
        data_output = losses_dict.get('data_out', output)  # 数据分支输出
        model_output = losses_dict.get('model_out', output) # 模型分支输出
        
        # 分支一致性损失
        consistency_loss = self.criterion(data_output, model_output)
        
        # 基于范数的稳定性损失
        stability_loss = torch.tensor(0.0).to(output.device)
        
        if hasattr(self, 'previous_model_output'):
            # 计算当前批次和之前批次的平均2-范数
            current_norm = torch.norm(model_output, p=2, dim=(1, 2, 3)).mean()  # 形状: [batch_size] -> 标量
            previous_norm = torch.norm(self.previous_model_output, p=2, dim=(1, 2, 3)).mean()
            
            # 计算相对变化（使用平滑处理避免除零）
            norm_ratio = torch.abs(current_norm - previous_norm) / (previous_norm + 1e-8)
            stability_loss = norm_ratio
        
        self.previous_model_output = model_output.detach().clone()
        
        # 加权总损失
        if self.current_epoch < 40: total_loss = (recon_loss + 
                    0.03 * consistency_loss +  # 一致性权重
                    0.01 * stability_loss)     # 稳定性权重（减小权重，因为范数变化可能较大）
        elif self.current_epoch < 70: total_loss = (recon_loss + 
                    0.03 * consistency_loss +  # 一致性权重
                    0.02 * stability_loss)     # 稳定性权重（减小权重，因为范数变化可能较大）
        else: total_loss = (recon_loss + 
                    0.03 * consistency_loss +  # 一致性权重
                    0.04 * stability_loss)     # 稳定性权重（减小权重，因为范数变化可能较大）

        return total_loss
    
    def compute_loss_2(self, output: torch.Tensor, target: torch.Tensor, 
                     losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用统计特征计算稳定性损失"""
        
        recon_loss = self.criterion(output, target)
        data_output = losses_dict.get('data_out', output)
        model_output = losses_dict.get('model_out', output)
        
        consistency_loss = self.criterion(data_output, model_output)
        stability_loss = torch.tensor(0.0).to(output.device)
        
        if hasattr(self, 'previous_model_output'):
            # 计算批次级别的统计特征（不依赖具体维度）
            current_stats = self.compute_batch_stats(model_output)
            previous_stats = self.compute_batch_stats(self.previous_model_output)
            
            # 计算统计特征的变化
            stability_loss = self.criterion(current_stats, previous_stats)
        
        self.previous_model_output = model_output.detach().clone()
        
        total_loss = recon_loss + 0.03 * consistency_loss + 0.05 * stability_loss
        
        return total_loss

    def compute_batch_stats(self, tensor: torch.Tensor) -> torch.Tensor:
        """计算批次级别的统计特征"""
        # 计算均值、标准差等统计量
        batch_mean = tensor.mean(dim=(1, 2, 3))  # 每个样本的均值
        batch_std = tensor.std(dim=(1, 2, 3))    # 每个样本的标准差
        batch_norm = torch.norm(tensor.view(tensor.size(0), -1), p=2, dim=1)  # 每个样本的2-范数
        
        # 返回这些统计量的均值（批次级别的代表性统计）
        stats = torch.stack([
            batch_mean.mean(),    # 整个批次的平均均值
            batch_std.mean(),     # 整个批次的平均标准差  
            batch_norm.mean()     # 整个批次的平均范数
        ])
        
        return stats

    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch_idx, (noisy_imgs, clean_imgs, _) in enumerate(train_loader):
                # 数据检测 - 只在第一个batch检查
                if batch_idx == 0 and self.current_epoch == 0:
                    self.logger.info(f"数据检测 - 噪声图像范围: [{noisy_imgs.min().item():.3f}, {noisy_imgs.max().item():.3f}]")
                    self.logger.info(f"数据检测 - 干净图像范围: [{clean_imgs.min().item():.3f}, {clean_imgs.max().item():.3f}]")
                
                noisy_imgs = noisy_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                # 前向传播 - 新模型接口
                self.optimizer.zero_grad()
                outputs, losses_dict = self.model(noisy_imgs)

                # 检查模型输出范围
                if batch_idx == 0 and self.current_epoch == 0:
                    self.logger.info(f"模型输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                
                # 计算损失
                loss = self.compute_loss_0(outputs, clean_imgs, losses_dict)
                
                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                # 更新参数
                self.optimizer.step()

                # 更新进度条
                epoch_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.2e}'
                })
                pbar.update(1)
        
        return epoch_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """在验证集上评估模型"""
        self.model.eval()
        val_psnr = 0
        val_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (noisy_imgs, clean_imgs, _) in enumerate(val_loader):
                # 数据检测 - 只在第一个batch检查
                if batch_idx == 0:
                    self.logger.info(f"验证数据检测 - 噪声图像范围: [{noisy_imgs.min().item():.3f}, {noisy_imgs.max().item():.3f}]")
                    self.logger.info(f"验证数据检测 - 干净图像范围: [{clean_imgs.min().item():.3f}, {clean_imgs.max().item():.3f}]")
                
                noisy_imgs = noisy_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                # 前向传播 - 新模型接口
                outputs, _ = self.model(noisy_imgs)
                
                # 检查模型输出范围
                if batch_idx == 0:
                    self.logger.info(f"验证模型输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                
                # 计算指标
                batch_size = noisy_imgs.size(0)
                num_samples += batch_size
                
                # 计算PSNR和SSIM
                for i in range(batch_size):
                    # 直接使用张量，不要转换为numpy
                    output_img = outputs[i]  # 形状为 [C, H, W]
                    target_img = clean_imgs[i]  # 形状为 [C, H, W]
                    
                    # 确保有批次维度
                    output_img = output_img.unsqueeze(0)  # 形状变为 [1, C, H, W]
                    target_img = target_img.unsqueeze(0)  # 形状变为 [1, C, H, W]
                    
                    val_psnr += psnr(output_img, target_img).item()
                    val_ssim += ssim(output_img, target_img).item()
        
        # 计算平均指标
        avg_psnr = val_psnr / num_samples
        avg_ssim = val_ssim / num_samples
        
        # 计算Lipschitz敏感度比（如果启用）
        lsr = 0
        lip_lower, lip_upper = 0, 0
        
        if self.lipschitz_enabled and (self.current_epoch % self.lipschitz_estimation_interval == 0):
            lip_lower, lip_upper = self.estimate_model_lipschitz(val_loader)
            self.lipschitz_lower_bounds.append(lip_lower)
            self.lipschitz_upper_bounds.append(lip_upper)
            
            # 使用上界计算敏感度比
            if lip_upper > 0:
                lsr = lip_upper / self.lipschitz_target
                
            self.logger.info(f"Lipschitz边界估计 - 下界: {lip_lower:.4f}, 上界: {lip_upper:.4f}, LSR: {lsr:.4f}")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lsr': lsr,
            'lip_lower': lip_lower,
            'lip_upper': lip_upper
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """分阶段训练循环"""
        self.model.to(self.device)
        
        total_epochs = 0
        for stage_idx, stage_config in enumerate(self.stages):
            if stage_idx < self.current_stage:
                total_epochs += stage_config['epochs']
                continue
                
            self.current_stage = stage_idx
            stage_name = stage_config['name']
            stage_epochs = stage_config['epochs']
            
            self.logger.info(f"开始阶段 {stage_idx+1}/{len(self.stages)}: {stage_name}")
            self.logger.info(f"阶段配置: {stage_config}")
            
            # 设置当前阶段的优化器
            self.setup_stage_optimizer(stage_idx)
            
            # 计算当前阶段的起始和结束epoch
            start_epoch = self.current_epoch
            end_epoch = start_epoch + stage_epochs
            
            # 执行当前阶段的训练
            for epoch in range(start_epoch, end_epoch):
                self.current_epoch = epoch
                
                # 训练阶段
                self.model.train()
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # 验证阶段
                self.model.eval()
                val_metrics = self.validate(val_loader)
                self.val_metrics.append(val_metrics)
                
                # 更新学习率
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 记录日志
                log_message = (f"Stage {stage_name} | Epoch {epoch+1}/{end_epoch} - "
                             f"Train Loss: {train_loss:.6f} - "
                             f"Val PSNR: {val_metrics['psnr']:.4f} - "
                             f"Val SSIM: {val_metrics['ssim']:.4f}")
                
                # 添加Lipschitz信息（如果可用）
                if self.lipschitz_enabled and val_metrics['lip_upper'] > 0:
                    log_message += (f" - Lipschitz: [{val_metrics['lip_lower']:.4f}, {val_metrics['lip_upper']:.4f}]")
                
                self.logger.info(log_message)
                
                # TensorBoard记录
                self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
                self.tb_writer.add_scalar('Metrics/val_psnr', val_metrics['psnr'], epoch)
                self.tb_writer.add_scalar('Metrics/val_ssim', val_metrics['ssim'], epoch)
                
                # 记录Lipschitz相关信息
                if self.lipschitz_enabled:
                    self.tb_writer.add_scalar('Lipschitz/lower_bound', val_metrics['lip_lower'], epoch)
                    self.tb_writer.add_scalar('Lipschitz/upper_bound', val_metrics['lip_upper'], epoch)
                    self.tb_writer.add_scalar('Lipschitz/sensitivity_ratio', val_metrics['lsr'], epoch)
                
                # 保存检查点
                is_best = val_metrics['psnr'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['psnr']
                    
                checkpoint = {
                    'epoch': epoch,
                    'stage': stage_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_metric': self.best_metric,
                    'train_losses': self.train_losses,
                    'val_metrics': self.val_metrics,
                    'lipschitz_lower_bounds': self.lipschitz_lower_bounds,
                    'lipschitz_upper_bounds': self.lipschitz_upper_bounds
                }
                
                checkpoint_path = os.path.join(self.experiment_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(checkpoint, checkpoint_path, is_best)
                
                # 定期保存
                if (epoch + 1) % self.config['logging'].get('save_interval', 10) == 0:
                    latest_path = os.path.join(self.experiment_dir, 'checkpoints', 'latest.pth')
                    save_checkpoint(checkpoint, latest_path, False)

                # 在每个epoch结束后更新图表
                if self.live_plot_enabled:
                    self.update_live_plot()
            
            self.logger.info(f"阶段 {stage_name} 完成")
        
        self.logger.info("所有训练阶段完成")
        self.tb_writer.close()

def create_hybrid_trainer(config: Dict, experiment_dir: str) -> SimpleHybridTrainer:
    """创建混合模型训练器（与train_hybrid.py兼容的接口）"""
    return SimpleHybridTrainer(config, experiment_dir)