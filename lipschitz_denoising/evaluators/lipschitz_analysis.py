# evaluators/lipschitz_analysis.py
import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    AUTO_LIRPA_AVAILABLE = True
except ImportError:
    AUTO_LIRPA_AVAILABLE = False
    print("警告: auto_LiRPA 不可用，将使用备用方法进行Lipschitz分析")

from ..utils import data_loader, checkpoint
from ..functions import lipschitz

class LipschitzAnalyzer:
    """Lipschitz分析器，用于估计和分析模型的Lipschitz常数"""
    
    def __init__(self, config_path: str, model, device: torch.device, logger=None):
        """
        初始化Lipschitz分析器
        
        Args:
            config_path: 配置文件路径
            model: 待分析的模型
            device: 计算设备
            logger: 日志记录器
        """
        self.config = self._load_config(config_path)
        self.model = model
        self.device = device
        self.logger = logger
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 准备测试数据
        self.test_loader = self._prepare_test_loader()
        
        if self.logger:
            self.logger.info("Lipschitz分析器初始化完成")
            if not AUTO_LIRPA_AVAILABLE and 'auto_lirpa' in self.config['evaluation']['lipschitz_analysis']['methods']:
                self.logger.warning("auto_LiRPA 不可用，相关分析将使用备用方法")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _prepare_test_loader(self) -> torch.utils.data.DataLoader:
        """准备测试数据加载器"""
        # 使用第一个测试集进行分析
        testset_cfg = self.config['dataset']['testsets'][0]
        
        dataset_config = {
            'data_dir': testset_cfg['path'],
            'noise_config': [{'type': 'gaussian', 'sigma': 25, 'ratio': 1.0}],
            'mode': 'test',
            'batch_size': self.config['resources']['batch_size']
        }
        
        _, _, test_loader = data_loader.create_data_loaders(
            self.config, dataset_config
        )
        
        if self.logger:
            self.logger.info(f"已加载测试集: {testset_cfg['name']}, 样本数: {len(test_loader.dataset)}")
        
        return test_loader
    
    def estimate_lipschitz_bounds(self) -> Dict[str, float]:
        """估计Lipschitz常数上下界"""
        results = {}
        
        if not self.config['evaluation']['lipschitz_analysis']['enabled']:
            if self.logger:
                self.logger.info("Lipschitz分析已禁用")
            return results
        
        if self.logger:
            self.logger.info("开始Lipschitz常数估计")
        
        # 应用不同的估计方法
        methods = self.config['evaluation']['lipschitz_analysis']['methods']
        
        for method_cfg in methods:
            method_type = method_cfg['type']
            
            if method_type == 'power_iteration':
                # 幂迭代法估计上界
                upper_bound = self._estimate_by_power_iteration(method_cfg)
                results['power_iteration_upper_bound'] = upper_bound
            
            elif method_type == 'jacobian' and AUTO_LIRPA_AVAILABLE:
                # Jacobian范数估计下界
                lower_bound = self._estimate_by_jacobian(method_cfg)
                results['jacobian_lower_bound'] = lower_bound
            
            elif method_type == 'auto_lirpa' and AUTO_LIRPA_AVAILABLE:
                # 使用auto_LiRPA进行精确分析
                lirpa_bounds = self._estimate_by_auto_lirpa(method_cfg)
                results['auto_lirpa_bounds'] = lirpa_bounds
            
            elif method_type == 'auto_lirpa' and not AUTO_LIRPA_AVAILABLE:
                if self.logger:
                    self.logger.warning("auto_LiRPA 不可用，跳过该方法")
        
        # 保存结果
        self._save_results(results, 'lipschitz_bounds')
        
        return results
    
    def _estimate_by_power_iteration(self, method_cfg: Dict) -> float:
        """使用幂迭代法估计Lipschitz常数上界"""
        max_iterations = method_cfg.get('max_iterations', 100)
        tolerance = method_cfg.get('tolerance', 1e-6)
        
        if self.logger:
            self.logger.info(f"使用幂迭代法估计Lipschitz上界 (迭代次数: {max_iterations})")
        
        # 使用少量样本进行估计
        sample_count = min(10, len(self.test_loader.dataset))
        upper_bounds = []
        
        for i, (clean_imgs, _) in enumerate(self.test_loader):
            if i >= sample_count:
                break
                
            clean_imgs = clean_imgs.to(self.device)
            
            # 估计该样本的Lipschitz常数上界
            upper_bound = lipschitz.estimate_lipschitz_bounds(
                self.model, 
                clean_imgs.unsqueeze(0),  # 添加batch维度
                num_batches=1,
                iterations=max_iterations,
                tolerance=tolerance
            )
            
            upper_bounds.append(upper_bound)
        
        avg_upper_bound = np.mean(upper_bounds)
        
        if self.logger:
            self.logger.info(f"幂迭代法估计的Lipschitz上界: {avg_upper_bound:.4f}")
        
        return avg_upper_bound
    
    def _estimate_by_jacobian(self, method_cfg: Dict) -> float:
        """使用Jacobian范数估计Lipschitz常数下界"""
        points = method_cfg.get('points', 1000)
        batch_size = method_cfg.get('batch_size', 1)
        
        if self.logger:
            self.logger.info(f"使用Jacobian范数估计Lipschitz下界 (采样点: {points})")
        
        # 收集样本用于Jacobian估计
        samples = []
        for i, (clean_imgs, _) in enumerate(self.test_loader):
            if len(samples) >= points:
                break
            samples.append(clean_imgs.to(self.device))
        
        # 计算Jacobian范数
        jacobian_norms = []
        
        for sample in tqdm(samples[:points], desc="Jacobian范数计算"):
            sample.requires_grad = True
            
            # 前向传播
            output, _ = self.model(sample.unsqueeze(0))
            
            # 计算Jacobian矩阵的Frobenius范数
            jacobian = torch.autograd.functional.jacobian(
                lambda x: self.model(x)[0], 
                sample.unsqueeze(0)
            )
            
            jacobian_norm = torch.norm(jacobian, p='fro').item()
            jacobian_norms.append(jacobian_norm)
        
        avg_jacobian_norm = np.mean(jacobian_norms)
        
        if self.logger:
            self.logger.info(f"Jacobian范数估计的Lipschitz下界: {avg_jacobian_norm:.4f}")
        
        return avg_jacobian_norm
    
    def _estimate_by_auto_lirpa(self, method_cfg: Dict) -> Dict[str, float]:
        """使用auto_LiRPA进行Lipschitz常数估计"""
        mode = method_cfg.get('mode', 'CROWN')
        perturbation = method_cfg.get('perturbation', 0.01)
        
        if self.logger:
            self.logger.info(f"使用auto_LiRPA ({mode}) 进行Lipschitz常数估计")
        
        # 准备有界模型
        bounded_model = BoundedModule(self.model, torch.randn(1, 1, 32, 32).to(self.device))
        
        # 使用少量样本进行估计
        sample_count = min(5, len(self.test_loader.dataset))
        bounds = {'lower': [], 'upper': []}
        
        for i, (clean_imgs, _) in enumerate(self.test_loader):
            if i >= sample_count:
                break
                
            clean_imgs = clean_imgs.to(self.device)
            
            # 定义扰动
            ptb = PerturbationLpNorm(norm=np.inf, eps=perturbation)
            bounded_input = BoundedTensor(clean_imgs, ptb)
            
            # 计算边界
            if mode == 'CROWN':
                lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')
            elif mode == 'CROWN-IBP':
                lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method='CROWN-IBP')
            else:
                lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method='IBP')
            
            # 计算Lipschitz常数
            lip_constant = (ub - lb).max().item() / (2 * perturbation)
            
            bounds['lower'].append(lb.mean().item())
            bounds['upper'].append(ub.mean().item())
        
        avg_lower = np.mean(bounds['lower'])
        avg_upper = np.mean(bounds['upper'])
        avg_lip_constant = (avg_upper - avg_lower) / (2 * perturbation)
        
        result = {
            'lower_bound': avg_lower,
            'upper_bound': avg_upper,
            'lipschitz_constant': avg_lip_constant
        }
        
        if self.logger:
            self.logger.info(f"auto_LiRPA估计的Lipschitz常数: {avg_lip_constant:.4f}")
        
        return result
    
    def analyze_layerwise_lipschitz(self) -> Dict[str, Dict[str, float]]:
        """分层分析Lipschitz常数"""
        results = {}
        
        layers_to_analyze = self.config['evaluation']['lipschitz_analysis']['layers_to_analyze']
        
        if layers_to_analyze == ['all']:
            # 分析所有层
            layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
                    layers.append((name, module))
        else:
            # 分析指定层
            layers = []
            for name, module in self.model.named_modules():
                if any(layer_name in name for layer_name in layers_to_analyze):
                    layers.append((name, module))
        
        if self.logger:
            self.logger.info(f"开始分层Lipschitz分析，共 {len(layers)} 层")
        
        for name, module in tqdm(layers, desc="分层Lipschitz分析"):
            layer_bounds = self._analyze_single_layer(module)
            results[name] = layer_bounds
        
        # 保存结果
        self._save_results(results, 'layerwise_lipschitz')
        
        return results
    
    def _analyze_single_layer(self, layer: torch.nn.Module) -> Dict[str, float]:
        """分析单层的Lipschitz常数"""
        bounds = {}
        
        if isinstance(layer, torch.nn.Conv2d):
            # 卷积层的谱范数
            weight = layer.weight
            spectral_norm = lipschitz.spectral_norm_conv2d(weight)
            bounds['spectral_norm'] = spectral_norm.item()
            
        elif isinstance(layer, torch.nn.Linear):
            # 线性层的谱范数
            weight = layer.weight
            spectral_norm = lipschitz.spectral_norm_linear(weight)
            bounds['spectral_norm'] = spectral_norm.item()
            
        elif isinstance(layer, torch.nn.ReLU):
            # ReLU的Lipschitz常数为1
            bounds['lipschitz_constant'] = 1.0
            
        return bounds
    
    def _save_results(self, results: Dict, analysis_type: str):
        """保存分析结果"""
        # 创建结果目录
        results_dir = os.path.join(
            self.config['experiment']['results_dir'],
            self.config['experiment']['name'],
            'lipschitz_analysis'
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存为JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{analysis_type}_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        if self.logger:
            self.logger.info(f"Lipschitz分析结果已保存至: {filepath}")
    
    def run_full_analysis(self):
        """执行完整的Lipschitz分析"""
        if self.logger:
            self.logger.info("开始完整Lipschitz分析")
        
        results = {}
        
        # 估计Lipschitz常数上下界
        bounds = self.estimate_lipschitz_bounds()
        results['global_bounds'] = bounds
        
        # 分层分析
        layerwise = self.analyze_layerwise_lipschitz()
        results['layerwise_analysis'] = layerwise
        
        if self.logger:
            self.logger.info("完整Lipschitz分析已完成")
        
        return results