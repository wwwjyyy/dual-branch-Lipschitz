# evaluators/robustness.py
import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

from ..utils import data_loader, checkpoint
from ..functions import metrics, noise_generation

class RobustnessEvaluator:
    """鲁棒性评估器，执行白盒/黑盒攻击、混合噪声测试和跨域泛化实验"""
    
    def __init__(self, config_path: str, model, device: torch.device, logger=None):
        """
        初始化鲁棒性评估器
        
        Args:
            config_path: 配置文件路径
            model: 待评估的模型
            device: 计算设备
            logger: 日志记录器
        """
        self.config = self._load_config(config_path)
        self.model = model
        self.device = device
        self.logger = logger
        
        # 加载测试数据集
        self.test_loaders = self._prepare_test_loaders()
        
        # 初始化攻击方法
        self.attack_methods = self._init_attack_methods()
        
        if self.logger:
            self.logger.info("鲁棒性评估器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _prepare_test_loaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """准备测试数据加载器"""
        test_loaders = {}
        
        for testset_cfg in self.config['dataset']['testsets']:
            name = testset_cfg['name']
            path = testset_cfg['path']
            
            # 创建数据集配置
            dataset_config = {
                'data_dir': path,
                'noise_config': self.config['evaluation']['noise_robustness']['types'],
                'mode': 'test'
            }
            
            # 创建数据加载器
            _, _, test_loader = data_loader.create_data_loaders(
                self.config, dataset_config
            )
            
            test_loaders[name] = test_loader
            
            if self.logger:
                self.logger.info(f"已加载测试集: {name}, 样本数: {len(test_loader.dataset)}")
        
        return test_loaders
    
    def _init_attack_methods(self) -> Dict:
        """初始化攻击方法"""
        attack_methods = {}
        
        # 白盒攻击
        if self.config['evaluation']['adversarial_robustness']['enabled']:
            whitebox_cfg = self.config['evaluation']['adversarial_robustness']['whitebox_attacks']
            
            for attack_cfg in whitebox_cfg:
                attack_type = attack_cfg['type']
                
                if attack_type == 'fgsm':
                    # 初始化FGSM攻击
                    attack_methods['fgsm'] = self._create_fgsm_attack(attack_cfg)
                elif attack_type == 'pgd':
                    # 初始化PGD攻击
                    attack_methods['pgd'] = self._create_pgd_attack(attack_cfg)
        
        return attack_methods
    
    def _create_fgsm_attack(self, attack_cfg: Dict) -> callable:
        """创建FGSM攻击函数"""
        def fgsm_attack(model, x, y, epsilon):
            """快速梯度符号方法攻击"""
            x.requires_grad = True
            
            # 前向传播
            output, _ = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 生成对抗样本
            perturbation = epsilon * torch.sign(x.grad.data)
            x_adv = torch.clamp(x + perturbation, 0, 1)
            
            return x_adv.detach()
        
        return fgsm_attack
    
    def _create_pgd_attack(self, attack_cfg: Dict) -> callable:
        """创建PGD攻击函数"""
        def pgd_attack(model, x, y, epsilon, alpha, iterations):
            """投影梯度下降攻击"""
            x_adv = x.clone().detach()
            
            for _ in range(iterations):
                x_adv.requires_grad = True
                
                # 前向传播
                output, _ = model(x_adv)
                loss = torch.nn.functional.mse_loss(output, y)
                
                # 反向传播
                model.zero_grad()
                loss.backward()
                
                # 生成对抗扰动
                perturbation = alpha * torch.sign(x_adv.grad.data)
                x_adv = torch.clamp(x_adv + perturbation, 0, 1)
                
                # 投影到ε球内
                delta = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
            
            return x_adv
        
        return pgd_attack
    
    def evaluate_noise_robustness(self) -> Dict[str, Dict[str, float]]:
        """评估噪声鲁棒性"""
        results = {}
        
        if not self.config['evaluation']['noise_robustness']['enabled']:
            if self.logger:
                self.logger.info("噪声鲁棒性评估已禁用")
            return results
        
        if self.logger:
            self.logger.info("开始噪声鲁棒性评估")
        
        for testset_name, test_loader in self.test_loaders.items():
            results[testset_name] = {}
            
            # 测试各种噪声类型
            for noise_cfg in self.config['evaluation']['noise_robustness']['types']:
                noise_type = noise_cfg['type']
                
                if noise_type == 'gaussian':
                    # 高斯噪声测试
                    sigma_results = self._test_gaussian_noise(test_loader, noise_cfg['sigma_range'])
                    results[testset_name][f'gaussian'] = sigma_results
                
                elif noise_type == 'poisson':
                    # 泊松噪声测试
                    lambda_results = self._test_poisson_noise(test_loader, noise_cfg['lambda_range'])
                    results[testset_name][f'poisson'] = lambda_results
                
                elif noise_type == 'impulse':
                    # 脉冲噪声测试
                    density_results = self._test_impulse_noise(test_loader, noise_cfg['density_range'])
                    results[testset_name][f'impulse'] = density_results
                
                elif noise_type == 'mixed':
                    # 混合噪声测试
                    mixed_results = self._test_mixed_noise(test_loader, noise_cfg['components_range'])
                    results[testset_name][f'mixed'] = mixed_results
            
            if self.logger:
                self.logger.info(f"完成测试集 {testset_name} 的噪声鲁棒性评估")
        
        # 保存结果
        self._save_results(results, 'noise_robustness')
        
        return results
    
    def _test_gaussian_noise(self, test_loader, sigma_range: List[float]) -> Dict[str, float]:
        """测试高斯噪声鲁棒性"""
        results = {}
        
        for sigma in sigma_range:
            psnr_values = []
            ssim_values = []
            
            for clean_imgs, _ in test_loader:
                clean_imgs = clean_imgs.to(self.device)
                
                # 添加高斯噪声
                noisy_imgs = noise_generation.add_gaussian_noise(clean_imgs, sigma=sigma)
                
                # 去噪
                with torch.no_grad():
                    denoised_imgs, _ = self.model(noisy_imgs)
                
                # 计算指标
                psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
                ssim_val = metrics.ssim(denoised_imgs, clean_imgs)
                
                psnr_values.append(psnr_val.item())
                ssim_values.append(ssim_val.item())
            
            results[f'sigma_{sigma}'] = {
                'psnr': np.mean(psnr_values),
                'ssim': np.mean(ssim_values)
            }
        
        return results
    
    def _test_poisson_noise(self, test_loader, lambda_range: List[float]) -> Dict[str, float]:
        """测试泊松噪声鲁棒性"""
        results = {}
        
        for lam in lambda_range:
            psnr_values = []
            ssim_values = []
            
            for clean_imgs, _ in test_loader:
                clean_imgs = clean_imgs.to(self.device)
                
                # 添加泊松噪声
                noisy_imgs = noise_generation.add_poisson_noise(clean_imgs, lam=lam)
                
                # 去噪
                with torch.no_grad():
                    denoised_imgs, _ = self.model(noisy_imgs)
                
                # 计算指标
                psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
                ssim_val = metrics.ssim(denoised_imgs, clean_imgs)
                
                psnr_values.append(psnr_val.item())
                ssim_values.append(ssim_val.item())
            
            results[f'lambda_{lam}'] = {
                'psnr': np.mean(psnr_values),
                'ssim': np.mean(ssim_values)
            }
        
        return results
    
    def _test_impulse_noise(self, test_loader, density_range: List[float]) -> Dict[str, float]:
        """测试脉冲噪声鲁棒性"""
        results = {}
        
        for density in density_range:
            psnr_values = []
            ssim_values = []
            
            for clean_imgs, _ in test_loader:
                clean_imgs = clean_imgs.to(self.device)
                
                # 添加脉冲噪声
                noisy_imgs = noise_generation.add_impulse_noise(clean_imgs, density=density)
                
                # 去噪
                with torch.no_grad():
                    denoised_imgs, _ = self.model(noisy_imgs)
                
                # 计算指标
                psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
                ssim_val = metrics.ssim(denoised_imgs, clean_imgs)
                
                psnr_values.append(psnr_val.item())
                ssim_values.append(ssim_val.item())
            
            results[f'density_{density}'] = {
                'psnr': np.mean(psnr_values),
                'ssim': np.mean(ssim_values)
            }
        
        return results
    
    def _test_mixed_noise(self, test_loader, components_range: List[Dict]) -> Dict[str, float]:
        """测试混合噪声鲁棒性"""
        results = {}
        
        # 为每种混合配置创建噪声配置
        mixed_config = []
        for i, comp in enumerate(components_range):
            mixed_config.append({
                'type': comp['type'],
                'ratio': comp['ratio'],
                'sigma' if comp['type'] == 'gaussian' else 
                'lambda' if comp['type'] == 'poisson' else 
                'density': comp.get('sigma', comp.get('lambda', comp.get('density', 0.1)))
            })
        
        psnr_values = []
        ssim_values = []
        mnr_values = []  # 混合噪声鲁棒性
        
        for clean_imgs, _ in test_loader:
            clean_imgs = clean_imgs.to(self.device)
            
            # 添加混合噪声
            noisy_imgs = noise_generation.add_mixed_noise(clean_imgs, mixed_config)
            
            # 去噪
            with torch.no_grad():
                denoised_imgs, _ = self.model(noisy_imgs)
            
            # 计算指标
            psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
            ssim_val = metrics.ssim(denoised_imgs, clean_imgs)
            mnr_val = metrics.mixed_noise_robustness(psnr_val, psnr_val)  # 这里需要调整
            
            psnr_values.append(psnr_val.item())
            ssim_values.append(ssim_val.item())
            mnr_values.append(mnr_val.item())
        
        results['mixed'] = {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'mnr': np.mean(mnr_values)
        }
        
        return results
    
    def evaluate_adversarial_robustness(self) -> Dict[str, Dict[str, float]]:
        """评估对抗鲁棒性"""
        results = {}
        
        if not self.config['evaluation']['adversarial_robustness']['enabled']:
            if self.logger:
                self.logger.info("对抗鲁棒性评估已禁用")
            return results
        
        if self.logger:
            self.logger.info("开始对抗鲁棒性评估")
        
        for testset_name, test_loader in self.test_loaders.items():
            results[testset_name] = {}
            
            # 测试各种攻击方法
            for attack_name, attack_fn in self.attack_methods.items():
                attack_results = self._test_adversarial_attack(test_loader, attack_name, attack_fn)
                results[testset_name][attack_name] = attack_results
            
            if self.logger:
                self.logger.info(f"完成测试集 {testset_name} 的对抗鲁棒性评估")
        
        # 保存结果
        self._save_results(results, 'adversarial_robustness')
        
        return results
    
    def _test_adversarial_attack(self, test_loader, attack_name: str, attack_fn: callable) -> Dict[str, float]:
        """测试特定对抗攻击"""
        results = {}
        
        if attack_name == 'fgsm':
            # 测试不同epsilon值
            epsilon_values = self.config['evaluation']['adversarial_robustness']['whitebox_attacks'][0]['epsilon']
            
            for epsilon in epsilon_values:
                psnr_values = []
                asr_values = []  # 攻击成功率
                
                for clean_imgs, _ in tqdm(test_loader, desc=f"FGSM ε={epsilon}"):
                    clean_imgs = clean_imgs.to(self.device)
                    
                    # 生成对抗样本
                    adv_imgs = attack_fn(self.model, clean_imgs, clean_imgs, epsilon)
                    
                    # 去噪
                    with torch.no_grad():
                        denoised_imgs, _ = self.model(adv_imgs)
                    
                    # 计算指标
                    psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
                    psnr_values.append(psnr_val.item())
                    
                    # 计算攻击成功率 (ASR)
                    # 这里简化处理，使用PSNR下降程度作为攻击成功指标
                    clean_psnr = metrics.psnr(clean_imgs, clean_imgs)  # 应该是1.0或inf
                    asr = 1.0 if psnr_val < clean_psnr * 0.5 else 0.0  # 简化判断
                    asr_values.append(asr)
                
                results[f'epsilon_{epsilon}'] = {
                    'psnr': np.mean(psnr_values),
                    'asr': np.mean(asr_values)
                }
        
        elif attack_name == 'pgd':
            # 测试不同迭代次数
            iterations = self.config['evaluation']['adversarial_robustness']['whitebox_attacks'][1]['iterations']
            epsilon = self.config['evaluation']['adversarial_robustness']['whitebox_attacks'][1]['epsilon']
            alpha = self.config['evaluation']['adversarial_robustness']['whitebox_attacks'][1]['alpha']
            
            for iter_count in iterations:
                psnr_values = []
                asr_values = []
                
                for clean_imgs, _ in tqdm(test_loader, desc=f"PGD iter={iter_count}"):
                    clean_imgs = clean_imgs.to(self.device)
                    
                    # 生成对抗样本
                    adv_imgs = attack_fn(self.model, clean_imgs, clean_imgs, epsilon, alpha, iter_count)
                    
                    # 去噪
                    with torch.no_grad():
                        denoised_imgs, _ = self.model(adv_imgs)
                    
                    # 计算指标
                    psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
                    psnr_values.append(psnr_val.item())
                    
                    # 计算攻击成功率
                    clean_psnr = metrics.psnr(clean_imgs, clean_imgs)
                    asr = 1.0 if psnr_val < clean_psnr * 0.5 else 0.0
                    asr_values.append(asr)
                
                results[f'iterations_{iter_count}'] = {
                    'psnr': np.mean(psnr_values),
                    'asr': np.mean(asr_values)
                }
        
        return results
    
    def evaluate_cross_domain(self) -> Dict[str, Dict[str, float]]:
        """评估跨域泛化能力"""
        results = {}
        
        if not self.config.get('transfer_learning', {}).get('enabled', False):
            if self.logger:
                self.logger.info("跨域评估已禁用")
            return results
        
        if self.logger:
            self.logger.info("开始跨域泛化能力评估")
        
        # 获取源域和目标域配置
        source_domains = self.config['dataset']['source_domains']
        target_domains = self.config['dataset']['target_domains']
        
        # 评估每个源域到目标域的迁移
        for source in source_domains:
            results[source['name']] = {}
            
            for target in target_domains:
                # 加载目标域数据
                target_loader = self._load_domain_data(target['path'], target['type'])
                
                # 评估迁移性能
                domain_results = self._evaluate_domain_transfer(source['name'], target_loader)
                results[source['name']][target['name']] = domain_results
                
                if self.logger:
                    self.logger.info(f"完成 {source['name']} -> {target['name']} 的跨域评估")
        
        # 保存结果
        self._save_results(results, 'cross_domain')
        
        return results
    
    def _load_domain_data(self, data_path: str, data_type: str) -> torch.utils.data.DataLoader:
        """加载特定域的数据"""
        dataset_config = {
            'data_dir': data_path,
            'noise_config': self.config['evaluation']['noise_robustness']['types'],
            'mode': 'test'
        }
        
        _, _, domain_loader = data_loader.create_data_loaders(
            self.config, dataset_config
        )
        
        return domain_loader
    
    def _evaluate_domain_transfer(self, source_name: str, target_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估域迁移性能"""
        results = {}
        
        psnr_values = []
        ssim_values = []
        lsr_values = []  # Lipschitz敏感度比
        
        for clean_imgs, _ in target_loader:
            clean_imgs = clean_imgs.to(self.device)
            
            # 添加噪声（使用默认配置）
            noisy_imgs = noise_generation.add_mixed_noise(
                clean_imgs, 
                self.config['evaluation']['noise_robustness']['types'][-1]  # 使用混合噪声
            )
            
            # 去噪
            with torch.no_grad():
                denoised_imgs, _ = self.model(noisy_imgs)
            
            # 计算指标
            psnr_val = metrics.psnr(denoised_imgs, clean_imgs)
            ssim_val = metrics.ssim(denoised_imgs, clean_imgs)
            lsr_val = metrics.lipschitz_sensitivity_ratio(self.model, noisy_imgs)
            
            psnr_values.append(psnr_val.item())
            ssim_values.append(ssim_val.item())
            lsr_values.append(lsr_val.item())
        
        # 计算域差异衰减率 (DDD)
        # 这里需要源域的性能作为基准，简化处理
        source_performance = 35.0  # 假设源域PSNR为35dB
        ddd_val = metrics.domain_difference_decay(source_performance, np.mean(psnr_values))
        
        results = {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'lsr': np.mean(lsr_values),
            'ddd': ddd_val
        }
        
        return results
    
    def _save_results(self, results: Dict, eval_type: str):
        """保存评估结果"""
        # 创建结果目录
        results_dir = os.path.join(
            self.config['experiment']['results_dir'],
            self.config['experiment']['name'],
            eval_type
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存为JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{eval_type}_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        if self.logger:
            self.logger.info(f"评估结果已保存至: {filepath}")
    
    def run_full_evaluation(self):
        """执行完整的鲁棒性评估"""
        if self.logger:
            self.logger.info("开始完整鲁棒性评估")
        
        results = {}
        
        # 噪声鲁棒性评估
        noise_results = self.evaluate_noise_robustness()
        results['noise_robustness'] = noise_results
        
        # 对抗鲁棒性评估
        adv_results = self.evaluate_adversarial_robustness()
        results['adversarial_robustness'] = adv_results
        
        # 跨域评估
        cross_domain_results = self.evaluate_cross_domain()
        results['cross_domain'] = cross_domain_results
        
        if self.logger:
            self.logger.info("完整鲁棒性评估已完成")
        
        return results