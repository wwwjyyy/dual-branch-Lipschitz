# evaluators/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import yaml
import os
from typing import Dict, List, Optional
import json

class ResultVisualizer:
    """结果可视化器，生成各种评估结果的可视化图表"""
    
    def __init__(self, config_path: str, results_dir: str, logger=None):
        """
        初始化结果可视化器
        
        Args:
            config_path: 配置文件路径
            results_dir: 结果目录路径
            logger: 日志记录器
        """
        self.config = self._load_config(config_path)
        self.results_dir = results_dir
        self.logger = logger
        self.figures_dir = os.path.join(results_dir, 'figures')
        
        # 创建图表目录
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        if self.logger:
            self.logger.info("结果可视化器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_results(self, result_type: str) -> Dict:
        """加载评估结果"""
        result_path = os.path.join(
            self.results_dir,
            self.config['experiment']['name'],
            result_type
        )
        
        # 查找最新的结果文件
        result_files = [f for f in os.listdir(result_path) if f.endswith('.json')]
        if not result_files:
            raise FileNotFoundError(f"未找到 {result_type} 结果文件")
        
        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(result_path, x)))
        filepath = os.path.join(result_path, latest_file)
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def plot_noise_robustness(self, save: bool = True) -> plt.Figure:
        """绘制噪声鲁棒性结果"""
        try:
            results = self.load_results('noise_robustness')
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到噪声鲁棒性结果文件")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 提取数据
        testset_names = list(results.keys())
        noise_types = ['gaussian', 'poisson', 'impulse', 'mixed']
        
        for i, noise_type in enumerate(noise_types):
            ax = axes[i]
            all_psnr = []
            all_params = []
            all_testsets = []
            
            for testset in testset_names:
                if noise_type in results[testset]:
                    noise_data = results[testset][noise_type]
                    
                    if noise_type == 'gaussian':
                        params = [float(k.split('_')[1]) for k in noise_data.keys()]
                        psnr_values = [noise_data[k]['psnr'] for k in noise_data.keys()]
                    elif noise_type == 'poisson':
                        params = [float(k.split('_')[1]) for k in noise_data.keys()]
                        psnr_values = [noise_data[k]['psnr'] for k in noise_data.keys()]
                    elif noise_type == 'impulse':
                        params = [float(k.split('_')[1]) for k in noise_data.keys()]
                        psnr_values = [noise_data[k]['psnr'] for k in noise_data.keys()]
                    else:  # mixed
                        params = [1.0]  # 混合噪声只有一个参数组
                        psnr_values = [noise_data['mixed']['psnr']]
                    
                    all_psnr.extend(psnr_values)
                    all_params.extend(params)
                    all_testsets.extend([testset] * len(params))
            
            # 创建DataFrame用于绘图
            df = pd.DataFrame({
                'Parameter': all_params,
                'PSNR': all_psnr,
                'Dataset': all_testsets
            })
            
            # 绘制曲线
            for testset in testset_names:
                testset_data = df[df['Dataset'] == testset]
                if not testset_data.empty:
                    ax.plot(testset_data['Parameter'], testset_data['PSNR'], 
                           marker='o', label=testset, linewidth=2)
            
            ax.set_xlabel('Noise Parameter' if noise_type != 'mixed' else 'Mixed Noise')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title(f'{noise_type.capitalize()} Noise Robustness')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.figures_dir, 'noise_robustness.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"噪声鲁棒性图表已保存至: {fig_path}")
        
        return fig
    
    def plot_adversarial_robustness(self, save: bool = True) -> plt.Figure:
        """绘制对抗鲁棒性结果"""
        try:
            results = self.load_results('adversarial_robustness')
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到对抗鲁棒性结果文件")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 提取FGSM攻击结果
        fgsm_data = {}
        asr_data = {}  # 攻击成功率
        
        for testset in results.keys():
            if 'fgsm' in results[testset]:
                fgsm_test = results[testset]['fgsm']
                epsilons = [float(k.split('_')[1]) for k in fgsm_test.keys()]
                psnr_values = [fgsm_test[k]['psnr'] for k in fgsm_test.keys()]
                asr_values = [fgsm_test[k]['asr'] for k in fgsm_test.keys()]
                
                fgsm_data[testset] = (epsilons, psnr_values)
                asr_data[testset] = (epsilons, asr_values)
        
        # 绘制FGSM攻击的PSNR
        ax1 = axes[0]
        for testset, (epsilons, psnr_values) in fgsm_data.items():
            ax1.plot(epsilons, psnr_values, marker='o', label=testset, linewidth=2)
        
        ax1.set_xlabel('Epsilon')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('FGSM Attack Robustness (PSNR)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制FGSM攻击的ASR
        ax2 = axes[1]
        for testset, (epsilons, asr_values) in asr_data.items():
            ax2.plot(epsilons, asr_values, marker='s', label=testset, linewidth=2)
        
        ax2.set_xlabel('Epsilon')
        ax2.set_ylabel('Attack Success Rate')
        ax2.set_title('FGSM Attack Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.figures_dir, 'adversarial_robustness.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"对抗鲁棒性图表已保存至: {fig_path}")
        
        return fig
    
    def plot_cross_domain_results(self, save: bool = True) -> plt.Figure:
        """绘制跨域评估结果"""
        try:
            results = self.load_results('cross_domain')
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到跨域评估结果文件")
            return None
        
        # 提取数据
        source_domains = list(results.keys())
        metrics = ['psnr', 'ssim', 'ddd', 'lsr']
        
        # 创建DataFrame
        data = []
        for source in source_domains:
            for target in results[source].keys():
                for metric in metrics:
                    if metric in results[source][target]:
                        value = results[source][target][metric]
                        data.append({
                            'Source': source,
                            'Target': target,
                            'Metric': metric.upper(),
                            'Value': value
                        })
        
        df = pd.DataFrame(data)
        
        # 绘制分组柱状图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = df[df['Metric'] == metric.upper()]
            
            if not metric_data.empty:
                # 创建透视表
                pivot_df = metric_data.pivot_table(
                    values='Value', index='Source', columns='Target'
                )
                
                # 绘制热力图
                sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
                ax.set_title(f'Cross-Domain {metric.upper()}')
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.figures_dir, 'cross_domain_results.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"跨域评估图表已保存至: {fig_path}")
        
        return fig
    
    def plot_lipschitz_analysis(self, save: bool = True) -> plt.Figure:
        """绘制Lipschitz分析结果"""
        try:
            results = self.load_results('lipschitz_bounds')
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到Lipschitz分析结果文件")
            return None
        
        # 提取全局边界
        global_bounds = results.get('global_bounds', {})
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制不同方法的Lipschitz估计
        methods = []
        values = []
        
        for method, value in global_bounds.items():
            if 'upper_bound' in method:
                methods.append(method)
                values.append(value)
            elif 'lipschitz_constant' in method:
                methods.append('auto_lirpa')
                values.append(value)
        
        ax1 = axes[0]
        bars = ax1.bar(methods, values, color=sns.color_palette("husl", len(methods)))
        ax1.set_ylabel('Lipschitz Constant')
        ax1.set_title('Lipschitz Constant Estimates by Different Methods')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 绘制层间Lipschitz常数分布
        try:
            layer_results = self.load_results('layerwise_lipschitz')
            layer_values = []
            
            for layer_name, layer_data in layer_results.items():
                if 'spectral_norm' in layer_data:
                    layer_values.append(layer_data['spectral_norm'])
                elif 'lipschitz_constant' in layer_data:
                    layer_values.append(layer_data['lipschitz_constant'])
            
            ax2 = axes[1]
            ax2.hist(layer_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Layer Lipschitz Constant')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Layer-wise Lipschitz Constants')
            
        except FileNotFoundError:
            if self.logger:
                self.logger.warning("未找到分层Lipschitz分析结果")
            ax2.set_title('Layer-wise Analysis Not Available')
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.figures_dir, 'lipschitz_analysis.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Lipschitz分析图表已保存至: {fig_path}")
        
        return fig
    
    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        if self.logger:
            self.logger.info("开始生成综合评估报告")
        
        # 创建所有图表
        figures = {}
        
        figures['noise_robustness'] = self.plot_noise_robustness(save=True)
        figures['adversarial_robustness'] = self.plot_adversarial_robustness(save=True)
        figures['cross_domain'] = self.plot_cross_domain_results(save=True)
        figures['lipschitz_analysis'] = self.plot_lipschitz_analysis(save=True)
        
        # 创建报告文档
        report_path = os.path.join(self.figures_dir, 'comprehensive_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# 综合评估报告\n\n")
            f.write("## 噪声鲁棒性评估\n")
            f.write("![噪声鲁棒性](noise_robustness.png)\n\n")
            
            f.write("## 对抗鲁棒性评估\n")
            f.write("![对抗鲁棒性](adversarial_robustness.png)\n\n")
            
            f.write("## 跨域评估\n")
            f.write("![跨域评估](cross_domain_results.png)\n\n")
            
            f.write("## Lipschitz分析\n")
            f.write("![Lipschitz分析](lipschitz_analysis.png)\n\n")
        
        if self.logger:
            self.logger.info(f"综合评估报告已保存至: {report_path}")
        
        return figures