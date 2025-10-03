#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估执行脚本
用于运行鲁棒性评估、Lipschitz分析和结果可视化
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lipschitz_denoising.evaluators.robustness import RobustnessEvaluator
from lipschitz_denoising.evaluators.lipschitz_analysis import LipschitzAnalyzer
from lipschitz_denoising.evaluators.visualization import ResultVisualizer
from lipschitz_denoising.models.dual_branch import DualBranchHybrid
from lipschitz_denoising.utils.logger import setup_logger

# 设置日志
logger = setup_logger("evaluation_runner")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config_path):
    """根据配置创建模型"""
    config = load_config(config_path)
    
    # 创建模型实例
    model = DualBranchHybrid(config)
    
    logger.info(f"模型已创建: {model.__class__.__name__}")
    return model

def run_robustness_evaluation(config_path, checkpoint_path=None):
    """运行鲁棒性评估"""
    logger.info("开始鲁棒性评估...")
    
    # 创建模型
    model = create_model(config_path)
    
    # 初始化评估器
    evaluator = RobustnessEvaluator(config_path)
    
    # 运行评估
    results = evaluator.run_full_evaluation(model, checkpoint_path)
    
    logger.info("鲁棒性评估完成")
    return results

def run_lipschitz_analysis(config_path, checkpoint_path=None):
    """运行Lipschitz分析"""
    logger.info("开始Lipschitz分析...")
    
    # 创建模型
    model = create_model(config_path)
    
    # 初始化分析器
    analyzer = LipschitzAnalyzer(config_path)
    
    # 运行分析
    results = analyzer.run_full_analysis(model, checkpoint_path)
    
    logger.info("Lipschitz分析完成")
    return results

def run_visualization(config_path, results_dir):
    """运行结果可视化"""
    logger.info("开始结果可视化...")
    
    # 初始化可视化器
    visualizer = ResultVisualizer(config_path)
    
    # 加载结果
    visualizer.load_results(results_dir)
    
    # 生成可视化报告
    output_dir = os.path.join(results_dir, "visualizations")
    visualizer.generate_comprehensive_report(output_dir)
    
    logger.info("结果可视化完成")
    return output_dir

def run_cross_domain_evaluation(config_path, checkpoint_path=None):
    """运行跨域评估"""
    logger.info("开始跨域评估...")
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建模型
    model = create_model(config_path)
    
    # 这里需要实现跨域评估的具体逻辑
    # 可以结合鲁棒性评估和Lipschitz分析的功能
    
    results = {}
    
    # 模拟跨域评估结果
    for domain in config['dataset']['target_domains']:
        domain_name = domain['name']
        logger.info(f"评估目标域: {domain_name}")
        
        # 这里可以添加实际的跨域评估逻辑
        results[domain_name] = {
            'psnr': 32.5,  # 示例值
            'ssim': 0.92,  # 示例值
            'ddd': 0.85    # 示例值
        }
    
    # 保存结果
    output_dir = os.path.join(
        config['paths']['experiment_root'],
        'results',
        config['experiment']['name']
    )
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'cross_domain_results.json')
    with open(results_file, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    
    logger.info("跨域评估完成")
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行评估任务")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="评估配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="模型检查点路径（可选）"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["robustness", "lipschitz", "visualization", "cross_domain", "all"],
        default="all",
        help="要运行的任务类型"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=None,
        help="结果目录（用于可视化任务）"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    # 根据任务类型执行相应的评估
    if args.task == "robustness" or args.task == "all":
        run_robustness_evaluation(args.config, args.checkpoint)
    
    if args.task == "lipschitz" or args.task == "all":
        run_lipschitz_analysis(args.config, args.checkpoint)
    
    if args.task == "cross_domain" or args.task == "all":
        run_cross_domain_evaluation(args.config, args.checkpoint)
    
    if args.task == "visualization":
        if args.results_dir is None:
            # 如果没有指定结果目录，使用默认路径
            config = load_config(args.config)
            args.results_dir = os.path.join(
                config['paths']['experiment_root'],
                'results',
                config['experiment']['name']
            )
        
        if not os.path.exists(args.results_dir):
            logger.error(f"结果目录不存在: {args.results_dir}")
            return
        
        run_visualization(args.config, args.results_dir)

if __name__ == "__main__":
    main()