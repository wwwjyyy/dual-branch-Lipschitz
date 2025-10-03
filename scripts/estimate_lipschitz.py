#!/usr/bin/env python3

import torch
import argparse
import os
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（lipschitz_denoising 的父目录）
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 将项目根目录添加到 Python 路径
sys.path.insert(0, project_root)

from lipschitz_denoising.evaluators import LipschitzAnalyzer
from lipschitz_denoising.utils import logger, checkpoint
from lipschitz_denoising.models import DualBranchHybrid

def main():
    parser = argparse.ArgumentParser(description='运行Lipschitz分析')
    parser.add_argument('--config', type=str, required=True, 
                       help='Lipschitz分析配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./results/lipschitz',
                       help='结果输出目录')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log = logger.setup_logger(args.output_dir, 'lipschitz_analysis')
    
    # 加载配置
    with open(args.config, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # 更新配置中的输出目录
    config['experiment']['results_dir'] = args.output_dir
    
    # 加载模型
    model = DualBranchHybrid(config)
    model = model.to(device)
    
    # 加载预训练权重
    checkpoint.load_checkpoint(args.model_path, model, logger=log)
    
    # 创建分析器
    analyzer = LipschitzAnalyzer(args.config, model, device, logger=log)
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    # 打印关键结果
    if 'global_bounds' in results:
        bounds = results['global_bounds']
        log.info("Lipschitz分析完成:")
        for method, value in bounds.items():
            if isinstance(value, dict):
                log.info(f"{method}: {value}")
            else:
                log.info(f"{method}: {value:.4f}")
    
    log.info("Lipschitz分析程序执行完毕")

if __name__ == '__main__':
    main()