#!/usr/bin/env python3
"""
改进的混合驱动去噪模型可视化脚本
基于PSNR/SSIM提升幅度选择最佳样本进行可视化
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from lipschitz_denoising.utils.data_loader import create_data_loaders
from lipschitz_denoising.functions import psnr, ssim
from lipschitz_denoising.models import DualBranchDenoise

def load_config(config_path, dataset_name=None):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理配置继承
    if '_base_' in config:
        base_configs = config['_base_'] if isinstance(config['_base_'], list) else [config['_base_']]
        base_config = {}
        
        for base_config_path in base_configs:
            full_base_path = os.path.join(os.path.dirname(config_path), base_config_path)
            with open(full_base_path, 'r') as f:
                current_base_config = yaml.safe_load(f)
            
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
    
    # 加载数据集特定配置
    if dataset_name:
        dataset_config_path = os.path.join(os.path.dirname(config_path), 'datasets', f"{dataset_name}.yaml")
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # 合并数据集配置
            for key, value in dataset_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    return config

def load_model(model_path, config, device):
    """加载训练好的模型"""
    model = DualBranchDenoise(config)
    
    # 检查模型文件类型
    if model_path.endswith('.pth'):
        # 如果是完整的模型状态字典
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # 如果是检查点文件
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def evaluate_all_images(model, data_loader, device, config):
    """评估数据集中的所有图片，返回包含噪声图像基准值的完整结果"""
    all_results = []
    
    print("开始评估数据集中的所有图片...")
    
    with torch.no_grad():
        for batch_idx, (noisy_imgs, clean_imgs, img_names) in enumerate(data_loader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 前向传播
            denoised_imgs, _ = model(noisy_imgs)
            
            batch_size = noisy_imgs.size(0)
            
            for i in range(batch_size):
                # 获取当前样本
                noisy_img = noisy_imgs[i].unsqueeze(0)  # 添加批次维度
                clean_img = clean_imgs[i].unsqueeze(0)
                denoised_img = denoised_imgs[i].unsqueeze(0)
                
                # 计算噪声图像与干净图像的基准指标
                noisy_psnr = psnr(noisy_img, clean_img).item()
                noisy_ssim = ssim(noisy_img, clean_img).item()
                
                # 计算去噪后图像的指标
                denoised_psnr = psnr(denoised_img, clean_img).item()
                denoised_ssim = ssim(denoised_img, clean_img).item()
                
                # 计算提升幅度
                psnr_improvement = denoised_psnr - 0.3 * noisy_psnr
                ssim_improvement = denoised_ssim - 0.3 * noisy_ssim
                
                # 计算相对提升比例
                psnr_improvement_ratio = psnr_improvement / max(noisy_psnr, 1e-8)  # 避免除零
                ssim_improvement_ratio = ssim_improvement / max(noisy_ssim, 1e-8)
                
                # 存储完整结果
                img_name = img_names[i] if i < len(img_names) else f"batch_{batch_idx}_img_{i}"
                all_results.append({
                    'name': img_name,
                    'clean': clean_img.squeeze(0).cpu(),
                    'noisy': noisy_img.squeeze(0).cpu(),
                    'denoised': denoised_img.squeeze(0).cpu(),
                    
                    # 噪声图像基准指标
                    'noisy_psnr': noisy_psnr,
                    'noisy_ssim': noisy_ssim,
                    
                    # 去噪后指标
                    'denoised_psnr': denoised_psnr,
                    'denoised_ssim': denoised_ssim,
                    
                    # 提升幅度
                    'psnr_improvement': psnr_improvement,
                    'ssim_improvement': ssim_improvement,
                    
                    # 相对提升比例
                    'psnr_improvement_ratio': psnr_improvement_ratio,
                    'ssim_improvement_ratio': ssim_improvement_ratio,
                    
                    # 各种评分策略
                    'absolute_score': denoised_psnr * denoised_ssim,  # 绝对指标乘积
                    'improvement_score': psnr_improvement * ssim_improvement,  # 提升幅度乘积
                    'relative_score': psnr_improvement_ratio * ssim_improvement_ratio,  # 相对提升乘积
                    'balanced_score': (denoised_psnr + 10 * psnr_improvement) * (denoised_ssim + 10 * ssim_improvement)  # 平衡评分
                })
            
            # 进度显示
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")
    
    print(f"评估完成！共处理 {len(all_results)} 张图片")
    return all_results

def select_best_sample(all_results, selection_method='balanced'):
    """根据不同的选择方法选择最佳样本"""
    
    if selection_method == 'absolute':
        # 基于去噪后绝对指标选择
        best_sample = max(all_results, key=lambda x: x['absolute_score'])
        method_name = "绝对指标乘积 (PSNR × SSIM)"
        
    elif selection_method == 'improvement':
        # 基于提升幅度选择
        best_sample = max(all_results, key=lambda x: x['improvement_score'])
        method_name = "提升幅度乘积 (ΔPSNR × ΔSSIM)"
        
    elif selection_method == 'relative':
        # 基于相对提升比例选择
        best_sample = max(all_results, key=lambda x: x['relative_score'])
        method_name = "相对提升比例乘积"
        
    elif selection_method == 'balanced':
        # 基于平衡评分选择（综合考虑绝对值和提升幅度）
        best_sample = max(all_results, key=lambda x: x['balanced_score'])
        method_name = "平衡评分 (绝对值 + 10×提升幅度)"
        
    elif selection_method == 'psnr_improvement':
        # 仅基于PSNR提升选择
        best_sample = max(all_results, key=lambda x: x['psnr_improvement'])
        method_name = "PSNR提升幅度"
        
    elif selection_method == 'ssim_improvement':
        # 仅基于SSIM提升选择
        best_sample = max(all_results, key=lambda x: x['ssim_improvement'])
        method_name = "SSIM提升幅度"
        
    else:
        # 默认使用平衡评分
        best_sample = max(all_results, key=lambda x: x['balanced_score'])
        method_name = "平衡评分"
    
    return best_sample, method_name

def visualize_sample(sample, output_path, config, selection_method):
    """可视化样本：原图、加噪图、去噪图，显示详细指标"""
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 设置标题
    axes[0].set_title('Clean Image (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].set_title('Noisy Image (Input)', fontsize=14, fontweight='bold')
    axes[2].set_title('Denoised Image (Output)', fontsize=14, fontweight='bold')
    
    # 转换张量为numpy并调整维度顺序
    clean_img = sample['clean'].numpy().transpose(1, 2, 0)
    noisy_img = sample['noisy'].numpy().transpose(1, 2, 0)
    denoised_img = sample['denoised'].numpy().transpose(1, 2, 0)
    
    # 确保图像值在[0,1]范围内
    clean_img = np.clip(clean_img, 0, 1)
    noisy_img = np.clip(noisy_img, 0, 1)
    denoised_img = np.clip(denoised_img, 0, 1)
    
    # 显示图像
    axes[0].imshow(clean_img)
    axes[1].imshow(noisy_img)
    axes[2].imshow(denoised_img)
    
    # 移除坐标轴
    for ax in axes:
        ax.axis('off')
    
    # 创建详细的指标文本
    metrics_text = (
        f"图像名称: {sample['name']}\n\n"
        f"噪声图像基准:\n"
        f"  PSNR: {sample['noisy_psnr']:.2f} dB\n"
        f"  SSIM: {sample['noisy_ssim']:.4f}\n\n"
        f"去噪后结果:\n"
        f"  PSNR: {sample['denoised_psnr']:.2f} dB (提升: +{sample['psnr_improvement']:.2f} dB)\n"
        f"  SSIM: {sample['denoised_ssim']:.4f} (提升: +{sample['ssim_improvement']:.4f})\n\n"
        f"相对提升:\n"
        f"  PSNR: {sample['psnr_improvement_ratio']:.2%}\n"
        f"  SSIM: {sample['ssim_improvement_ratio']:.2%}\n\n"
        f"选择方法: {selection_method}"
    )
    
    # # 添加指标信息框
    # plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
    #             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")
    
    # 显示图像
    plt.show()

def create_comparison_grid(all_results, output_path, top_k=9, selection_method='balanced'):
    """创建前K个最佳样本的对比网格，基于指定的选择方法"""
    # 根据选择方法确定排序键
    if selection_method == 'absolute':
        sort_key = 'absolute_score'
    elif selection_method == 'improvement':
        sort_key = 'improvement_score'
    elif selection_method == 'relative':
        sort_key = 'relative_score'
    elif selection_method == 'balanced':
        sort_key = 'balanced_score'
    elif selection_method == 'psnr_improvement':
        sort_key = 'psnr_improvement'
    elif selection_method == 'ssim_improvement':
        sort_key = 'ssim_improvement'
    else:
        sort_key = 'balanced_score'
    
    # 按指定评分排序
    sorted_results = sorted(all_results, key=lambda x: x[sort_key], reverse=True)[:top_k]
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(top_k)))
    
    # 创建大图
    fig, axes = plt.subplots(grid_size, grid_size * 3, figsize=(5 * grid_size * 3, 5 * grid_size))
    
    if grid_size == 1:
        axes = axes.reshape(1, -1)
    
    # 遍历每个样本
    for idx, sample in enumerate(sorted_results):
        row = idx // grid_size
        col = (idx % grid_size) * 3
        
        if row >= grid_size or col >= grid_size * 3:
            break
        
        # 转换图像
        clean_img = sample['clean'].numpy().transpose(1, 2, 0)
        noisy_img = sample['noisy'].numpy().transpose(1, 2, 0)
        denoised_img = sample['denoised'].numpy().transpose(1, 2, 0)
        
        clean_img = np.clip(clean_img, 0, 1)
        noisy_img = np.clip(noisy_img, 0, 1)
        denoised_img = np.clip(denoised_img, 0, 1)
        
        # 显示三张图像
        axes[row, col].imshow(clean_img)
        axes[row, col+1].imshow(noisy_img)
        axes[row, col+2].imshow(denoised_img)
        
        # 设置标题和移除坐标轴
        title_text = (f'Clean\nPSNR: {sample["noisy_psnr"]:.2f} dB\n'
                     f'SSIM: {sample["noisy_ssim"]:.4f}')
        axes[row, col].set_title(title_text, fontsize=8)
        axes[row, col].axis('off')
        
        title_text = (f'Noisy\nPSNR: {sample["noisy_psnr"]:.2f} dB\n'
                     f'SSIM: {sample["noisy_ssim"]:.4f}')
        axes[row, col+1].set_title(title_text, fontsize=8)
        axes[row, col+1].axis('off')
        
        title_text = (f'Denoised\nPSNR: {sample["denoised_psnr"]:.2f} dB\n'
                     f'SSIM: {sample["denoised_ssim"]:.4f}\n'
                     f'ΔPSNR: +{sample["psnr_improvement"]:.2f} dB')
        axes[row, col+2].set_title(title_text, fontsize=8)
        axes[row, col+2].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(sorted_results), grid_size * grid_size):
        row = idx // grid_size
        col = (idx % grid_size) * 3
        for i in range(3):
            axes[row, col+i].axis('off')
    
    plt.suptitle(f'Top {len(sorted_results)} Denoising Results (Sorted by {selection_method})', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"对比网格已保存至: {output_path}")

def print_statistical_analysis(all_results):
    """打印详细的统计分析"""
    # 提取各种指标
    noisy_psnr = [r['noisy_psnr'] for r in all_results]
    noisy_ssim = [r['noisy_ssim'] for r in all_results]
    denoised_psnr = [r['denoised_psnr'] for r in all_results]
    denoised_ssim = [r['denoised_ssim'] for r in all_results]
    psnr_improvement = [r['psnr_improvement'] for r in all_results]
    ssim_improvement = [r['ssim_improvement'] for r in all_results]
    
    print(f"\n=== 详细统计分析 ===")
    print(f"总样本数: {len(all_results)}")
    
    print(f"\n噪声图像指标:")
    print(f"  PSNR - 平均值: {np.mean(noisy_psnr):.2f} dB, 标准差: {np.std(noisy_psnr):.2f} dB")
    print(f"  SSIM - 平均值: {np.mean(noisy_ssim):.4f}, 标准差: {np.std(noisy_ssim):.4f}")
    
    print(f"\n去噪后图像指标:")
    print(f"  PSNR - 平均值: {np.mean(denoised_psnr):.2f} dB, 标准差: {np.std(denoised_psnr):.2f} dB")
    print(f"  SSIM - 平均值: {np.mean(denoised_ssim):.4f}, 标准差: {np.std(denoised_ssim):.4f}")
    
    print(f"\n提升幅度:")
    print(f"  PSNR提升 - 平均值: {np.mean(psnr_improvement):.2f} dB, 最大值: {max(psnr_improvement):.2f} dB")
    print(f"  SSIM提升 - 平均值: {np.mean(ssim_improvement):.4f}, 最大值: {max(ssim_improvement):.4f}")
    
    print(f"\n最佳样本:")
    print(f"  最佳PSNR: {max(denoised_psnr):.2f} dB")
    print(f"  最佳SSIM: {max(denoised_ssim):.4f}")
    print(f"  最大PSNR提升: {max(psnr_improvement):.2f} dB")
    print(f"  最大SSIM提升: {max(ssim_improvement):.4f}")

def main():
    parser = argparse.ArgumentParser(description='改进的去噪模型可视化脚本')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称 (如: bsd68)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--config', type=str, default='lipschitz_denoising/configs/hybrid.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='输出目录路径')
    parser.add_argument('--selection_method', type=str, default='balanced',
                       choices=['absolute', 'improvement', 'relative', 'balanced', 
                               'psnr_improvement', 'ssim_improvement'],
                       help='选择最佳样本的方法')
    parser.add_argument('--create_grid', action='store_true',
                       help='创建前K个最佳样本的对比网格')
    parser.add_argument('--top_k', type=int, default=9,
                       help='对比网格中显示的样本数量')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = load_config(args.config, args.dataset)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据集...")
    dataset_config = config['dataset']
    _, _, test_loader = create_data_loaders(config, dataset_config)
    
    # 检查数据加载器
    if hasattr(test_loader, 'dataset'):
        print(f"测试集大小: {len(test_loader.dataset)}")
    else:
        print("警告: 无法获取测试集大小信息")
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, config, device)
    
    # 评估所有图片（包含噪声图像基准值）
    all_results = evaluate_all_images(model, test_loader, device, config)
    
    # 根据指定方法选择最佳样本
    best_sample, method_name = select_best_sample(all_results, args.selection_method)
    
    print(f"\n最佳样本信息 (选择方法: {method_name}):")
    print(f"图像名称: {best_sample['name']}")
    print(f"噪声图像 - PSNR: {best_sample['noisy_psnr']:.2f} dB, SSIM: {best_sample['noisy_ssim']:.4f}")
    print(f"去噪图像 - PSNR: {best_sample['denoised_psnr']:.2f} dB, SSIM: {best_sample['denoised_ssim']:.4f}")
    print(f"提升幅度 - PSNR: +{best_sample['psnr_improvement']:.2f} dB, SSIM: +{best_sample['ssim_improvement']:.4f}")
    print(f"相对提升 - PSNR: {best_sample['psnr_improvement_ratio']:.2%}, SSIM: {best_sample['ssim_improvement_ratio']:.2%}")
    
    # 可视化最佳样本
    best_output_path = os.path.join(args.output_dir, f'best_result_{args.selection_method}.png')
    visualize_sample(best_sample, best_output_path, config, method_name)
    
    # 可选：创建对比网格
    if args.create_grid:
        grid_output_path = os.path.join(args.output_dir, f'comparison_grid_{args.selection_method}.png')
        create_comparison_grid(all_results, grid_output_path, args.top_k, args.selection_method)
    
    # 保存详细结果到CSV文件
    csv_path = os.path.join(args.output_dir, 'detailed_results.csv')
    with open(csv_path, 'w') as f:
        f.write("Image Name,"
                "Noisy_PSNR,Noisy_SSIM,"
                "Denoised_PSNR,Denoised_SSIM,"
                "PSNR_Improvement,SSIM_Improvement,"
                "PSNR_Improvement_Ratio,SSIM_Improvement_Ratio,"
                "Absolute_Score,Improvement_Score,Relative_Score,Balanced_Score\n")
        
        for result in all_results:
            f.write(f"{result['name']},"
                   f"{result['noisy_psnr']:.4f},{result['noisy_ssim']:.4f},"
                   f"{result['denoised_psnr']:.4f},{result['denoised_ssim']:.4f},"
                   f"{result['psnr_improvement']:.4f},{result['ssim_improvement']:.4f},"
                   f"{result['psnr_improvement_ratio']:.6f},{result['ssim_improvement_ratio']:.6f},"
                   f"{result['absolute_score']:.6f},{result['improvement_score']:.6f},"
                   f"{result['relative_score']:.6f},{result['balanced_score']:.6f}\n")
    
    print(f"详细结果已保存至: {csv_path}")
    
    # 打印统计信息
    print_statistical_analysis(all_results)
    
    # 比较不同选择方法的结果
    print(f"\n=== 不同选择方法对比 ===")
    methods = ['absolute', 'improvement', 'balanced']
    for method in methods:
        sample, name = select_best_sample(all_results, method)
        print(f"{name}:")
        print(f"  图像: {sample['name']}")
        print(f"  PSNR: {sample['denoised_psnr']:.2f} dB (提升: +{sample['psnr_improvement']:.2f} dB)")
        print(f"  SSIM: {sample['denoised_ssim']:.4f} (提升: +{sample['ssim_improvement']:.4f})")

if __name__ == "__main__":
    main()