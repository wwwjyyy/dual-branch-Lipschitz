import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import random

# 导入已实现的噪声生成函数
from lipschitz_denoising.functions import (
    add_gaussian_noise,
    add_poisson_noise,
    add_impulse_noise,
    add_mixed_noise
)

class DenoisingDataset(Dataset):
    """
    去噪数据集类，支持多种噪声类型和混合噪声
    """
    
    def __init__(self, data_dir, noise_config, transform=None, mode='train', 
                 img_extensions=['.png', '.jpg', '.jpeg', '.bmp', '.tif']):
        """
        初始化去噪数据集
        
        参数:
            data_dir (str): 数据目录路径
            noise_config (dict): 噪声配置
            transform: 数据转换/增强
            mode (str): 数据集模式 ('train', 'val', 'test')
            img_extensions: 支持的图像扩展名
        """
        self.data_dir = data_dir
        self.noise_config = noise_config
        self.transform = transform
        self.mode = mode
        self.img_extensions = img_extensions
        
        # 收集所有图像文件
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        if not self.image_paths:
            raise ValueError(f"在目录 {data_dir} 中未找到图像文件")
        
        # 划分训练/验证/测试集
        if mode in ['train', 'val']:
            train_paths, val_paths = train_test_split(
                self.image_paths, test_size=0.2, random_state=42
            )
            self.image_paths = train_paths if mode == 'train' else val_paths
        # 测试集使用全部图像
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载干净图像
        img_path = self.image_paths[idx]
        clean_img = Image.open(img_path).convert('L')  # 转换为灰度
        
        if self.transform:
            clean_img = self.transform(clean_img)
        
        # 添加噪声
        noisy_img = self.add_noise(clean_img, self.noise_config)
        
        return noisy_img, clean_img, img_path
    
    def add_noise(self, clean_img, noise_config):
        """
        根据配置添加噪声到图像
        
        参数:
            clean_img: 干净图像张量 [C, H, W]
            noise_config: 噪声配置
            
        返回:
            noisy_img: 添加噪声后的图像张量 [C, H, W]
        """
        # 保存原始范围信息
        original_min = clean_img.min()
        original_max = clean_img.max()
        original_range = original_max - original_min
        
        # 临时将图像转换到[0,1]范围添加噪声
        if original_min < 0 or original_max > 1:
            # 如果图像不在[0,1]范围，先转换到[0,1]
            clean_img_01 = (clean_img - original_min) / original_range
        else:
            clean_img_01 = clean_img.clone()
        
        noisy_img = clean_img_01.clone()
        
        if noise_config.get('add_synthetic_noise', True):
            # 获取噪声配置
            noise_components = noise_config.get('components', [])
            
            for comp in noise_components:
                noise_type = comp.get('type', 'gaussian')
                ratio = comp.get('ratio', 1.0)
                
                if random.random() < ratio:
                    if noise_type == 'gaussian':
                        sigma = comp.get('sigma', 25)
                        noisy_img = add_gaussian_noise(noisy_img, sigma=sigma)
                    
                    elif noise_type == 'poisson':
                        lambda_val = comp.get('lambda', 30)
                        noisy_img = add_poisson_noise(noisy_img, lam=lambda_val)
                    
                    elif noise_type == 'impulse':
                        density = comp.get('density', 0.1)
                        salt_vs_pepper = comp.get('salt_vs_pepper', 0.5)
                        noisy_img = add_impulse_noise(noisy_img, density=density, salt_vs_pepper=salt_vs_pepper)
                    
                    elif noise_type == 'motion':
                        intensity = comp.get('intensity', 0.1)
                        # 运动模糊实现
                        kernel_size = int(intensity * 10) + 1
                        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
                        
                        # 添加批次维度进行卷积，然后移除
                        noisy_img_batch = noisy_img.unsqueeze(0)
                        noisy_img_batch = torch.nn.functional.conv2d(
                            noisy_img_batch, kernel, padding=kernel_size//2
                        )
                        noisy_img = noisy_img_batch.squeeze(0)
        
        # 确保像素值在[0, 1]范围内
        noisy_img = torch.clamp(noisy_img, 0, 1)
        
        # 将噪声图像转换回原始范围
        if original_min < 0 or original_max > 1:
            noisy_img = noisy_img * original_range + original_min
        
        return noisy_img

def create_data_loaders(config, dataset_config):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        config: 主配置
        dataset_config: 数据集特定配置
        
    返回:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if dataset_config.get('type') == 'gray' else transforms.Lambda(lambda x: x),
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=dataset_config.get('mean', [0.5]), 
            std=dataset_config.get('std', [0.5])
        )
    ])
    
    # 创建数据集
    data_dir = dataset_config['path']
    noise_config = dataset_config['noise']
    
    train_dataset = DenoisingDataset(
        data_dir, noise_config['train'], transform, mode='train'
    )
    
    val_dataset = DenoisingDataset(
        data_dir, noise_config['test'], transform, mode='val'
    )
    
    test_dataset = DenoisingDataset(
        data_dir, noise_config['test'], transform, mode='test'
    )
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    num_workers = config['experiment']['num_workers']
    
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

def get_dataset_stats(data_dir, img_extensions=['.png', '.jpg', '.jpeg', '.bmp', '.tif']):
    """
    计算数据集的统计信息（均值、标准差）
    
    参数:
        data_dir (str): 数据目录路径
        img_extensions: 支持的图像扩展名
        
    返回:
        mean, std: 数据集的均值和标准差
    """
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in img_extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        raise ValueError(f"在目录 {data_dir} 中未找到图像文件")
    
    # 计算均值和标准差
    pixel_sum = 0
    pixel_sq_sum = 0
    num_pixels = 0
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # 转换为灰度
        img_array = np.array(img) / 255.0  # 归一化到[0, 1]
        
        pixel_sum += np.sum(img_array)
        pixel_sq_sum += np.sum(img_array ** 2)
        num_pixels += img_array.size
    
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    
    return mean, std