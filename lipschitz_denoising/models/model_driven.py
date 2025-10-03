import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class SimpleWaveletDenoise(nn.Module):
    """简化的小波去噪模块"""
    
    def __init__(self, wavelet='haar', threshold=0.1):
        super(SimpleWaveletDenoise, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        
        # 初始化小波变换和逆变换
        self.dwt = DWTForward(J=0, wave=wavelet, mode='zero')
        self.idwt = DWTInverse(wave=wavelet, mode='zero')
    
    def soft_threshold(self, x, threshold):
        """软阈值函数"""
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))
    
    def forward(self, x):
        """
        应用小波阈值去噪
        输入: [batch, channels, height, width]
        输出: 去噪后的图像，形状与输入相同
        """
        # 执行小波变换
        yl, yh = self.dwt(x)
        
        # 对细节系数应用软阈值，并将每个尺度的方向系数堆叠成张量
        thresholded_yh = []
        for level in yh:
            thresholded_level = []
            for orientation in level:
                thresholded_orientation = self.soft_threshold(orientation, self.threshold)
                thresholded_level.append(thresholded_orientation)
            # 关键修改：将每个尺度的方向系数堆叠成张量
            thresholded_level = torch.stack(thresholded_level, dim=2)
            thresholded_yh.append(thresholded_level)
        
        # 执行逆小波变换
        denoised = self.idwt((yl, thresholded_yh))
        
        return denoised
    
class HighFrequencyExtractor(nn.Module):
    """提取图像高频分量"""
    def __init__(self):
        super().__init__()
        # 使用拉普拉斯算子近似高频
        self.laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, x):
        # 应用拉普拉斯滤波器提取边缘/高频
        weight = self.laplacian_kernel.to(x.device).repeat(x.size(1), 1, 1, 1)
        hf = F.conv2d(x, weight, padding=1, groups=x.size(1))
        return torch.abs(hf)  # 取绝对值得到边缘强度

class SimpleModelDriven(nn.Module):
    """简化的模型驱动分支"""
    
    def __init__(self, in_channels=1, tv_weight=0.01):
        super().__init__()
        self.tv_weight = tv_weight
        self.hf_extractor = HighFrequencyExtractor()
        
        # UNet风格的网络结构
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )
    
    def tv_loss(self, x):
        """仅对高频区域计算TV Loss"""
        hf_map = self.hf_extractor(x)
        # 归一化到0-1
        hf_map = (hf_map - hf_map.min()) / (hf_map.max() - hf_map.min() + 1e-8)
        
        # 计算标准TV Loss
        diff_i = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_j = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # 用高频图加权（高频区域权重高）
        # 注意：高频图需要调整尺寸以匹配梯度图
        hf_weight_i = (hf_map[:, :, :, 1:] + hf_map[:, :, :, :-1]) / 2
        hf_weight_j = (hf_map[:, :, 1:, :] + hf_map[:, :, :-1, :]) / 2
        
        weighted_tv = torch.sum(hf_weight_i * diff_i) + torch.sum(hf_weight_j * diff_j)
        return weighted_tv / x.size(0)
    
    def forward(self, x, compute_tv=True):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        
        dec2 = self.decoder2(enc2)
        dec1 = self.decoder1(torch.cat([enc1, dec2], dim=1))
        
        if compute_tv:
            tv = self.tv_loss(dec1)
            return dec1, tv
        return dec1