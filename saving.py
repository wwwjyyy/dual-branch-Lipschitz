import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim  # 需要安装scikit-image: pip install scikit-image

# 直接在脚本中定义 DnCNN 模型
class DnCNN(nn.Module):
    def __init__(self, depth=7, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise

# 简单的数据集类
class DenoisingDataset(Dataset):
    def __init__(self, data_dir, patch_size=50, noise_std=25.0):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.noise_std = noise_std / 255.0
        
        self.image_paths = []
        for ext in ['png', 'jpg', 'bmp', 'tif']:
            self.image_paths.extend([
                os.path.join(data_dir, fname) for fname in os.listdir(data_dir) 
                if fname.lower().endswith(ext)
            ])
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths) * 10
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx % len(self.image_paths)]
        image = Image.open(img_path).convert('L')
        
        # 随机裁剪
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            # 如果图像太小，先调整大小
            image = image.resize((max(width, self.patch_size), max(height, self.patch_size)))
            width, height = image.size
            
        i = torch.randint(0, height - self.patch_size, (1,)).item()
        j = torch.randint(0, width - self.patch_size, (1,)).item()
        patch = image.crop((j, i, j + self.patch_size, i + self.patch_size))
        
        clean = self.to_tensor(patch)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0, 1)
        
        return noisy, clean

# 计算PSNR的函数
def calculate_psnr(img1, img2):
    # 将张量转换为numpy数组
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy().squeeze()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy().squeeze()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

# 计算SSIM的函数
def calculate_ssim(img1, img2):
    # 将张量转换为numpy数组
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy().squeeze()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy().squeeze()
    
    # 确保图像数据范围在0-1之间
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # 计算SSIM
    return ssim(img1, img2, data_range=1.0)

def main():
    # 配置参数
    data_dir = './data/basic_gray/BSD68'  # 确保路径正确
    patch_size = 50
    batch_size = 16
    noise_std = 25.0
    lr = 0.001
    epochs = 50
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备数据
    dataset = DenoisingDataset(data_dir, patch_size, noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化模型
    model = DnCNN().to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算PSNR和SSIM
            with torch.no_grad():
                # 将输出限制在0-1范围内
                outputs_clamped = torch.clamp(outputs, 0, 1)
                
                # 计算当前批次的PSNR和SSIM
                batch_psnr = 0
                batch_ssim = 0
                for i in range(outputs_clamped.size(0)):
                    batch_psnr += calculate_psnr(outputs_clamped[i], clean[i])
                    batch_ssim += calculate_ssim(outputs_clamped[i], clean[i])
                
                batch_psnr /= outputs_clamped.size(0)
                batch_ssim /= outputs_clamped.size(0)
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.6f}, PSNR: {batch_psnr:.4f}, SSIM: {batch_ssim:.4f}')
        
        # 计算平均损失、PSNR和SSIM
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}, '
              f'Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            os.makedirs('./experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'./experiments/checkpoints/dncnn_epoch_{epoch+1}.pth')
    
    print('训练完成！')

if __name__ == '__main__':
    main()