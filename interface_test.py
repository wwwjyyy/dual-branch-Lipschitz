# test_interface.py
import torch
from lipschitz_denoising.models import DualBranchDenoise

def test_interface_compatibility():
    """测试接口是否匹配"""
    
    # 创建模型实例
    model = DualBranchDenoise()
    
    # 创建测试输入
    batch_size, channels, height, width = 2, 1, 64, 64
    test_input = torch.randn(batch_size, channels, height, width)
    
    # 测试前向传播
    with torch.no_grad():
        output = model(test_input)
    
    # 验证输出形状
    assert output.shape == test_input.shape, f"输出形状不匹配: {output.shape} vs {test_input.shape}"
    
    # 验证融合权重
    fusion_weight = model.get_fusion_weight()
    assert 0 <= fusion_weight <= 1, f"融合权重超出范围: {fusion_weight}"
    
    print("✅ 接口测试通过！")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"当前融合权重: {fusion_weight:.4f}")

    # 可以添加更多的边界测试
def test_fusion_weight_stability():
    model = DualBranchDenoise()
    
    # 测试多次运行权重是否稳定
    weights = []
    for _ in range(10):
        test_input = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            _ = model(test_input)
        weights.append(model.get_fusion_weight())
    
    # 验证权重稳定性
    assert all(0.6 <= w <= 0.65 for w in weights), "权重波动过大"
    print("✅ 融合权重稳定性测试通过")

if __name__ == "__main__":
    test_interface_compatibility()
    test_fusion_weight_stability()