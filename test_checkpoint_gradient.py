#!/usr/bin/env python
"""
单元测试：验证非重入 checkpoint 在 decoder_all 解冻策略下是否能正常反传梯度
场景：encoder frozen，decoder requires_grad=True
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SimpleDecLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x, h_input):
        return self.linear(x + h_input)

def test_checkpoint_gradient_with_frozen_encoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # 模拟编码器输出（不需要梯度）
    h_encoder = torch.randn(2, 10, 64, device=device, requires_grad=False)
    
    # 模拟解码层
    dec_layer = SimpleDecLayer(dim=64).to(device)
    
    # 冻结解码层参数（模拟"decoder_all"前的初始状态）
    for param in dec_layer.parameters():
        param.requires_grad = False
    
    # 然后解冻（模拟"decoder_all"生效）
    for param in dec_layer.parameters():
        param.requires_grad = True
    
    # 前向传播（使用非重入 checkpoint）
    h_V = torch.randn(2, 10, 64, device=device, requires_grad=True)
    h_out = checkpoint(dec_layer, h_V, h_encoder, use_reentrant=False)
    
    # 简单的 loss
    loss = h_out.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in dec_layer.named_parameters():
        if param.grad is not None:
            print(f"✅ {name} has gradient: {param.grad.norm().item():.4f}")
            has_grad = True
        else:
            print(f"❌ {name} has NO gradient!")
    
    if has_grad:
        print("\n✅ SUCCESS: Non-reentrant checkpoint preserves decoder gradients when encoder is frozen!")
        return True
    else:
        print("\n❌ FAILURE: Decoder gradients were not computed!")
        return False

if __name__ == '__main__':
    success = test_checkpoint_gradient_with_frozen_encoder()
    exit(0 if success else 1)
