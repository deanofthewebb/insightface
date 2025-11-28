#!/usr/bin/env python3
"""
Test script to verify LResNet100E-IR PyTorch implementation.
"""

import torch
from backbones import get_model


def test_lresnet_architecture():
    """Test that LResNet100E-IR can be instantiated and runs forward pass"""
    print("=" * 60)
    print("Testing LResNet100E-IR Architecture")
    print("=" * 60)
    
    model = get_model('lresnet100e_ir', fp16=False, num_features=512)
    model.eval()
    
    print(f"\n✓ Model instantiated successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 112, 112)
    
    print(f"\n✓ Testing forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful!")
    print(f"✓ Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {output.shape}"
    print(f"✓ Output shape matches expected (batch_size, 512)")
    
    # Print model structure summary
    print("\n" + "=" * 60)
    print("Model Structure Summary")
    print("=" * 60)
    
    for name, module in model.named_children():
        if hasattr(module, '__len__'):
            print(f"{name:15s}: {len(module)} blocks")
        else:
            print(f"{name:15s}: {module.__class__.__name__}")
    
    # Detailed layer breakdown
    print("\n" + "=" * 60)
    print("Layer Configuration")
    print("=" * 60)
    
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in layer_names:
        layer = getattr(model, layer_name)
        print(f"\n{layer_name}: {len(layer)} blocks")
        for i, block in enumerate(layer):
            print(f"  Block {i}: {block.__class__.__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return model


def compare_with_iresnet():
    """Compare LResNet100E-IR with IResNet100 parameter counts"""
    print("\n" + "=" * 60)
    print("Comparing LResNet100E-IR vs IResNet100")
    print("=" * 60)
    
    lresnet = get_model('lresnet100e_ir', fp16=False, num_features=512)
    iresnet = get_model('r100', fp16=False, num_features=512)
    
    lresnet_params = sum(p.numel() for p in lresnet.parameters())
    iresnet_params = sum(p.numel() for p in iresnet.parameters())
    
    print(f"\nLResNet100E-IR: {lresnet_params:,} parameters")
    print(f"IResNet100:     {iresnet_params:,} parameters")
    print(f"Difference:     {abs(lresnet_params - iresnet_params):,} parameters")
    
    print("\nNote: LResNet100E-IR uses bottleneck blocks (1x1→3x3→1x1)")
    print("      IResNet100 uses basic blocks (3x3→3x3)")
    print("      LResNet should have MORE parameters due to bottleneck expansion")


if __name__ == '__main__':
    model = test_lresnet_architecture()
    compare_with_iresnet()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Run convert_onnx_weights_to_pytorch.py to convert ONNX weights")
    print("2. Use converted weights in training with --backbone-pth")
    print("3. No need for --onnx-backbone anymore!")
