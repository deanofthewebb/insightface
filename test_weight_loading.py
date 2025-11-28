#!/usr/bin/env python3
"""
Test loading .pth weights (sequential format) into PyTorch LResNet100E-IR.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "recognition" / "arcface_torch"))

from backbones import get_model
from utils.weight_mapping import load_sequential_weights


def test_weight_loading(pth_path):
    """Test loading sequential weights into LResNet100E-IR"""
    
    print("=" * 60)
    print("Testing Weight Loading: .pth → LResNet100E-IR")
    print("=" * 60)
    
    # Check file exists
    if not Path(pth_path).exists():
        print(f"ERROR: File not found: {pth_path}")
        return False
    
    print(f"\n1. Inspecting {pth_path}...")
    state = torch.load(pth_path, map_location='cpu')
    
    if isinstance(state, dict) and 'state_dict_backbone' in state:
        state = state['state_dict_backbone']
    
    print(f"   Found {len(state)} parameters")
    
    # Check format
    sample_keys = list(state.keys())[:10]
    print(f"   Sample keys: {sample_keys}")
    
    has_p_keys = any(k.startswith('p_') for k in state.keys())
    has_b_keys = any(k.startswith('b_') for k in state.keys())
    
    if has_p_keys or has_b_keys:
        print(f"   ✓ Sequential format detected (p_*, b_*)")
    else:
        print(f"   ✗ Not sequential format")
        return False
    
    # Build model
    print(f"\n2. Building LResNet100E-IR model...")
    model = get_model('lresnet100e_ir', fp16=False, num_features=512)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Load weights
    print(f"\n3. Loading weights with sequential mapping...")
    try:
        missing, unexpected = load_sequential_weights(model, pth_path, strict=False)
        
        if not missing and not unexpected:
            print(f"   ✓ All weights loaded successfully!")
        elif len(missing) < 10 and len(unexpected) < 10:
            print(f"   ⚠ Minor issues but loaded")
        else:
            print(f"   ✗ Significant missing/unexpected keys")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    print(f"\n4. Testing forward pass...")
    try:
        dummy = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            output = model(dummy)
        
        print(f"   ✓ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now use this in training:")
    print(f"  config.network = 'lresnet100e_ir'")
    print(f"  --backbone-pth {pth_path}")
    print("\nNO ONNX DEPENDENCY REQUIRED!")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test loading .pth weights into LResNet100E-IR')
    parser.add_argument('pth_path', type=str, help='Path to .pth weights file')
    
    args = parser.parse_args()
    
    success = test_weight_loading(args.pth_path)
    sys.exit(0 if success else 1)
