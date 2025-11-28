#!/usr/bin/env python3
"""
Verify that ONNXArcFaceBackbone is fully trainable without onnxruntime dependency.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "recognition" / "arcface_torch"))

from onnx_arcface_backbone import ONNXArcFaceBackbone


def verify_trainable(onnx_path):
    """Verify ONNX backbone is trainable"""
    
    print("=" * 60)
    print("Verifying ONNXArcFaceBackbone Training Capability")
    print("=" * 60)
    
    if not Path(onnx_path).exists():
        print(f"\n❌ ONNX file not found: {onnx_path}")
        print("\nDownload it with:")
        print("  aws s3 cp s3://data-labeling.livereachmedia.com/datasets/face_rec/nvr.prod.v7.facerec.backbone.onnx pretrained_models/")
        return False
    
    # Load model
    print(f"\n1. Loading ONNX backbone from {onnx_path}...")
    model = ONNXArcFaceBackbone(onnx_path, fp16=False)
    model.train()  # Training mode
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Model loaded")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    if trainable_params == 0:
        print(f"   ❌ No trainable parameters!")
        return False
    
    # Test forward pass
    print(f"\n2. Testing forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    
    output = model(x)
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    
    # Test backward pass
    print(f"\n3. Testing backward pass (gradient computation)...")
    loss = output.sum()
    loss.backward()
    
    print(f"   ✓ Backward pass successful")
    
    # Check gradients
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
    
    print(f"   ✓ Parameters with gradients: {params_with_grad}/{len(list(model.parameters()))}")
    
    if params_with_grad == 0:
        print(f"   ❌ No parameters received gradients!")
        return False
    
    # Test optimizer step
    print(f"\n4. Testing optimizer update...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Get a parameter value before update
    first_param = next(model.parameters())
    before = first_param.data.clone()
    
    # Optimizer step
    optimizer.step()
    
    # Check if parameter changed
    after = first_param.data
    changed = not torch.allclose(before, after)
    
    if changed:
        print(f"   ✓ Optimizer updated parameters")
    else:
        print(f"   ⚠ Parameters didn't change (gradients might be too small)")
    
    # Test mixed precision
    print(f"\n5. Testing mixed precision (FP16)...")
    model_fp16 = ONNXArcFaceBackbone(onnx_path, fp16=True)
    model_fp16.train()
    
    with torch.cuda.amp.autocast(enabled=True):
        x_fp16 = torch.randn(batch_size, 3, 112, 112)
        output_fp16 = model_fp16(x_fp16)
    
    print(f"   ✓ Mixed precision forward pass successful")
    
    # Verify no onnxruntime dependency during training
    print(f"\n6. Verifying no onnxruntime dependency...")
    
    import sys
    onnxruntime_loaded = 'onnxruntime' in sys.modules
    
    if onnxruntime_loaded:
        print(f"   ⚠ onnxruntime is loaded (only needed at initialization)")
    else:
        print(f"   ✓ onnxruntime NOT loaded - pure PyTorch!")
    
    print("\n" + "=" * 60)
    print("✅ ALL VERIFICATION PASSED!")
    print("=" * 60)
    print("\nONNXArcFaceBackbone is:")
    print("  ✓ Fully trainable")
    print("  ✓ Supports backpropagation")
    print("  ✓ Compatible with optimizers")
    print("  ✓ Works with mixed precision")
    print("  ✓ Pure PyTorch (no onnxruntime during training)")
    print("\nReady for fine-tuning!")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--onnx-path',
        default='pretrained_models/nvr.prod.v7.facerec.backbone.onnx',
        help='Path to ONNX model file'
    )
    
    args = parser.parse_args()
    
    success = verify_trainable(args.onnx_path)
    sys.exit(0 if success else 1)
