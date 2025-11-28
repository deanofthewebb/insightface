#!/usr/bin/env python3
"""
Convert ONNX-style weights (.pth with p_*, b_* keys) to native PyTorch LResNet100E-IR format.

This script:
1. Loads the ONNX model to understand the architecture
2. Loads the .pth weights (which contain ONNX parameter names)
3. Maps ONNX param names (p_0, p_1, ...) to PyTorch module names
4. Saves a new .pth file compatible with PyTorch LResNet100E-IR

Usage:
    python convert_onnx_weights_to_pytorch.py \
        --onnx-model /path/to/model.onnx \
        --onnx-weights /path/to/weights.pth \
        --output /path/to/pytorch_weights.pth
"""

import argparse
import logging
from collections import OrderedDict

import torch
import onnx
from onnx import numpy_helper

from backbones import get_model


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_onnx_param_names(onnx_path):
    """
    Load ONNX model and extract all initializer names in order.
    
    Returns:
        List of (name, shape) tuples for all ONNX initializers
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    
    params = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        params.append((init.name, tuple(arr.shape)))
    
    logging.info(f"Loaded ONNX model with {len(params)} initializers")
    return params


def build_pytorch_param_names(pytorch_model):
    """
    Extract ordered parameter names from PyTorch model.
    
    Returns:
        List of (name, shape) tuples for all PyTorch parameters
    """
    params = []
    for name, param in pytorch_model.named_parameters():
        params.append((name, tuple(param.shape)))
    
    # Also include buffers (like running_mean, running_var)
    for name, buffer in pytorch_model.named_buffers():
        params.append((name, tuple(buffer.shape)))
    
    logging.info(f"PyTorch model has {len(params)} parameters + buffers")
    return params


def create_name_mapping(onnx_params, pytorch_params):
    """
    Create mapping from ONNX parameter names to PyTorch parameter names.
    
    Strategy:
    1. Match by shape (ONNX and PyTorch params should have same shapes in same order)
    2. Handle special cases (biases, BN stats, etc.)
    
    Returns:
        Dict[str, str]: ONNX name -> PyTorch name
    """
    mapping = {}
    
    # Group by shape to help with matching
    onnx_by_shape = {}
    for name, shape in onnx_params:
        if shape not in onnx_by_shape:
            onnx_by_shape[shape] = []
        onnx_by_shape[shape].append(name)
    
    pytorch_by_shape = {}
    for name, shape in pytorch_params:
        if shape not in pytorch_by_shape:
            pytorch_by_shape[shape] = []
        pytorch_by_shape[shape].append(name)
    
    logging.info("\n" + "=" * 60)
    logging.info("Shape-based parameter matching:")
    logging.info("=" * 60)
    
    for shape in sorted(onnx_by_shape.keys()):
        onnx_names = onnx_by_shape.get(shape, [])
        pytorch_names = pytorch_by_shape.get(shape, [])
        
        if len(onnx_names) != len(pytorch_names):
            logging.warning(
                f"Shape {shape}: {len(onnx_names)} ONNX params vs "
                f"{len(pytorch_names)} PyTorch params"
            )
            continue
        
        for onnx_name, pytorch_name in zip(onnx_names, pytorch_names):
            mapping[onnx_name] = pytorch_name
            logging.debug(f"  {onnx_name:20s} -> {pytorch_name}")
    
    logging.info(f"\nMapped {len(mapping)} parameters")
    return mapping


def convert_weights(onnx_weights_path, name_mapping):
    """
    Load ONNX-style .pth weights and convert to PyTorch format using the mapping.
    
    Returns:
        OrderedDict: PyTorch-compatible state dict
    """
    logging.info(f"\nLoading ONNX weights from: {onnx_weights_path}")
    onnx_state = torch.load(onnx_weights_path, map_location='cpu')
    
    # Handle both raw state_dict and checkpoint formats
    if isinstance(onnx_state, dict) and 'state_dict_backbone' in onnx_state:
        onnx_state = onnx_state['state_dict_backbone']
    
    logging.info(f"Loaded {len(onnx_state)} weights from .pth")
    
    pytorch_state = OrderedDict()
    unmapped = []
    
    for onnx_name, tensor in onnx_state.items():
        if onnx_name in name_mapping:
            pytorch_name = name_mapping[onnx_name]
            pytorch_state[pytorch_name] = tensor
        else:
            unmapped.append(onnx_name)
    
    if unmapped:
        logging.warning(f"\nUnmapped ONNX parameters ({len(unmapped)}):")
        for name in unmapped[:10]:
            logging.warning(f"  - {name}")
        if len(unmapped) > 10:
            logging.warning(f"  ... and {len(unmapped) - 10} more")
    
    logging.info(f"\nConverted {len(pytorch_state)} parameters to PyTorch format")
    return pytorch_state


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX-style weights to PyTorch LResNet100E-IR format'
    )
    parser.add_argument(
        '--onnx-model',
        type=str,
        required=True,
        help='Path to ONNX model (.onnx) - defines architecture'
    )
    parser.add_argument(
        '--onnx-weights',
        type=str,
        required=True,
        help='Path to ONNX-style weights (.pth with p_*, b_* keys)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for converted PyTorch weights (.pth)'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='lresnet100e_ir',
        help='PyTorch backbone architecture (default: lresnet100e_ir)'
    )
    parser.add_argument(
        '--embedding-size',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )
    parser.add_argument(
        '--test-load',
        action='store_true',
        help='Test loading the converted weights into PyTorch model'
    )
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("ONNX to PyTorch Weight Conversion")
    logging.info("=" * 60)
    
    # Step 1: Load ONNX architecture
    logging.info("\nStep 1: Analyzing ONNX model...")
    onnx_params = load_onnx_param_names(args.onnx_model)
    
    # Step 2: Build PyTorch model
    logging.info(f"\nStep 2: Building PyTorch {args.network} model...")
    pytorch_model = get_model(
        args.network,
        dropout=0.0,
        fp16=False,
        num_features=args.embedding_size
    )
    pytorch_params = build_pytorch_param_names(pytorch_model)
    
    # Step 3: Create name mapping
    logging.info("\nStep 3: Creating parameter name mapping...")
    name_mapping = create_name_mapping(onnx_params, pytorch_params)
    
    # Step 4: Convert weights
    logging.info("\nStep 4: Converting weights...")
    pytorch_state = convert_weights(args.onnx_weights, name_mapping)
    
    # Step 5: Save converted weights
    logging.info(f"\nStep 5: Saving converted weights to {args.output}...")
    torch.save(pytorch_state, args.output)
    logging.info("✓ Saved successfully")
    
    # Optional: Test loading
    if args.test_load:
        logging.info("\nStep 6: Testing weight loading...")
        try:
            missing, unexpected = pytorch_model.load_state_dict(pytorch_state, strict=False)
            if missing:
                logging.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
            if not missing and not unexpected:
                logging.info("✓ All weights loaded successfully!")
            
            # Test forward pass
            logging.info("\nTesting forward pass...")
            pytorch_model.eval()
            dummy = torch.randn(1, 3, 112, 112)
            with torch.no_grad():
                out = pytorch_model(dummy)
            logging.info(f"✓ Forward pass successful! Output shape: {out.shape}")
            
        except Exception as e:
            logging.error(f"✗ Failed to load weights: {e}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Conversion complete!")
    logging.info("=" * 60)
    logging.info(f"\nTo use in training:")
    logging.info(f"  1. Set config.network = '{args.network}'")
    logging.info(f"  2. Pass --backbone-pth {args.output} to train_v3.py")
    logging.info(f"  3. Do NOT pass --onnx-backbone (use native PyTorch now)")


if __name__ == '__main__':
    main()
