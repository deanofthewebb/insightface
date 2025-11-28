"""
Map sequential parameter names (p_0, p_1, b_0, b_1, ...) to PyTorch LResNet100E-IR module names.

This is for loading weights that were originally converted from ONNX/MXNet format.
No ONNX dependency required - we just need to know the model architecture.
"""

from collections import OrderedDict
import torch
import torch.nn as nn


def get_lresnet100e_ir_param_order(model):
    """
    Get ordered list of parameter/buffer names from LResNet100E-IR model.
    
    Returns:
        List of (name, shape) tuples in model traversal order
    """
    params = []
    
    # Parameters (weights, biases, BN scales)
    for name, param in model.named_parameters():
        params.append((name, tuple(param.shape)))
    
    # Buffers (BN running stats)
    for name, buffer in model.named_buffers():
        params.append((name, tuple(buffer.shape)))
    
    return params


def map_sequential_to_pytorch(sequential_state_dict, pytorch_model):
    """
    Map sequential parameter names (p_0, p_1, b_0, b_1, ...) to PyTorch module names.
    
    Args:
        sequential_state_dict: State dict with keys like 'p_0', 'p_1', 'b_0', etc.
        pytorch_model: PyTorch model (e.g., LResNet100E-IR)
    
    Returns:
        OrderedDict with proper PyTorch parameter names
    """
    # Get ordered parameters from PyTorch model
    pytorch_params = get_lresnet100e_ir_param_order(pytorch_model)
    
    # Separate p_* and b_* keys
    p_keys = sorted([k for k in sequential_state_dict.keys() if k.startswith('p_')], 
                    key=lambda x: int(x.split('_')[1]))
    b_keys = sorted([k for k in sequential_state_dict.keys() if k.startswith('b_')],
                    key=lambda x: int(x.split('_')[1]))
    
    # All sequential keys in order
    all_seq_keys = p_keys + b_keys
    
    print(f"Sequential state dict: {len(all_seq_keys)} parameters")
    print(f"  - {len(p_keys)} weight parameters (p_*)")
    print(f"  - {len(b_keys)} buffer/bias parameters (b_*)")
    print(f"PyTorch model: {len(pytorch_params)} parameters + buffers")
    
    if len(all_seq_keys) != len(pytorch_params):
        print(f"WARNING: Count mismatch! {len(all_seq_keys)} vs {len(pytorch_params)}")
    
    # Map sequential names to PyTorch names by matching shapes and order
    mapped_state = OrderedDict()
    
    # Group by shape for better matching
    seq_by_shape = {}
    for key in all_seq_keys:
        tensor = sequential_state_dict[key]
        shape = tuple(tensor.shape)
        if shape not in seq_by_shape:
            seq_by_shape[shape] = []
        seq_by_shape[shape].append(key)
    
    pytorch_by_shape = {}
    for name, shape in pytorch_params:
        if shape not in pytorch_by_shape:
            pytorch_by_shape[shape] = []
        pytorch_by_shape[shape].append(name)
    
    # Match by shape
    for shape in sorted(seq_by_shape.keys()):
        seq_names = seq_by_shape[shape]
        pt_names = pytorch_by_shape.get(shape, [])
        
        if len(seq_names) != len(pt_names):
            print(f"WARNING: Shape {shape} has {len(seq_names)} sequential vs {len(pt_names)} PyTorch params")
            continue
        
        for seq_name, pt_name in zip(seq_names, pt_names):
            mapped_state[pt_name] = sequential_state_dict[seq_name]
    
    print(f"\nMapped {len(mapped_state)}/{len(pytorch_params)} parameters")
    
    return mapped_state


def load_sequential_weights(model, ckpt_path, strict=False):
    """
    Load weights from checkpoint with sequential names (p_*, b_*) into PyTorch model.
    
    Args:
        model: PyTorch model (e.g., LResNet100E-IR)
        ckpt_path: Path to .pth checkpoint
        strict: Whether to use strict loading (default: False)
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    print(f"Loading sequential weights from: {ckpt_path}")
    
    # Load checkpoint
    state = torch.load(ckpt_path, map_location='cpu')
    
    # Handle full checkpoint format
    if isinstance(state, dict) and 'state_dict_backbone' in state:
        state = state['state_dict_backbone']
    
    # Check if it's sequential format
    has_p_keys = any(k.startswith('p_') for k in state.keys())
    has_b_keys = any(k.startswith('b_') for k in state.keys())
    
    if not (has_p_keys or has_b_keys):
        # Not sequential format, load directly
        print("Standard PyTorch format detected, loading directly...")
        return model.load_state_dict(state, strict=strict)
    
    # Map sequential names to PyTorch names
    print("Sequential format (p_*, b_*) detected, mapping to PyTorch names...")
    mapped_state = map_sequential_to_pytorch(state, model)
    
    # Load into model
    missing, unexpected = model.load_state_dict(mapped_state, strict=strict)
    
    if missing:
        print(f"Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                print(f"  - {k}")
    
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"  - {k}")
    
    return missing, unexpected
