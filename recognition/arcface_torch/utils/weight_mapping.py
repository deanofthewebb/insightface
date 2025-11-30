"""
Map sequential parameter names (p_0, p_1, b_0, b_1, ...) to PyTorch module names.

This is for loading weights that were originally trained using ONNXArcFaceBackbone.
Supports both shape-based mapping (approximate) and ONNX graph-based mapping (precise).
"""

from collections import OrderedDict
import os
import torch
import torch.nn as nn

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def get_lresnet100e_ir_param_order(model):
    """
    Get ordered list of parameter/buffer names from any PyTorch model.
    
    Note: Function name is historical (from LResNet), but works with any model.
    
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


def extract_onnx_param_order(onnx_path):
    """
    Extract parameter names from ONNX file in the order ONNXArcFaceBackbone processes them.
    
    This replicates the logic from ONNXArcFaceBackbone.__init__() to determine:
    1. Which initializers are parameters vs buffers
    2. The order they are registered (p_0, p_1, ..., b_0, b_1, ...)
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Tuple of (param_names, buffer_names) where each is a list of ONNX initializer names
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx package is required for ONNX-based mapping. Install with: pip install onnx")
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # Collect all initializer names
    initializer_names = [init.name for init in graph.initializer]
    
    # Determine which are parameters vs buffers (same logic as ONNXArcFaceBackbone)
    param_inits = set()
    bn_stat_inits = set()
    
    for node in graph.node:
        op = node.op_type
        if op == "BatchNormalization":
            # inputs: x, scale, bias, mean, var
            if len(node.input) >= 5:
                scale_name = node.input[1]
                bias_name = node.input[2]
                mean_name = node.input[3]
                var_name = node.input[4]
                param_inits.add(scale_name)
                param_inits.add(bias_name)
                bn_stat_inits.add(mean_name)
                bn_stat_inits.add(var_name)
        elif op == "Conv":
            # inputs: x, W, (B)
            if len(node.input) >= 2:
                param_inits.add(node.input[1])
            if len(node.input) >= 3 and node.input[2]:
                param_inits.add(node.input[2])
        elif op == "Gemm":
            # Fully connected: x, W, (B)
            if len(node.input) >= 2:
                param_inits.add(node.input[1])
            if len(node.input) >= 3:
                param_inits.add(node.input[2])
        elif op in ("PRelu", "PReLU"):
            if len(node.input) >= 2:
                param_inits.add(node.input[1])
    
    # Separate into params and buffers in initializer order
    param_names = []
    buffer_names = []
    
    for name in initializer_names:
        if name in bn_stat_inits:
            buffer_names.append(name)
        elif name in param_inits:
            param_names.append(name)
        else:
            buffer_names.append(name)
    
    return param_names, buffer_names


def map_onnx_to_pytorch(onnx_path, pytorch_model):
    """
    Map ONNX initializer names to PyTorch module parameter/buffer names.
    
    This creates a mapping by:
    1. Extracting ONNX parameters in the order ONNXArcFaceBackbone would process them
    2. Extracting PyTorch parameters in named_parameters() order
    3. Matching them by position and shape
    
    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Target PyTorch model
        
    Returns:
        Dict mapping ONNX name -> PyTorch name
    """
    # Extract ONNX parameter order
    onnx_param_names, onnx_buffer_names = extract_onnx_param_order(onnx_path)
    
    # Get PyTorch params and buffers in order
    pytorch_params = []
    for name, param in pytorch_model.named_parameters():
        pytorch_params.append((name, tuple(param.shape)))
    
    pytorch_buffers = []
    for name, buffer in pytorch_model.named_buffers():
        pytorch_buffers.append((name, tuple(buffer.shape)))
    
    # Load ONNX to get shapes
    model = onnx.load(onnx_path)
    initializer_tensors = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        initializer_tensors[init.name] = arr
    
    # Build ONNX param/buffer shapes
    onnx_param_shapes = []
    for name in onnx_param_names:
        if name in initializer_tensors:
            shape = tuple(initializer_tensors[name].shape)
            onnx_param_shapes.append((name, shape))
    
    onnx_buffer_shapes = []
    for name in onnx_buffer_names:
        if name in initializer_tensors:
            shape = tuple(initializer_tensors[name].shape)
            onnx_buffer_shapes.append((name, shape))
    
    # Map by position and validate shapes
    mapping = {}
    
    print(f"\nONNX model has {len(onnx_param_shapes)} parameters, {len(onnx_buffer_shapes)} buffers")
    print(f"PyTorch model has {len(pytorch_params)} parameters, {len(pytorch_buffers)} buffers")
    
    # Map parameters by position
    for i, ((onnx_name, onnx_shape), (pt_name, pt_shape)) in enumerate(zip(onnx_param_shapes, pytorch_params)):
        if onnx_shape == pt_shape:
            mapping[onnx_name] = pt_name
        else:
            print(f"  WARNING: Param {i}: shape mismatch {onnx_name} {onnx_shape} vs {pt_name} {pt_shape}")
    
    # Map buffers by position
    for i, ((onnx_name, onnx_shape), (pt_name, pt_shape)) in enumerate(zip(onnx_buffer_shapes, pytorch_buffers)):
        if onnx_shape == pt_shape:
            mapping[onnx_name] = pt_name
        else:
            print(f"  WARNING: Buffer {i}: shape mismatch {onnx_name} {onnx_shape} vs {pt_name} {pt_shape}")
    
    print(f"Mapped {len(mapping)} ONNX -> PyTorch names")
    
    return mapping


def map_sequential_to_pytorch(sequential_state_dict, pytorch_model, onnx_path=None):
    """
    Map sequential parameter names (p_0, p_1, b_0, b_1, ...) to PyTorch module names.
    
    Args:
        sequential_state_dict: State dict with keys like 'p_0', 'p_1', 'b_0', etc.
        pytorch_model: PyTorch model (e.g., LResNet100E-IR, IResNet100)
        onnx_path: Optional path to ONNX file for precise graph-based mapping
    
    Returns:
        OrderedDict with proper PyTorch parameter names
    """
    # Separate p_* and b_* keys
    p_keys = sorted([k for k in sequential_state_dict.keys() if k.startswith('p_')], 
                    key=lambda x: int(x.split('_')[1]))
    b_keys = sorted([k for k in sequential_state_dict.keys() if k.startswith('b_')],
                    key=lambda x: int(x.split('_')[1]))
    
    print(f"Sequential state dict: {len(p_keys)} parameters (p_*) + {len(b_keys)} buffers (b_*)")
    
    # If ONNX path provided, use graph-based mapping
    if onnx_path and os.path.exists(onnx_path) and ONNX_AVAILABLE:
        print(f"Using ONNX graph from {onnx_path} for precise mapping...")
        
        # Get ONNX parameter names in order
        onnx_param_names, onnx_buffer_names = extract_onnx_param_order(onnx_path)
        
        # Create mapping: p_N/b_N -> ONNX name
        seq_to_onnx = {}
        for i, onnx_name in enumerate(onnx_param_names):
            seq_to_onnx[f'p_{i}'] = onnx_name
        for i, onnx_name in enumerate(onnx_buffer_names):
            seq_to_onnx[f'b_{i}'] = onnx_name
        
        # Create mapping: ONNX name -> PyTorch name
        onnx_to_pytorch = map_onnx_to_pytorch(onnx_path, pytorch_model)
        
        # Compose mappings: p_N/b_N -> ONNX name -> PyTorch name
        mapped_state = OrderedDict()
        unmapped = []
        
        for seq_key, tensor in sequential_state_dict.items():
            if seq_key in seq_to_onnx:
                onnx_name = seq_to_onnx[seq_key]
                if onnx_name in onnx_to_pytorch:
                    pt_name = onnx_to_pytorch[onnx_name]
                    mapped_state[pt_name] = tensor
                else:
                    unmapped.append(f"{seq_key} -> {onnx_name} (no PyTorch match)")
            else:
                unmapped.append(f"{seq_key} (no ONNX match)")
        
        if unmapped:
            print(f"\nWARNING: {len(unmapped)} parameters could not be mapped:")
            for msg in unmapped[:10]:
                print(f"  - {msg}")
            if len(unmapped) > 10:
                print(f"  ... and {len(unmapped) - 10} more")
        
        pytorch_param_count = len(list(pytorch_model.named_parameters())) + len(list(pytorch_model.named_buffers()))
        print(f"\nMapped {len(mapped_state)}/{pytorch_param_count} parameters")
        
        return mapped_state
    
    # Fallback: shape-based mapping (less precise)
    print("Using shape-based mapping (no ONNX file provided)...")
    
    # Get ordered parameters from PyTorch model
    pytorch_params = get_lresnet100e_ir_param_order(pytorch_model)
    
    # All sequential keys in order
    all_seq_keys = p_keys + b_keys
    
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


def load_sequential_weights(model, ckpt_path, strict=False, onnx_path=None):
    """
    Load weights from checkpoint with sequential names (p_*, b_*) into PyTorch model.
    
    Args:
        model: PyTorch model (e.g., LResNet100E-IR, IResNet100)
        ckpt_path: Path to .pth checkpoint
        strict: Whether to use strict loading (default: False)
        onnx_path: Optional path to ONNX file for precise graph-based mapping
    
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
    mapped_state = map_sequential_to_pytorch(state, model, onnx_path=onnx_path)
    
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
