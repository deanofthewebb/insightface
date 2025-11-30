import numpy as np
import onnx
import torch


def looks_like_sequential_weights(state_dict):
    """Check if state dict uses sequential naming (p_*, b_*)"""
    keys = list(state_dict.keys())
    if not keys:
        return False
    sequential_keys = sum(k.startswith(("p_", "b_")) for k in keys)
    return sequential_keys > 0.7 * len(keys)


def infer_architecture_from_param_count(num_params_buffers):
    """
    Infer the likely model architecture from parameter + buffer count.
    
    Returns suggested architecture name and confidence level.
    """
    arch_signatures = {
        'r18': (170, 210),
        'r34': (315, 350),
        'r50': (460, 495),
        'r100': (900, 950),
        'lresnet50e_ir': (625, 665),
        'lresnet100e_ir': (1250, 1285),
        'mbf': (315, 350),
    }
    
    for arch, (min_count, max_count) in arch_signatures.items():
        if min_count <= num_params_buffers <= max_count:
            return arch, 'high'
    
    return None, 'unknown'


def convert_onnx(net, path_module, output, opset=11, simplify=False, onnx_reference=None):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module, map_location='cpu')
    
    if isinstance(weight, dict) and 'state_dict_backbone' in weight:
        weight = weight['state_dict_backbone']
    
    if looks_like_sequential_weights(weight):
        print(f"Detected sequential weight format (p_*, b_*). Mapping to PyTorch names...")
        
        p_keys = [k for k in weight.keys() if k.startswith('p_')]
        b_keys = [k for k in weight.keys() if k.startswith('b_')]
        total_params = len(p_keys) + len(b_keys)
        
        print(f"Checkpoint contains {len(p_keys)} parameters (p_*) + {len(b_keys)} buffers (b_*) = {total_params} total")
        
        model_param_count = len(list(net.named_parameters())) + len(list(net.named_buffers()))
        print(f"Target model has {model_param_count} parameters + buffers")
        
        if abs(total_params - model_param_count) > 10:
            suggested_arch, confidence = infer_architecture_from_param_count(total_params)
            print(f"\n⚠️  WARNING: Parameter count mismatch!")
            print(f"   Checkpoint: {total_params} params+buffers")
            print(f"   Model:      {model_param_count} params+buffers")
            if suggested_arch:
                print(f"   Suggested architecture: --network {suggested_arch}")
            print(f"\n   This may result in loading errors or incorrect mappings.")
            print(f"   Consider using the correct architecture matching your checkpoint.\n")
        
        from utils.weight_mapping import load_sequential_weights
        
        # Use ONNX reference if provided for precise mapping
        if onnx_reference:
            print(f"Using ONNX reference file for graph-based mapping: {onnx_reference}")
        
        missing, unexpected = load_sequential_weights(
            net, path_module, strict=False, onnx_path=onnx_reference
        )
        
        if missing or unexpected:
            print(f"\n⚠️  Weight loading completed with issues:")
            if missing:
                print(f"   Missing keys: {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
            print(f"   Export may fail or produce incorrect results.\n")
    else:
        net.load_state_dict(weight, strict=True)
    
    net.eval()
    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(
        description='ArcFace PyTorch to ONNX converter',
        epilog='''
Examples:
  # Convert standard PyTorch checkpoint
  python torch2onnx.py model.pth --network r100 --output model.onnx
  
  # Convert sequential format (p_*, b_*) checkpoint with shape-based mapping
  python torch2onnx.py backbone.pth --network r18 --output model.onnx
  
  # Convert sequential format with ONNX graph-based mapping (precise)
  python torch2onnx.py backbone.pth --network lresnet100e_ir \\
    --onnx-reference /path/to/original.onnx --output model.onnx
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, 
                       help='backbone network (r18, r34, r50, r100, lresnet100e_ir, etc.)')
    parser.add_argument('--onnx-reference', type=str, default=None,
                       help='ONNX file used during training (for precise p_*/b_* mapping)')
    parser.add_argument('--simplify', type=bool, default=False, help='apply onnx simplification')
    parser.add_argument('--embedding-size', type=int, default=512, 
                       help='embedding dimension (default: 512)')
    args = parser.parse_args()
    
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    if args.network is None:
        print("Error: --network is required")
        print("Available networks: r18, r34, r50, r100, lresnet50e_ir, lresnet100e_ir, mbf")
        exit(1)
    
    if args.onnx_reference and not os.path.exists(args.onnx_reference):
        print(f"Warning: ONNX reference file not found: {args.onnx_reference}")
        print("Falling back to shape-based mapping...")
        args.onnx_reference = None
    
    print(f"Loading checkpoint from: {input_file}")
    print(f"Target architecture: {args.network}")
    print(f"Embedding size: {args.embedding_size}")
    if args.onnx_reference:
        print(f"ONNX reference: {args.onnx_reference}")
    print()
    
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=args.embedding_size)
    
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    
    convert_onnx(backbone_onnx, input_file, args.output, 
                simplify=args.simplify, onnx_reference=args.onnx_reference)
    
    print(f"\n✓ Successfully exported to: {args.output}")
