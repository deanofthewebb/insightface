# model_utils.py (or inside onnx_arcface_backbone.py)
import torch

def print_model_summary(model, input_shape=(1, 3, 112, 112)):
    """
    Simple, dependency‑free summary for ONNXArcFaceBackbone.
    Prints input/output shapes and parameter counts.
    """
    device = next(model.parameters()).device
    model.eval()

    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        out = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== ONNXArcFaceBackbone Summary ===")
    print(f"Device:            {device}")
    print(f"Input shape:       {tuple(dummy.shape)}")
    print(f"Output shape:      {tuple(out.shape)}")
    print(f"Embedding dim:     {out.shape[1]}")
    print(f"Total params:      {total_params:,}")
    print(f"Trainable params:  {trainable_params:,}")
    print(f"Frozen params:     {total_params - trainable_params:,}")
    print("Parameter tensors:")
    for name, p in model.named_parameters():
        print(f"  {name:30s} {list(p.shape)}  requires_grad={p.requires_grad}")
