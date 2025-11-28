# ReidNet V3 Training Setup - ONNX Backbone Fine-tuning

## Architecture & Weights

### The .pth File Issue
The `nvr.prod.v7.facerec.backbone.pth` file contains **fused/optimized weights** (237 parameters):
- BatchNorm layers were fused into Conv layers during ONNX export
- This is an inference optimization that reduces memory and compute
- **Cannot be directly loaded into unfused PyTorch models** (LResNet100E-IR needs 1268 parameters)

### Solution: ONNXArcFaceBackbone Wrapper
The ONNX model (`nvr.prod.v7.facerec.backbone.onnx`) contains the **full architecture graph**:
- Complete LResNet100E-IR structure with all layers
- All weights are already embedded in the .onnx file
- The wrapper loads the ONNX graph and registers all initializers as `nn.Parameter`
- **Fully trainable** - all parameters require gradients
- **Pure PyTorch** - no onnxruntime dependency during training, only at initialization

## Training Configuration

### Config File: `configs/reidnet_v3_finetune.py`
```python
config.network = "r100"           # Ignored when --onnx-backbone is used
config.embedding_size = 512       # Will be overridden by ONNX model output dim
config.margin_list = (1.0, 0.5, 0.0)
config.batch_size = 128           # Per-GPU batch size
config.lr = 0.01                  # Conservative LR for fine-tuning
config.num_epoch = 24
config.fp16 = True
```

### Training Command
```bash
# Auto-detect number of GPUs and use torchrun for distributed training
# Replace N with your GPU count (1, 2, 4, 8, etc.)
torchrun --nproc_per_node=N --standalone train_v3.py configs/reidnet_v3_finetune.py \
  --onnx-backbone /path/to/nvr.prod.v7.facerec.backbone.onnx

# Example for single GPU:
torchrun --nproc_per_node=1 --standalone train_v3.py configs/reidnet_v3_finetune.py \
  --onnx-backbone pretrained_models/nvr.prod.v7.facerec.backbone.onnx

# Example for 8 GPUs:
torchrun --nproc_per_node=8 --standalone train_v3.py configs/reidnet_v3_finetune.py \
  --onnx-backbone pretrained_models/nvr.prod.v7.facerec.backbone.onnx
```

**Note**: The notebook automatically detects GPU count and uses `torchrun` accordingly.

## How train_v3.py Handles This

### Case 1: --onnx-backbone provided (lines 93-112)
```python
if args.onnx_backbone is not None:
    # Load ONNX model and wrap it
    core = ONNXArcFaceBackbone(args.onnx_backbone, fp16=cfg.fp16)
    
    # Infer embedding size from model
    dummy = torch.randn(1, 3, 112, 112)
    emb = core(dummy)
    cfg.embedding_size = emb.shape[1]  # Override config with actual 512
    
    # Optionally load .pth if provided (but weights are already in .onnx)
    if args.backbone_pth:
        load_backbone_state(core, args.backbone_pth)
```

### ONNXArcFaceBackbone Implementation
- Parses ONNX graph nodes (Conv, BatchNorm, PReLU, Add, etc.)
- Registers all ONNX initializers as `nn.Parameter` (trainable by default)
- Implements forward pass by interpreting ONNX graph with PyTorch ops
- Uses PyTorch autograd for backprop
- Compatible with DDP, mixed precision, gradient clipping

## Training Will Work Because:

1. ✅ **Architecture is correct**: ONNX graph contains full LResNet100E-IR structure
2. ✅ **Weights are loaded**: All parameters from ONNX initializers
3. ✅ **Fully trainable**: All parameters registered with `requires_grad=True`
4. ✅ **No runtime dependency**: Pure PyTorch after initialization
5. ✅ **DDP compatible**: Wrapped in `DistributedDataParallel` (line 171-177)
6. ✅ **Mixed precision**: Works with `torch.cuda.amp.autocast()`
7. ✅ **Gradient flow**: All ops implemented with PyTorch differentiable operations

## Notebook Flow

The `reidnet_v3_training_brevdev_refactored.ipynb` already handles this:

1. **Download ONNX model** (cell 6):
   ```python
   aws s3 cp s3://.../nvr.prod.v7.facerec.backbone.onnx pretrained_models/
   ```

2. **Create config** (cell 7):
   - Generates `configs/reidnet_v3_finetune.py`
   - Sets `network="r100"` (ignored when using --onnx-backbone)

3. **Start training** (cell 8):
   - Auto-detects number of GPUs using `torch.cuda.device_count()`
   - Uses `torchrun --nproc_per_node=N --standalone` for distributed training
   - Adds `--onnx-backbone` flag if ONNX model is available
   ```python
   num_gpus = torch.cuda.device_count()
   cmd = ["torchrun", f"--nproc_per_node={num_gpus}", "--standalone", 
          "train_v3.py", "configs/reidnet_v3_finetune.py"]
   if ONNX_BACKBONE:
       cmd.extend(["--onnx-backbone", ONNX_BACKBONE])
   ```

## Why Not Use Native PyTorch LResNet100E-IR?

The .pth weights are **fused** - BatchNorm parameters are absorbed into Conv layers:
- Native PyTorch model needs separate BN layers (1268 parameters total)
- .pth file only has fused weights (237 parameters)
- Shape mismatches make direct loading impossible
- Would need the original unfused MXNet .params file (not available)

## Summary

**Status**: ✅ Ready to train

**Command**:
```bash
torchrun --nproc_per_node=8 train_v3.py configs/reidnet_v3_finetune.py \
  --onnx-backbone pretrained_models/nvr.prod.v7.facerec.backbone.onnx
```

**Architecture**: LResNet100E-IR via ONNXArcFaceBackbone wrapper  
**Pretrained weights**: Loaded from ONNX initializers  
**Trainable**: Yes, all parameters have gradients  
**ONNX dependency**: Only at initialization to parse graph, not during training
