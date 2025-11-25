#!/usr/bin/env python3
import argparse
import logging
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import numpy_helper


def get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.FLOATS:
                return list(a.floats)
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode("utf-8")
            if a.type == onnx.AttributeProto.STRINGS:
                return [s.decode("utf-8") for s in a.strings]
    return default


class ONNXArcFaceBackbone(nn.Module):
    """
    PyTorch wrapper around an ArcFace ONNX backbone (e.g. LResNet100E-IR).

    - Loads all ONNX initializers as Parameters or buffers.
    - Implements forward by interpreting the ONNX graph with PyTorch ops.
    - Fully differentiable → can be fine-tuned.
    """

    def __init__(self, onnx_path: str, fp16: bool = False):
        super().__init__()
        self.fp16 = fp16

        logging.info(f"[INFO] Loading ONNX model from: {onnx_path}")
        self.onnx_model = onnx.load(onnx_path)
        self.graph = self.onnx_model.graph

        # Collect initializers (ONNX weights) as numpy arrays
        self.initializer_tensors: Dict[str, np.ndarray] = {}
        for init in self.graph.initializer:
            arr = numpy_helper.to_array(init)
            # make a writable float32 copy to avoid numpy->torch non-writable warning
            arr = np.array(arr, dtype=np.float32, copy=True)
            self.initializer_tensors[init.name] = arr

        self.input_names = [i.name for i in self.graph.input]
        self.output_names = [o.name for o in self.graph.output]

        if len(self.input_names) != 1:
            logging.warning(
                f"[WARN] Expected exactly one input, found {len(self.input_names)}: {self.input_names}"
            )
        if len(self.output_names) != 1:
            logging.warning(
                f"[WARN] Expected exactly one output, found {len(self.output_names)}: {self.output_names}"
            )

        # Maps from ONNX initializer name -> registered module param/buffer name
        self._param_name_map: Dict[str, str] = {}
        self._buffer_name_map: Dict[str, str] = {}
        self._param_count = 0
        self._buffer_count = 0

        # Decide which initializers are trainable parameters vs constant buffers
        param_inits = set()
        bn_stat_inits = set()  # running_mean, running_var

        for node in self.graph.node:
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
            elif op in ("PRelu", "PReLU", "PRelu"):
                if len(node.input) >= 2:
                    param_inits.add(node.input[1])
            # Other ops (Reshape/Concat/etc) use shape constants → buffers

        # Instantiate Parameters and buffers
        for name in self.initializer_tensors.keys():
            if name in bn_stat_inits:
                self._get_buffer(name)  # BN running_mean/var
            elif name in param_inits:
                self._get_param(name, requires_grad=True)
            else:
                # Other initializers are constants (e.g. shape tensors for Reshape)
                self._get_buffer(name)

        logging.info(
            f"[INFO] ONNX params: {len(self._param_name_map)}, buffers: {len(self._buffer_name_map)}"
        )

    # ---- helpers to create/register params & buffers ----

    def _get_param(self, onnx_name: str, requires_grad: bool = True) -> nn.Parameter:
        if onnx_name in self._param_name_map:
            return getattr(self, self._param_name_map[onnx_name])
        if onnx_name not in self.initializer_tensors:
            raise KeyError(f"Initializer '{onnx_name}' not found in ONNX model.")
        arr = self.initializer_tensors[onnx_name]
        tensor = torch.from_numpy(arr.copy())  # copy to ensure writable
        param = nn.Parameter(tensor, requires_grad=requires_grad)
        safe_name = f"p_{self._param_count}"
        self._param_count += 1
        self.register_parameter(safe_name, param)
        self._param_name_map[onnx_name] = safe_name
        return param

    def _get_buffer(self, onnx_name: str) -> torch.Tensor:
        if onnx_name in self._buffer_name_map:
            return getattr(self, self._buffer_name_map[onnx_name])
        if onnx_name not in self.initializer_tensors:
            raise KeyError(f"Initializer '{onnx_name}' not found in ONNX model.")
        arr = self.initializer_tensors[onnx_name]
        tensor = torch.from_numpy(arr.copy())
        safe_name = f"b_{self._buffer_count}"
        self._buffer_count += 1
        self.register_buffer(safe_name, tensor)
        self._buffer_name_map[onnx_name] = safe_name
        return tensor

    def _get_value(self, name: str) -> torch.Tensor:
        """
        Return current tensor (parameter or buffer) for a given ONNX initializer name.
        This respects device moves (e.g. .to('cuda')).
        """
        if name in self._param_name_map:
            return getattr(self, self._param_name_map[name])
        if name in self._buffer_name_map:
            return getattr(self, self._buffer_name_map[name])
        raise KeyError(f"Value '{name}' not found in param/buffer maps.")

    # ---- ONNX graph interpreter ----

    def _run_onnx(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute ONNX graph with given inputs using PyTorch ops.
        """
        values: Dict[str, torch.Tensor] = dict(inputs)

        # Make initializers visible as values too (but fetch from module so device is correct)
        for onnx_name in self._param_name_map.keys():
            values[onnx_name] = self._get_value(onnx_name)
        for onnx_name in self._buffer_name_map.keys():
            values[onnx_name] = self._get_value(onnx_name)

        for node in self.graph.node:
            op = node.op_type

            if op == "Conv":
                x = values[node.input[0]]
                W = values[node.input[1]]
                B = None
                if len(node.input) >= 3 and node.input[2]:
                    B = values[node.input[2]]

                strides = get_attr(node, "strides", [1, 1])
                dilations = get_attr(node, "dilations", [1, 1])
                pads = get_attr(node, "pads", [0, 0, 0, 0])
                group = get_attr(node, "group", 1)

                if len(pads) == 4:
                    padding = (pads[0], pads[1])
                else:
                    padding = 0

                y = F.conv2d(
                    x,
                    W,
                    bias=B,
                    stride=tuple(strides),
                    padding=padding,
                    dilation=tuple(dilations),
                    groups=group,
                )
                values[node.output[0]] = y

            elif op == "BatchNormalization":
                x = values[node.input[0]]
                gamma = values[node.input[1]]
                beta = values[node.input[2]]
                running_mean = values[node.input[3]]
                running_var = values[node.input[4]]

                eps = get_attr(node, "epsilon", 1e-5)
                momentum = get_attr(node, "momentum", 0.9)

                y = F.batch_norm(
                    x,
                    running_mean,
                    running_var,
                    weight=gamma,
                    bias=beta,
                    training=self.training,
                    momentum=momentum,
                    eps=eps,
                )
                values[node.output[0]] = y

            elif op in ("Relu", "Relu6"):
                x = values[node.input[0]]
                values[node.output[0]] = F.relu(x)

            elif op in ("PRelu", "PReLU", "PRelu"):
                x = values[node.input[0]]
                slope = values[node.input[1]]
                # F.prelu expects scalar or 1D [C]; ONNX often stores [C,1,1] or [1,C,1,1]
                slope_flat = slope
                if slope_flat.dim() > 1:
                    slope_flat = slope_flat.reshape(-1)
                if slope_flat.numel() not in (1, x.shape[1]):
                    raise RuntimeError(
                        f"PReLU slope shape {tuple(slope.shape)} not compatible with "
                        f"input channels {x.shape[1]}"
                    )
                values[node.output[0]] = F.prelu(x, slope_flat)

            elif op == "Add":
                a = values[node.input[0]]
                b = values[node.input[1]]
                values[node.output[0]] = a + b

            elif op == "GlobalAveragePool":
                x = values[node.input[0]]
                values[node.output[0]] = F.adaptive_avg_pool2d(x, (1, 1))

            elif op == "AveragePool":
                x = values[node.input[0]]
                kernel = get_attr(node, "kernel_shape", [1, 1])
                strides = get_attr(node, "strides", kernel)
                pads = get_attr(node, "pads", [0, 0, 0, 0])
                padding = (pads[0], pads[1]) if pads else 0
                values[node.output[0]] = F.avg_pool2d(
                    x, kernel_size=tuple(kernel), stride=tuple(strides), padding=padding
                )

            elif op == "MaxPool":
                x = values[node.input[0]]
                kernel = get_attr(node, "kernel_shape", [2, 2])
                strides = get_attr(node, "strides", kernel)
                pads = get_attr(node, "pads", [0, 0, 0, 0])
                padding = (pads[0], pads[1]) if pads else 0
                values[node.output[0]] = F.max_pool2d(
                    x, kernel_size=tuple(kernel), stride=tuple(strides), padding=padding
                )

            elif op == "Flatten":
                x = values[node.input[0]]
                axis = get_attr(node, "axis", 1)
                if axis != 1:
                    new_shape = (
                        int(np.prod(x.shape[:axis])),
                        int(np.prod(x.shape[axis:])),
                    )
                    y = x.reshape(*new_shape)
                else:
                    y = x.reshape(x.shape[0], -1)
                values[node.output[0]] = y

            elif op == "Gemm":
                # Fully connected layer
                A = values[node.input[0]]
                W = values[node.input[1]]
                B = None
                if len(node.input) >= 3 and node.input[2]:
                    B = values[node.input[2]]

                alpha = get_attr(node, "alpha", 1.0)
                beta = get_attr(node, "beta", 1.0)
                transA = get_attr(node, "transA", 0)
                transB = get_attr(node, "transB", 1)

                if transA:
                    A = A.transpose(-1, -2)
                if transB:
                    W = W.transpose(0, 1)

                y = alpha * A.matmul(W)
                if B is not None:
                    y = y + beta * B
                values[node.output[0]] = y

            elif op == "MatMul":
                a = values[node.input[0]]
                b = values[node.input[1]]
                values[node.output[0]] = a.matmul(b)

            elif op == "Reshape":
                data = values[node.input[0]]
                shape_name = node.input[1]
                shape_tensor = values[shape_name]
                shape = shape_tensor.to(torch.long).cpu().numpy().tolist()
                values[node.output[0]] = data.reshape(*shape)

            elif op == "Concat":
                tensors = [values[n] for n in node.input]
                axis = get_attr(node, "axis", 1)
                values[node.output[0]] = torch.cat(tensors, dim=axis)

            elif op == "Transpose":
                x = values[node.input[0]]
                perm = get_attr(node, "perm", None)
                if perm is None:
                    perm = list(range(x.dim() - 1, -1, -1))
                values[node.output[0]] = x.permute(*perm)

            elif op in ("Identity", "Dropout"):
                x = values[node.input[0]]
                values[node.output[0]] = x

            else:
                raise NotImplementedError(f"Unsupported ONNX op_type: {op}")

        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: expects input [N, 3, H, W] → returns embedding (e.g. [N, 512]).
        """
        if self.fp16 and x.is_cuda:
            ctx = torch.amp.autocast("cuda", enabled=True)
        else:
            ctx = nullcontext()

        with ctx:
            input_name = self.input_names[0] if self.input_names else "data"
            outputs = self._run_onnx({input_name: x})
            out_name = self.output_names[0] if self.output_names else list(outputs.keys())[-1]
            y = outputs[out_name]
        return y


# ---------------- CLI / utility ----------------

def build_model(onnx_path: str, device: str = "cuda", fp16: bool = False) -> ONNXArcFaceBackbone:
    device = torch.device(device)
    model = ONNXArcFaceBackbone(onnx_path, fp16=fp16).to(device)
    model.eval()
    # quick sanity check
    dummy = torch.randn(1, 3, 112, 112, device=device)
    with torch.no_grad():
        out = model(dummy)
    logging.info(f"[INFO] Dummy forward output shape: {tuple(out.shape)}")
    return model

def compare_onnx_and_pytorch(
    onnx_path: str,
    model: ONNXArcFaceBackbone,
    device: str = "cuda",
    num_tests: int = 5,
) -> None:
    """
    Compare ONNX runtime output vs PyTorch ONNXArcFaceBackbone output.

    - Runs num_tests random inputs (N=1, C=3, H=112, W=112).
    - Prints abs and relative differences for each test.
    - Prints global max abs/rel diff at the end.

    Requires: pip install onnxruntime or onnxruntime-gpu
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is not installed. "
            "Install it with `pip install onnxruntime-gpu` (or `onnxruntime`)."
        ) from e

    # Choose providers based on device
    if device.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # ONNX Runtime session
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_input = sess.get_inputs()[0]
    ort_output = sess.get_outputs()[0]
    ort_in_name = ort_input.name
    ort_out_name = ort_output.name

    model.eval()  # ensure BN uses running stats

    max_abs = 0.0
    max_rel = 0.0

    for i in range(num_tests):
        # Random input
        x = torch.randn(1, 3, 112, 112, device=device, dtype=torch.float32)

        # PyTorch output
        with torch.no_grad():
            y_pt = model(x)
        y_pt_np = y_pt.detach().cpu().numpy().astype("float32")

        # ONNX Runtime output (always runs on CPU, even with CUDAExecutionProvider,
        # but the provider will use GPU internally if available)
        x_ort = x.detach().cpu().numpy().astype("float32")
        y_ort = sess.run([ort_out_name], {ort_in_name: x_ort})[0].astype("float32")

        # Differences
        abs_diff = np.max(np.abs(y_pt_np - y_ort))
        denom = np.maximum(np.abs(y_ort), 1e-6)
        rel_diff = np.max(np.abs(y_pt_np - y_ort) / denom)

        max_abs = max(max_abs, float(abs_diff))
        max_rel = max(max_rel, float(rel_diff))

        print(f"[COMPARE] Test {i}: max_abs={abs_diff:.6g}, max_rel={rel_diff:.6g}")

    print(f"[COMPARE] Overall max abs diff: {max_abs:.6g}")
    print(f"[COMPARE] Overall max rel diff: {max_rel:.6g}")



def main():
    parser = argparse.ArgumentParser(
        description="Wrap ArcFace ONNX backbone into a trainable PyTorch model (graph-aware)."
    )
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model (e.g. nvr.prod.v7.onnx)")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save backbone state_dict (e.g. nvr_prod_v7_backbone.pth)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--fp16", action="store_true", help="Use autocast fp16 in forward")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="If set, compare PyTorch wrapper output vs ONNX Runtime output.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of random test inputs to use when comparing outputs.",
    )
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build model (this already does a dummy forward and logs the output shape)
    model = build_model(args.onnx_path, device=args.device, fp16=args.fp16)

    # Optional ONNX vs PyTorch comparison
    if args.compare:
        compare_onnx_and_pytorch(
            args.onnx_path,
            model,
            device=args.device,
            num_tests=args.num_tests,
        )

    # Optional: save state_dict for later reuse
    if args.save:
        save_path = args.save
        torch.save(model.state_dict(), save_path)
        logging.info(f"[INFO] Saved backbone state_dict to: {save_path}")



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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = ONNXArcFaceBackbone(args.onnx_path, device=args.device, fp16=False)
    print_model_summary(model)