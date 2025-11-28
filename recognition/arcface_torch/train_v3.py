#!/usr/bin/env python
import argparse
import logging
import os

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

from onnx_arcface_backbone import ONNXArcFaceBackbone, compare_onnx_and_pytorch
from utils.model_utils import print_model_summary

assert torch.__version__ >= "1.9.0"

# ---------------- distributed init ----------------
try:
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    WORLD_SIZE = 1
    RANK = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=RANK,
        world_size=WORLD_SIZE,
    )


def load_backbone_state(core, ckpt_path: str):
    """Load backbone weights from a .pth file.

    Supports:
      - pure state_dict
      - full checkpoint with key 'state_dict_backbone'
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict_backbone" in state:
        core.load_state_dict(state["state_dict_backbone"], strict=True)
        logging.info(f"[Backbone] Loaded state_dict_backbone from {ckpt_path}")
    else:
        core.load_state_dict(state, strict=True)
        logging.info(f"[Backbone] Loaded raw state_dict from {ckpt_path}")


def build_backbone(cfg, args):
    """Build backbone according to config and CLI flags."""
    local_device = f"cuda:{args.local_rank}"
    torch.cuda.set_device(args.local_rank)

    # --- Case 1: ONNX backbone (LResNet100E-IR graph) ---
    if args.onnx_backbone is not None:
        logging.info(f"[Backbone] Using ONNXArcFaceBackbone from {args.onnx_backbone}")
        core = ONNXArcFaceBackbone(
            args.onnx_backbone,
            fp16=cfg.fp16,
        ).to(local_device)

        # Infer embedding dimension from a dummy forward
        core.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 112, 112, device=local_device)
            emb = core(dummy)
        emb_dim = int(emb.shape[1])
        if cfg.embedding_size != emb_dim:
            logging.warning(
                f"[Backbone] cfg.embedding_size={cfg.embedding_size} "
                f"!= ONNX embedding_dim={emb_dim}. Overriding."
            )
            cfg.embedding_size = emb_dim

    # --- Case 2: standard arcface_torch backbone ---
    else:
        logging.info(f"[Backbone] Using built-in backbone: {cfg.network}")
        core = get_model(
            cfg.network,
            dropout=0.0,
            fp16=cfg.fp16,
            num_features=cfg.embedding_size,
        ).to(local_device)

    # --- Optional: load raw PyTorch checkpoint weights (.pth) ---
    if args.backbone_pth is not None:
        load_backbone_state(core, args.backbone_pth)

    # --- Optional: print summary & compare with ONNX ---
    if RANK == 0 and args.print_summary:
        print_model_summary(core, input_shape=(1, 3, 112, 112))

    if RANK == 0 and args.compare_onnx and args.onnx_backbone is not None:
        compare_onnx_and_pytorch(
            args.onnx_backbone,
            pytorch_model=core,
            device=local_device,
        )

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=core,
        broadcast_buffers=False,
        device_ids=[args.local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    backbone.train()
    backbone._set_static_graph()
    return backbone


def main(args):
    cfg = get_config(args.config)

    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(args.local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(RANK, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if RANK == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers,
    )

    backbone = build_backbone(cfg, args)

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    # PartialFC head + optimizer
    module_partial_fc = PartialFC_V2(
        margin_loss,
        cfg.embedding_size,
        cfg.num_classes,
        cfg.sample_rate,
        cfg.fp16,
    )
    module_partial_fc.train().cuda()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    # LR schedule
    cfg.total_batch_size = cfg.batch_size * WORLD_SIZE
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1,
    )

    # Optional resume from full training checkpoint
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        ckpt_path = os.path.join(cfg.output, f"checkpoint_gpu_{RANK}.pt")
        logging.info(f"[Resume] Loading {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        backbone.module.load_state_dict(checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["state_lr_scheduler"])
        del checkpoint

    for key, value in cfg.items():
        num_space = max(1, 25 - len(key))
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets,
        rec_prefix=cfg.rec,
        summary_writer=summary_writer,
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer,
    )

    loss_meter = AverageMeter()
    scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # --------------- training loop ---------------
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            img = img.cuda(args.local_rank, non_blocking=True)
            local_labels = local_labels.cuda(args.local_rank, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                local_embeddings = backbone(img)
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                scaler.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad()

            lr_scheduler.step()

            with torch.no_grad():
                loss_meter.update(loss.item(), 1)
                callback_logging(
                    global_step,
                    loss_meter,
                    epoch,
                    cfg.fp16,
                    lr_scheduler.get_last_lr()[0],
                    scaler,
                )

            if global_step % cfg.verbose == 0 and global_step > 0:
                callback_verification(global_step, backbone)

            if cfg.save_all_states and global_step % cfg.save_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "state_dict_backbone": backbone.module.state_dict(),
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(cfg.output, f"checkpoint_gpu_{RANK}.pt"),
                )

        if RANK == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="ArcFace training with ONNX or PyTorch backbone."
    )
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for DDP")
    parser.add_argument(
        "--onnx-backbone",
        type=str,
        default=None,
        help="Path to ONNX backbone (LResNet100E-IR). "
             "If omitted, uses cfg.network + get_model().",
    )
    parser.add_argument(
        "--backbone-pth",
        type=str,
        default=None,
        help="Optional .pth file with backbone weights (raw state_dict or "
             "full checkpoint with key 'state_dict_backbone').",
    )
    parser.add_argument(
        "--compare-onnx",
        action="store_true",
        help="Compare PyTorch backbone vs ONNXRuntime outputs (requires --onnx-backbone).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a simple model summary for the backbone.",
    )
    main(parser.parse_args())
