# configs/reidnet_v3_finetune.py
# Fine-tuning ReidNet V3 using ONNX backbone from nvr.prod.v7
from easydict import EasyDict as edict

cfg = edict()

# ------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------
cfg.output = "/home/ubuntu/checkpoints/reidnet_v3/work_dirs"
cfg.resume = False
cfg.fp16 = True
cfg.seed = 2048

# ------------------------------------------------------------------
# Dataset (RecordIO format)
# ------------------------------------------------------------------
cfg.rec = "/home/ubuntu/insightface_training/datasets/reidnet_v3_train"

# Dataset statistics - update these based on your actual data
cfg.num_classes = 100_000        # Number of identities
cfg.num_image = 1_000_000        # Total training images

# Validation targets (empty if no validation recs available)
cfg.val_targets = []

# ------------------------------------------------------------------
# Training schedule
# ------------------------------------------------------------------
cfg.batch_size = 128             # Per-GPU batch size (adjust for A100 24GB)
cfg.num_workers = 2
cfg.dali = False

cfg.num_epoch = 20               # Fine-tuning epochs
cfg.warmup_epoch = 0             # No warmup for fine-tuning
cfg.optimizer = "sgd"            # SGD is standard for ArcFace
cfg.lr = 0.01                    # Conservative LR for fine-tuning
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

# ------------------------------------------------------------------
# ArcFace / PartialFC hyperparameters
# ------------------------------------------------------------------
cfg.embedding_size = 512         # ONNX backbone outputs 512-D embeddings

# CombinedMarginLoss margins [m1, m2, m3]
cfg.margin_list = [1.0, 0.5, 0.0]

cfg.interclass_filtering_threshold = 0.0
cfg.sample_rate = 1.0            # Use all classes (100k is manageable)

# ------------------------------------------------------------------
# Logging / checkpointing
# ------------------------------------------------------------------
cfg.frequent = 20                # Log loss every N steps
cfg.verbose = 2000               # Verification callback every N steps
cfg.gradient_acc = 1             # Gradient accumulation

cfg.save_all_states = True
cfg.save_interval = 5000         # Checkpoint every N steps

# ------------------------------------------------------------------
# Network architecture
# ------------------------------------------------------------------
# This is ignored when using --onnx-backbone but kept for logging
cfg.network = "nvr_v7_onnx"
