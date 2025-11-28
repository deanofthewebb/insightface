# configs/reidnet_v3_finetune.py
# Fine-tuning ReidNet V3 using ONNX backbone from nvr.prod.v7
from easydict import EasyDict as edict

config = edict()

# ------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------
config.output = "/home/ubuntu/checkpoints/reidnet_v3/work_dirs"
config.resume = False
config.fp16 = True
config.seed = 2048

# ------------------------------------------------------------------
# Dataset (RecordIO format)
# ------------------------------------------------------------------
config.rec = "/home/ubuntu/insightface_training/datasets/reidnet_v3_rec"

# Dataset statistics - update these based on your actual data
config.num_classes = 100_000        # Number of identities
config.num_image = 1_000_000        # Total training images

# Validation targets (empty if no validation recs available)
config.val_targets = []

# ------------------------------------------------------------------
# Training schedule
# ------------------------------------------------------------------
config.batch_size = 128             # Per-GPU batch size (adjust for A100 24GB)
config.num_workers = 2
config.dali = False
config.dali_aug = False

config.num_epoch = 20               # Fine-tuning epochs
config.warmup_epoch = 0             # No warmup for fine-tuning
config.optimizer = "sgd"            # SGD is standard for ArcFace
config.lr = 0.01                    # Conservative LR for fine-tuning
config.weight_decay = 5e-4
config.momentum = 0.9

# ------------------------------------------------------------------
# ArcFace / PartialFC hyperparameters
# ------------------------------------------------------------------
config.embedding_size = 512         # ONNX backbone outputs 512-D embeddings

# CombinedMarginLoss margins [m1, m2, m3]
config.margin_list = [1.0, 0.5, 0.0]

config.interclass_filtering_threshold = 0.0
config.sample_rate = 1.0            # Use all classes (100k is manageable)

# ------------------------------------------------------------------
# Logging / checkpointing
# ------------------------------------------------------------------
config.frequent = 20                # Log loss every N steps
config.verbose = 2000               # Verification callback every N steps
config.gradient_acc = 1             # Gradient accumulation

config.save_all_states = True
config.save_interval = 5000         # Checkpoint every N steps

# ------------------------------------------------------------------
# Network architecture
# ------------------------------------------------------------------
# This is ignored when using --onnx-backbone but kept for logging
config.network = "nvr_v7_onnx"
