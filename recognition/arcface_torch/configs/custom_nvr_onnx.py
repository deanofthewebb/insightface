# configs/custom_nvr_onnx.py
from easydict import EasyDict as edict

config = edict()

# ---- Backbone / loss ----
config.network = "r100"           # ignored when using --onnx-backbone, kept for logs
config.margin_list = (1.0, 0.5, 0.0)
config.embedding_size = 512       # ONNX ArcFace IR-100 outputs 512-D
config.sample_rate = 1.0          # 100k classes is fine without sampling
config.interclass_filtering_threshold = 0.0
config.fp16 = True

config.resume = False
config.output = "work_dirs/custom_nvr_onnx_100k"

# ---- Data ----
config.rec = "/path/to/your_custom_rec"   # <<< TODO: set this

# You said ~100k identities:
config.num_classes = 100_000              # <<< ~100k IDs

# Rough guess: ~10 images/ID → ~1M images; adjust if you know exact count.
config.num_image = 1_000_000              # <<< TODO: set to your actual image count

config.num_workers = 4
config.dali = False

# ---- Batch / optimizer ----
# Per-GPU batch size. With 4 GPUs this gives global batch = 512.
# For 1 GPU with 16–24 GB, 128 is usually safe; drop to 64 if you hit OOM.
config.batch_size = 128

config.optimizer = "sgd"      # official ArcFace uses SGD + 0.1 LR
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# ---- Training length ----
# 20–30 epochs is typical for fine-tuning on a new dataset at this scale.
config.num_epoch = 24
config.warmup_epoch = 0       # often fine to skip warmup for fine-tuning

# ---- Evaluation ----
# If you don't have LFW/CFP-FP/AgeDB recs set up, keep this empty.
config.val_targets = []       # e.g., ['lfw', 'cfp_fp', 'agedb_30']

# ---- Logging / saving ----
config.verbose = 2000         # run verification callback every N steps (ignored if val_targets=[])
config.frequent = 20          # log training loss every N steps
config.save_all_states = True
config.save_interval = 20_000 # steps between full checkpoints
config.gradient_acc = 1       # set >1 if you want virtual larger batch

config.seed = 2048
