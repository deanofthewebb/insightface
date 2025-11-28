from easydict import EasyDict as edict

config = edict()

config.network = "r100"
config.embedding_size = 512

config.margin_list = (1.0, 0.5, 0.0)
config.interclass_filtering_threshold = 0.0

config.output = "/home/ubuntu/checkpoints/reidnet_v3/work_dirs"
config.resume = False

config.rec = "/home/ubuntu/insightface_training/datasets/reidnet_v3_train"
config.num_classes = 100_000
config.num_image = 1_000_000
config.num_workers = 8
config.dali = False

config.batch_size = 256

config.lr = 0.01
config.optimizer = "sgd"
config.momentum = 0.9
config.weight_decay = 1e-4

config.sample_rate = 1.0
config.fp16 = True

config.num_epoch = 20
config.warmup_epoch = 0

config.verbose = 2000
config.frequent = 20
config.save_all_states = True
config.save_interval = 4000
config.gradient_acc = 1

config.val_targets = []

config.seed = 2048
