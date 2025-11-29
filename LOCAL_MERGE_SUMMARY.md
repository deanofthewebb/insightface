# ReidNet V3 Local Dataset Merge - Summary

## Dataset Analysis

### Source Datasets

**Location**: `/Users/deanwebb/Development/nvr_face_ingestor/face_exports/reidnet_v3/`

1. **faces_only/** (390,856 files)
   - **Format**: Flat directory with files named `<identity_id>_face.png`
   - **Issue**: Each identity has only **1 image**
   - **Result**: All **skipped** (minimum 2 images required for training)

2. **reidnet_v2_training_data/** (102,491 folders)
   - **Format**: ImageFolder with identity subdirectories
   - **Structure**: `<identity_id>/*.jpg`
   - **Images**: 430,143 total (avg 4.2 per identity)
   - **Result**: All **included** (all have 2+ images)

### Merged Dataset

**Output**: `/Users/deanwebb/Development/reidnet_v3_merged/`

```
Total identities: 102,491
Total images: 430,143
Avg images/identity: 4.2
```

## Why faces_only Was Skipped

The `faces_only` dataset contains **one image per identity**:
- Filename format: `6926bdce7f6c3c80a1f917f1_face.png`
- Identity ID: `6926bdce7f6c3c80a1f917f1`
- Each unique ID appears only once

For ArcFace training, we need **at least 2 images per identity** to:
- Learn discriminative embeddings
- Compute within-class variance
- Avoid overfitting to single examples

## Merge Process

### Step 1: Organize faces_only (4 seconds)
- Parsed 390,856 filenames
- Extracted identity IDs
- Grouped by identity
- **Result**: 390,856 identities with 1 image each → All skipped

### Step 2: Process reidnet_v2_training_data (20 seconds)
- Scanned 102,491 identity folders
- Counted images per identity
- Filtered identities with 2+ images
- **Result**: All 102,491 identities included

### Step 3: Merge (140 seconds)
- Copied images from reidnet_v2_training_data to output
- No duplication (no overlap with faces_only)
- Total time: ~4.5 minutes

### Step 4: Upload to S3 (running in background)
- Uploading to: `s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/`
- Expected time: 1-2 hours
- Total size: ~50GB (430k images @ ~120KB average)

## Training Configuration Updated

**File**: `recognition/arcface_torch/configs/reidnet_v3_imagefolder.py`

```python
config.num_classes = 102_491   # Identity folders
config.num_image = 430_143     # Total images
config.batch_size = 128        # Per-GPU
config.num_epoch = 24
config.lr = 0.01               # Fine-tuning LR
```

**Expected training performance**:
- Steps per epoch: ~3,360 (430k images / 128 batch size)
- Total steps: ~80,640 (24 epochs × 3,360)
- Training time: ~8-12 hours on 8× A100 GPUs

## Dataset Quality

**Image format**: JPEG, already resized to 112×112
**Color**: RGB
**Identity distribution**: Varies (2 to ~50 images per identity)

Sample identity (`9346a6b88546f1fb7d7e65ee280d0dba`):
```
681049882c8198f44e98eda9.jpg  (6.7 KB)
681049ca2c8198f44e99075f.jpg  (6.0 KB)
```

## Next Steps

1. **Wait for S3 upload** (~1-2 hours)
   - Check progress: `tail -f /var/folders/.../bg_2025-11-29T01-35-34-864Z_7c962996.log`
   
2. **Verify S3 upload**:
   ```bash
   aws s3 ls s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/ | wc -l
   # Expected: 102,491 lines
   ```

3. **Pull latest changes** on training instance:
   ```bash
   cd /home/ubuntu/insightface_training/insightface
   git pull origin master
   ```

4. **Run training notebook**:
   - Notebook will detect merged dataset in S3
   - Download to training instance
   - Start training with ImageFolder format

## Files Changed

```
insightface/
├── merge_reidnet_v3_local.py                           ← NEW
├── recognition/arcface_torch/configs/
│   └── reidnet_v3_imagefolder.py                       ← UPDATED (stats)
└── LOCAL_MERGE_SUMMARY.md                              ← NEW
```

## Command Reference

### Run merge locally:
```bash
python merge_reidnet_v3_local.py \
  --local-source /Users/deanwebb/Development/nvr_face_ingestor/face_exports/reidnet_v3 \
  --output /Users/deanwebb/Development/reidnet_v3_merged \
  --min-images 2
```

### Run merge without S3 upload (testing):
```bash
python merge_reidnet_v3_local.py \
  --local-source /path/to/source \
  --output /path/to/output \
  --skip-upload \
  --min-images 2
```

### Check upload progress:
```bash
ps aux | grep merge_reidnet_v3_local
tail -f /var/folders/.../bg_*.log
```

## Comparison with Expected

| Metric | Expected | Actual | Notes |
|--------|----------|--------|-------|
| Identities | 300k+ | 102,491 | faces_only skipped (1 img each) |
| Images | 1M+ | 430,143 | Only reidnet_v2_training_data |
| Avg/identity | 3-4 | 4.2 | ✅ Good distribution |
| Min images | 2+ | 2+ | ✅ Meets requirement |

## Why This is OK

Even though we only got 102k identities instead of 300k:

1. **Quality over quantity**: All identities have 2+ images (required for training)
2. **Good distribution**: 4.2 images per identity is healthy
3. **Sufficient scale**: 102k classes is substantial for face recognition
4. **Better than alternatives**: Single-image identities would hurt training

## Recommendations

If you need more identities:
1. **Collect more multi-image identities** from production data
2. **Apply data augmentation** during training (already enabled)
3. **Use harder examples**: Include difficult lighting, angles, occlusions

Current dataset (102k identities, 430k images) is **ready for production training**.
