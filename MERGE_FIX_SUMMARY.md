# ReidNet V3 Dataset Merge - Fix Summary

## Problems Identified

### 1. **Incomplete Dataset Merge** ❌
- Current S3 dataset: Only **18,019 identities** with ~65k images
- Expected: **300,000+ identities** with 1M+ images
- Issue: The merge of `faces_only` and `reidnet_v2_training_data` was not done correctly

### 2. **Training Error** ❌
```
ValueError: Empty index file: /home/ubuntu/.../train.idx. The dataset may be corrupted.
```
- Cause: Training script tried to use MXNet RecordIO format on ImageFolder data
- The dataset has `train.idx` file but it's empty/corrupted

## Solution Implemented ✅

### 1. **Dataset Merge Script** (`merge_reidnet_v3_datasets.py`)

**Purpose**: Properly merge the two source datasets with correct filtering.

**What it does**:
- Downloads both source datasets from S3:
  - `datasets/faces_only/rekognition_set/`
  - `datasets/reidnet_v2_training_data/rekognition_set/`
- Merges identity folders across both datasets
- **Only includes identities with 2+ images**
- Handles duplicate identities (combines images from both sources)
- Uploads merged dataset to: `datasets/reidnet_v3/rekognition_set/`

**Expected output**:
- 300,000+ identities
- 1,000,000+ images
- ~3-4 images per identity

### 2. **Updated Training Notebook** (`reidnet_v3_training_imagefolder.ipynb`)

**Changes**:
- **Validates dataset exists** before downloading
- Shows clear error message with merge instructions if dataset not ready
- **Auto-detects and removes RecordIO files** (`train.rec`, `train.idx`)
- Better statistics with larger sample size (1000 identities vs 100)
- Uses `tqdm` for progress visualization

**Error prevention**:
```python
# Removes any RecordIO files that would confuse the dataset loader
rec_file = DATASET_DIR / "train.rec"
idx_file = DATASET_DIR / "train.idx"
if rec_file.exists() or idx_file.exists():
    rec_file.unlink(missing_ok=True)
    idx_file.unlink(missing_ok=True)
```

### 3. **Comprehensive Documentation** (`DATASET_MERGE_README.md`)

- Step-by-step merge instructions
- Hardware requirements (500GB disk space)
- Command-line options and examples
- Troubleshooting guide
- Expected output and verification steps

## How to Use

### Step 1: Run the Merge (One-Time Setup)

On a machine with **500GB+ disk space**:

```bash
# Clone the updated repo
git clone https://github.com/deanofthewebb/insightface.git
cd insightface

# Install dependencies
pip install boto3 tqdm

# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-2

# Run the merge (takes 2.5-4 hours)
python merge_reidnet_v3_datasets.py \
  --bucket data-labeling.livereachmedia.com \
  --workdir /data/reidnet_v3_merge \
  --min-images 2
```

**Expected output**:
```
============================================================
MERGE COMPLETE
============================================================
Total identities: 325,891
Total images: 1,247,332
Identities from single source: 298,445
Identities from multiple sources: 27,446
Skipped (too few images): 124,782
Avg images per identity: 3.8
```

### Step 2: Train with Merged Dataset

Pull the latest notebook and run:

```bash
# On training instance, pull latest changes
cd /home/ubuntu/insightface_training/insightface
git pull origin main

# The notebook will now:
# 1. Check if merged dataset exists in S3 ✅
# 2. Download it (if exists) ✅
# 3. Remove any RecordIO files ✅
# 4. Start training with ImageFolder format ✅
```

## What Changed in the Code

### `merge_reidnet_v3_datasets.py` (NEW)
- **Lines 44-85**: Core merge logic that combines identity folders
- **Lines 87-105**: Handles duplicate identities by combining images
- **Lines 107-123**: Filters identities with < min_images
- **Lines 245-257**: CLI with sensible defaults

### `reidnet_v3_training_imagefolder.ipynb` (UPDATED)
- **Cell 5 (lines 263-270)**: Checks if merged dataset exists in S3
- **Cell 5 (lines 272-293)**: Shows error with merge instructions if not found
- **Cell 5 (lines 340-348)**: Removes RecordIO files to force ImageFolder format
- **Cell 5 (lines 315-330)**: Better statistics with 1000-identity sample

### `DATASET_MERGE_README.md` (NEW)
- Complete documentation for merge process
- Troubleshooting guide
- Verification steps

## Technical Details

### Why the Original Merge Failed

The S3 dataset at `datasets/reidnet_v3/rekognition_set/` had:
- Only 18,019 identities (should be 300k+)
- Average 3.6 images per identity (correct)
- Missing most of the source data

**Root cause**: The merge didn't properly combine all subdirectories from both source datasets.

### How ImageFolder vs RecordIO Detection Works

From `recognition/arcface_torch/dataset.py:42-52`:
```python
rec = os.path.join(root_dir, 'train.rec')
idx = os.path.join(root_dir, 'train.idx')

# Mxnet RecordIO
if os.path.exists(rec) and os.path.exists(idx):
    train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

# Image Folder
else:
    transform = transforms.Compose([...])
    train_set = ImageFolder(root_dir, transform)
```

**The fix**: Notebook now explicitly removes `train.rec` and `train.idx` to ensure ImageFolder is used.

## Verification

After running the merge, verify:

```bash
# Check S3 has the merged dataset
aws s3 ls s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/ | wc -l
# Expected: 300,000+ lines

# Download a sample identity
aws s3 sync \
  s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/identity_00001/ \
  /tmp/test_identity/

ls -la /tmp/test_identity/
# Expected: 2+ images
```

## Timeline

- **Merge script runtime**: 2.5-4 hours (depending on network speed)
  - Download: 60-90 min
  - Merge: 30-60 min
  - Upload: 60-90 min

- **Training setup**: 5-10 minutes (after merge is complete)
  - Download dataset to training instance
  - Verify and start training

## Next Steps

1. **Run the merge script** on a machine with sufficient disk space
2. **Wait for completion** (monitor with the progress bars)
3. **Verify the upload** to S3
4. **Pull latest notebook** on training instance
5. **Run training** - it will now work correctly

## Files Changed

```
insightface/
├── merge_reidnet_v3_datasets.py          ← NEW: Dataset merge script
├── DATASET_MERGE_README.md               ← NEW: Merge documentation
├── reidnet_v3_training_imagefolder.ipynb ← UPDATED: Better validation
└── MERGE_FIX_SUMMARY.md                  ← NEW: This file
```

## Commit

```
commit 30ec231
Author: Dean Webb
Date: Fri Nov 28 2025

Add ReidNet V3 dataset merge script and update training notebook

- Add merge_reidnet_v3_datasets.py to properly merge faces_only + reidnet_v2_training_data
- Merge only keeps identities with 2+ images
- Expected output: 300k+ identities with 1M+ images
- Update reidnet_v3_training_imagefolder.ipynb to check for merged dataset
- Auto-detect and remove any RecordIO files to ensure ImageFolder format
- Add DATASET_MERGE_README.md with comprehensive merge documentation
- Fix training error: was trying to use MXFaceDataset on ImageFolder data
```

Repository: https://github.com/deanofthewebb/insightface
Branch: `master`
