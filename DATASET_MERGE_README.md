# ReidNet V3 Dataset Merge Guide

## Overview

The ReidNet V3 training dataset is created by merging two source datasets:
- **faces_only**: Original face dataset
- **reidnet_v2_training_data**: Additional training data from v2

This merge process combines identity folders from both sources, keeping only identities with **2 or more images**.

## Expected Results

After merging:
- **300,000+** unique identities
- **1,000,000+** total images
- Average **3-4 images per identity**

## Prerequisites

### Storage Requirements
- **300GB** temporary disk space for downloading source datasets
- **200GB** for merged dataset
- Total: ~500GB recommended

### Software Requirements
- Python 3.8+
- AWS CLI configured with credentials
- Required packages: `boto3`, `tqdm`

```bash
pip install boto3 tqdm
```

### AWS Credentials
Set up AWS credentials for S3 access:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-2
```

Or configure via `aws configure`

## Running the Merge

### Basic Usage

```bash
python merge_reidnet_v3_datasets.py \
  --bucket data-labeling.livereachmedia.com \
  --workdir /data/reidnet_v3_merge \
  --min-images 2
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bucket` | `data-labeling.livereachmedia.com` | S3 bucket name |
| `--workdir` | `/tmp/reidnet_v3_merge` | Local working directory |
| `--min-images` | `2` | Minimum images per identity |
| `--skip-download` | `false` | Skip downloading (use existing local data) |
| `--skip-upload` | `false` | Skip uploading to S3 (for testing) |

### Example: Testing Locally

```bash
# Download and merge, but don't upload
python merge_reidnet_v3_datasets.py \
  --workdir /data/merge_test \
  --min-images 2 \
  --skip-upload
```

### Example: Resume After Download

```bash
# If download completed but upload failed
python merge_reidnet_v3_datasets.py \
  --workdir /data/reidnet_v3_merge \
  --skip-download
```

## Process Steps

The script performs these steps automatically:

### 1. Download Source Datasets (60-90 minutes)
Downloads from S3:
- `s3://data-labeling.livereachmedia.com/datasets/faces_only/rekognition_set/`
- `s3://data-labeling.livereachmedia.com/datasets/reidnet_v2_training_data/rekognition_set/`

### 2. Scan and Merge (30-60 minutes)
- Scans all identity directories across both sources
- Merges identities with same ID from different sources
- Filters out identities with fewer than `--min-images` images
- Handles filename collisions automatically

### 3. Upload Merged Dataset (60-90 minutes)
Uploads to:
- `s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/`

**Total time: 2.5-4 hours**

## Output

```
============================================================
REIDNET V3 DATASET MERGE
============================================================
Bucket: data-labeling.livereachmedia.com
Working directory: /data/reidnet_v3_merge
Minimum images per identity: 2
============================================================

📥 DOWNLOADING SOURCE DATASETS
============================================================
📥 Downloading from s3://.../faces_only/rekognition_set/
   ✅ faces_only: 320,482 identities
📥 Downloading from s3://.../reidnet_v2_training_data/rekognition_set/
   ✅ reidnet_v2_training_data: 305,291 identities

============================================================
MERGING DATASETS
============================================================

📊 Scanning source datasets...
   - faces_only
   - reidnet_v2

✅ Found 450,673 unique identities across all sources

🔄 Merging identities (min 2 images)...
Processing: 100%|████████████████| 450673/450673

============================================================
MERGE COMPLETE
============================================================
Total identities: 325,891
Total images: 1,247,332
Identities from single source: 298,445
Identities from multiple sources: 27,446
Skipped (too few images): 124,782
Avg images per identity: 3.8

============================================================
UPLOADING TO S3
============================================================
📤 Uploading to s3://.../datasets/reidnet_v3/rekognition_set/

✅ Dataset uploaded to s3://.../datasets/reidnet_v3/rekognition_set/
   Total: 325,891 identities, 1,247,332 images

============================================================
✅ MERGE COMPLETE
============================================================
```

## Dataset Structure

The merged dataset uses **ImageFolder** format:

```
reidnet_v3/rekognition_set/
├── identity_00001/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
├── identity_00002/
│   ├── img_001.jpg
│   └── img_002.jpg
└── identity_NNNNN/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

Each identity folder contains 2+ images (112x112 RGB JPEGs).

## Troubleshooting

### Out of Disk Space

Monitor disk usage:
```bash
df -h /data
```

Clean up after successful upload:
```bash
rm -rf /data/reidnet_v3_merge/sources
```

### S3 Download/Upload Failures

The script will show specific AWS CLI errors. Common issues:
- **Credentials expired**: Re-configure AWS credentials
- **Network timeout**: Retry with same `--workdir` (AWS CLI resumes)
- **Permission denied**: Verify IAM permissions for S3 bucket

### Memory Issues

The script processes identities one at a time, so memory usage should be minimal (~500MB). If you encounter issues, check:
- Available system memory: `free -h`
- Python memory usage: Add `--workers 1` if using parallel processing

### Corrupted Dataset

If the merge produces unexpected results:
```bash
# Clean working directory and restart
rm -rf /data/reidnet_v3_merge
python merge_reidnet_v3_datasets.py --workdir /data/reidnet_v3_merge
```

## Verification

After merge completion, verify the dataset:

### Check S3 Upload
```bash
aws s3 ls s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/ | head -20
```

### Count Identities
```bash
aws s3 ls s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/ | wc -l
```

Expected: 300,000+ lines

### Sample Identity
```bash
# Download a sample identity to verify image count
aws s3 sync \
  s3://data-labeling.livereachmedia.com/datasets/reidnet_v3/rekognition_set/identity_00001/ \
  /tmp/sample_identity/

ls -la /tmp/sample_identity/
```

Each identity should have 2+ images.

## Integration with Training

Once the merged dataset is uploaded to S3, the training notebook will automatically:

1. Detect the merged dataset in S3
2. Download it to the training instance
3. Verify ImageFolder format (no RecordIO files)
4. Start training with correct dataset loader

If the merged dataset is **not** found, the notebook will show instructions to run this script.

## Notes

- The merge is **idempotent**: Running multiple times produces the same result
- Source datasets are **not modified**: Only reads from S3
- Filename collisions are handled by appending source name
- Progress is saved: Can resume if interrupted (use `--skip-download`)

## Support

For issues with:
- **Merge script**: Check this README and error messages
- **Training**: See `TRAINING_SETUP.md` and notebook comments
- **Dataset format**: See `prepare_imagefolder_dataset.py` for preprocessing
