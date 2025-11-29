#!/usr/bin/env python3
"""
Merge ReidNet V3 datasets from multiple S3 sources.

This script:
1. Downloads subdirectories from faces_only and reidnet_v2_training_data
2. Merges identity folders with 2+ images
3. Uploads the merged dataset to S3
"""

import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
import shutil


def count_images_in_dir(directory):
    """Count image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sum(1 for f in directory.iterdir() 
               if f.is_file() and f.suffix.lower() in image_extensions)


def sync_from_s3(bucket, prefix, local_dir, quiet=True):
    """Download directory from S3."""
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix}",
        str(local_dir)
    ]
    if quiet:
        cmd.append("--quiet")
    
    print(f"📥 Downloading from s3://{bucket}/{prefix}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"S3 sync failed: {result.stderr}")
    
    return local_dir


def merge_datasets(source_dirs, output_dir, min_images=2):
    """
    Merge multiple ImageFolder datasets.
    
    Args:
        source_dirs: List of source dataset directories
        output_dir: Output directory for merged dataset
        min_images: Minimum images per identity to include
    
    Returns:
        Dict with merge statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all identities across source datasets
    identity_map = defaultdict(list)
    
    print("\n📊 Scanning source datasets...")
    for source_dir in source_dirs:
        print(f"   - {source_dir.name}")
        identity_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
        
        for identity_dir in tqdm(identity_dirs, desc=f"  Scanning {source_dir.name}", leave=False):
            image_count = count_images_in_dir(identity_dir)
            if image_count >= min_images:
                identity_map[identity_dir.name].append((identity_dir, image_count))
    
    print(f"\n✅ Found {len(identity_map):,} unique identities across all sources")
    
    # Merge identities
    print(f"\n🔄 Merging identities (min {min_images} images)...")
    
    stats = {
        'total_identities': 0,
        'total_images': 0,
        'single_source': 0,
        'multi_source': 0,
        'skipped_too_few': 0
    }
    
    for identity_name, sources in tqdm(identity_map.items(), desc="Merging identities"):
        output_identity_dir = output_dir / identity_name
        output_identity_dir.mkdir(exist_ok=True)
        
        total_images = 0
        
        # Copy images from all sources for this identity
        for source_dir, img_count in sources:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in source_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            for img_file in image_files:
                # Create unique filename if there's a collision
                dest_file = output_identity_dir / img_file.name
                if dest_file.exists():
                    # Add source directory name to make unique
                    stem = img_file.stem
                    suffix = img_file.suffix
                    dest_file = output_identity_dir / f"{stem}_{source_dir.parent.name}{suffix}"
                
                shutil.copy2(img_file, dest_file)
                total_images += 1
        
        # Verify minimum images requirement after merge
        if total_images >= min_images:
            stats['total_identities'] += 1
            stats['total_images'] += total_images
            
            if len(sources) == 1:
                stats['single_source'] += 1
            else:
                stats['multi_source'] += 1
        else:
            # Remove directory if below threshold
            shutil.rmtree(output_identity_dir)
            stats['skipped_too_few'] += 1
    
    return stats


def upload_to_s3(local_dir, bucket, prefix, quiet=True):
    """Upload directory to S3."""
    cmd = [
        "aws", "s3", "sync",
        str(local_dir),
        f"s3://{bucket}/{prefix}"
    ]
    if quiet:
        cmd.append("--quiet")
    
    print(f"\n📤 Uploading to s3://{bucket}/{prefix}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description='Merge ReidNet V3 datasets from S3 sources')
    parser.add_argument('--bucket', default='data-labeling.livereachmedia.com',
                        help='S3 bucket name')
    parser.add_argument('--workdir', default='/tmp/reidnet_v3_merge',
                        help='Local working directory')
    parser.add_argument('--min-images', type=int, default=2,
                        help='Minimum images per identity (default: 2)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading from S3 (use existing local data)')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip uploading to S3 (for testing)')
    args = parser.parse_args()
    
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("REIDNET V3 DATASET MERGE")
    print("=" * 80)
    print(f"Bucket: {args.bucket}")
    print(f"Working directory: {workdir}")
    print(f"Minimum images per identity: {args.min_images}")
    print("=" * 80)
    
    # Source datasets configuration
    sources = [
        {
            'name': 'faces_only',
            'prefix': 'datasets/faces_only/rekognition_set/',
            'local_dir': workdir / 'sources' / 'faces_only'
        },
        {
            'name': 'reidnet_v2_training_data',
            'prefix': 'datasets/reidnet_v2_training_data/rekognition_set/',
            'local_dir': workdir / 'sources' / 'reidnet_v2'
        }
    ]
    
    # Download source datasets
    if not args.skip_download:
        print("\n📥 DOWNLOADING SOURCE DATASETS")
        print("=" * 80)
        for source in sources:
            sync_from_s3(args.bucket, source['prefix'], source['local_dir'])
            
            # Count identities
            identity_dirs = [d for d in source['local_dir'].iterdir() if d.is_dir()]
            print(f"   ✅ {source['name']}: {len(identity_dirs):,} identities")
    else:
        print("\n⏭️  Skipping download (using existing local data)")
    
    # Merge datasets
    print("\n" + "=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)
    
    output_dir = workdir / 'merged' / 'rekognition_set'
    source_dirs = [s['local_dir'] for s in sources]
    
    stats = merge_datasets(source_dirs, output_dir, min_images=args.min_images)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Total identities: {stats['total_identities']:,}")
    print(f"Total images: {stats['total_images']:,}")
    print(f"Identities from single source: {stats['single_source']:,}")
    print(f"Identities from multiple sources: {stats['multi_source']:,}")
    print(f"Skipped (too few images): {stats['skipped_too_few']:,}")
    print(f"Avg images per identity: {stats['total_images'] / stats['total_identities']:.1f}")
    
    # Upload to S3
    if not args.skip_upload:
        print("\n" + "=" * 80)
        print("UPLOADING TO S3")
        print("=" * 80)
        
        output_prefix = 'datasets/reidnet_v3/rekognition_set/'
        upload_to_s3(output_dir, args.bucket, output_prefix)
        
        print(f"\n✅ Dataset uploaded to s3://{args.bucket}/{output_prefix}")
        print(f"   Total: {stats['total_identities']:,} identities, {stats['total_images']:,} images")
    else:
        print("\n⏭️  Skipping upload to S3")
        print(f"   Merged dataset at: {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ MERGE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
