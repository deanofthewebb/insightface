#!/usr/bin/env python3
"""
Merge ReidNet V3 datasets from local directories.

This script handles two different formats:
1. faces_only: Flat directory with files named <identity_id>_face.png
2. reidnet_v2_training_data: Already organized into identity folders

Merges both into ImageFolder format with identity folders containing 2+ images.
"""

import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
import shutil
import re


def parse_identity_from_filename(filename):
    """
    Extract identity ID from filename.
    Expected format: <identity_id>_face.png or <identity_id>.jpg
    """
    # Remove extension
    stem = Path(filename).stem
    
    # Try to extract identity ID (everything before _face or the whole name)
    if '_face' in stem:
        return stem.split('_face')[0]
    else:
        return stem


def organize_flat_directory(source_dir, temp_dir, min_images=2):
    """
    Organize a flat directory of images into identity folders.
    
    Args:
        source_dir: Directory containing flat list of images
        temp_dir: Temporary directory to organize images
        min_images: Minimum images per identity
    
    Returns:
        Dict with statistics
    """
    source_dir = Path(source_dir)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Organizing flat directory: {source_dir.name}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in source_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"   Found {len(image_files):,} image files")
    
    # Group by identity
    identity_map = defaultdict(list)
    print("   Grouping by identity...")
    for img_file in tqdm(image_files, desc="   Parsing filenames", leave=False):
        identity_id = parse_identity_from_filename(img_file.name)
        identity_map[identity_id].append(img_file)
    
    print(f"   Found {len(identity_map):,} unique identities")
    
    # Create identity folders
    stats = {
        'identities': 0,
        'images': 0,
        'skipped': 0
    }
    
    print("   Creating identity folders...")
    for identity_id, images in tqdm(identity_map.items(), desc="   Organizing", leave=False):
        if len(images) < min_images:
            stats['skipped'] += 1
            continue
        
        identity_dir = temp_dir / identity_id
        identity_dir.mkdir(exist_ok=True)
        
        for img_file in images:
            dest = identity_dir / img_file.name
            shutil.copy2(img_file, dest)
            stats['images'] += 1
        
        stats['identities'] += 1
    
    return stats


def count_images_in_dir(directory):
    """Count image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sum(1 for f in directory.iterdir() 
               if f.is_file() and f.suffix.lower() in image_extensions)


def merge_identity_folders(source_dirs, output_dir, min_images=2):
    """
    Merge multiple directories containing identity folders.
    
    Args:
        source_dirs: List of source directories (each contains identity subfolders)
        output_dir: Output directory for merged dataset
        min_images: Minimum images per identity
    
    Returns:
        Dict with merge statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all identities across sources
    identity_map = defaultdict(list)
    
    print("\n📊 Scanning organized datasets...")
    for source_dir in source_dirs:
        source_dir = Path(source_dir)
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


def upload_to_s3(local_dir, bucket, prefix):
    """Upload directory to S3."""
    cmd = [
        "aws", "s3", "sync",
        str(local_dir),
        f"s3://{bucket}/{prefix}",
        "--quiet"
    ]
    
    print(f"\n📤 Uploading to s3://{bucket}/{prefix}")
    print("   This may take 1-2 hours...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description='Merge ReidNet V3 datasets from local sources')
    parser.add_argument('--local-source', 
                        default='/Users/deanwebb/Development/nvr_face_ingestor/face_exports/reidnet_v3',
                        help='Local source directory containing faces_only and reidnet_v2_training_data')
    parser.add_argument('--output', required=True,
                        help='Output directory for merged dataset')
    parser.add_argument('--bucket', default='data-labeling.livereachmedia.com',
                        help='S3 bucket name (for upload)')
    parser.add_argument('--min-images', type=int, default=2,
                        help='Minimum images per identity (default: 2)')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip uploading to S3')
    parser.add_argument('--temp-dir', default='/tmp/reidnet_v3_organize',
                        help='Temporary directory for organizing flat images')
    args = parser.parse_args()
    
    local_source = Path(args.local_source)
    output_dir = Path(args.output)
    temp_dir = Path(args.temp_dir)
    
    print("=" * 80)
    print("REIDNET V3 LOCAL DATASET MERGE")
    print("=" * 80)
    print(f"Local source: {local_source}")
    print(f"Output: {output_dir}")
    print(f"Minimum images per identity: {args.min_images}")
    print("=" * 80)
    
    # Source paths
    faces_only = local_source / 'faces_only'
    reidnet_v2 = local_source / 'reidnet_v2_training_data'
    
    if not faces_only.exists():
        print(f"\n❌ Error: faces_only not found at {faces_only}")
        sys.exit(1)
    
    if not reidnet_v2.exists():
        print(f"\n❌ Error: reidnet_v2_training_data not found at {reidnet_v2}")
        sys.exit(1)
    
    # Step 1: Organize faces_only (flat directory)
    organized_faces_only = temp_dir / 'faces_only_organized'
    print("\n" + "=" * 80)
    print("STEP 1: ORGANIZE FACES_ONLY")
    print("=" * 80)
    stats_faces = organize_flat_directory(faces_only, organized_faces_only, args.min_images)
    print(f"\n✅ Organized faces_only:")
    print(f"   Identities: {stats_faces['identities']:,}")
    print(f"   Images: {stats_faces['images']:,}")
    print(f"   Skipped (< {args.min_images} images): {stats_faces['skipped']:,}")
    
    # Step 2: Merge both datasets
    print("\n" + "=" * 80)
    print("STEP 2: MERGE DATASETS")
    print("=" * 80)
    
    source_dirs = [organized_faces_only, reidnet_v2]
    stats_merge = merge_identity_folders(source_dirs, output_dir, args.min_images)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Total identities: {stats_merge['total_identities']:,}")
    print(f"Total images: {stats_merge['total_images']:,}")
    print(f"Identities from single source: {stats_merge['single_source']:,}")
    print(f"Identities from multiple sources: {stats_merge['multi_source']:,}")
    print(f"Skipped (too few images): {stats_merge['skipped_too_few']:,}")
    if stats_merge['total_identities'] > 0:
        print(f"Avg images per identity: {stats_merge['total_images'] / stats_merge['total_identities']:.1f}")
    
    # Upload to S3
    if not args.skip_upload:
        print("\n" + "=" * 80)
        print("STEP 3: UPLOAD TO S3")
        print("=" * 80)
        
        output_prefix = 'datasets/reidnet_v3/rekognition_set/'
        upload_to_s3(output_dir, args.bucket, output_prefix)
        
        print(f"\n✅ Dataset uploaded to s3://{args.bucket}/{output_prefix}")
        print(f"   Total: {stats_merge['total_identities']:,} identities, {stats_merge['total_images']:,} images")
    else:
        print("\n⏭️  Skipping upload to S3")
        print(f"   Merged dataset at: {output_dir}")
    
    # Cleanup temp directory
    print("\n🧹 Cleaning up temporary files...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 80)
    print("✅ MERGE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
