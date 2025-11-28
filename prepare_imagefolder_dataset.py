#!/usr/bin/env python3
"""
Prepare InsightFace ImageFolder dataset from raw identity directories.
- Resize all images to 112x112 (InsightFace standard)
- Convert to RGB
- Maintain identity directory structure
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse


def process_image(img_path, output_dir, target_size=(112, 112)):
    """
    Process a single image: resize to target_size and save.
    
    Args:
        img_path: Path to input image
        output_dir: Output directory for processed image
        target_size: Target image size (width, height)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Open image
        img = Image.open(img_path)
        
        # Convert to RGB (handle grayscale, RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size with high-quality resampling
        img = img.resize(target_size, Image.LANCZOS)
        
        # Save to output directory with same filename
        output_path = output_dir / img_path.name
        img.save(output_path, 'JPEG', quality=95)
        
        return True, None
    except Exception as e:
        return False, f"Error processing {img_path}: {e}"


def process_identity(identity_dir, input_root, output_root, target_size=(112, 112)):
    """
    Process all images for a single identity.
    
    Args:
        identity_dir: Path to identity directory
        input_root: Root input directory
        output_root: Root output directory
        target_size: Target image size
    
    Returns:
        Dict with statistics
    """
    identity_name = identity_dir.name
    output_identity_dir = output_root / identity_name
    output_identity_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in identity_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        return {
            'identity': identity_name,
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': []
        }
    
    # Process all images
    success_count = 0
    errors = []
    
    for img_path in image_files:
        success, error = process_image(img_path, output_identity_dir, target_size)
        if success:
            success_count += 1
        else:
            errors.append(error)
    
    return {
        'identity': identity_name,
        'total': len(image_files),
        'success': success_count,
        'failed': len(errors),
        'errors': errors[:5]  # Keep first 5 errors
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare InsightFace ImageFolder dataset')
    parser.add_argument('--input', required=True, help='Input directory with identity subdirectories')
    parser.add_argument('--output', required=True, help='Output directory for processed dataset')
    parser.add_argument('--size', type=int, default=112, help='Target image size (default: 112x112)')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    target_size = (args.size, args.size)
    
    if not input_root.exists():
        print(f"Error: Input directory does not exist: {input_root}")
        sys.exit(1)
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Find all identity directories
    identity_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    
    print("=" * 80)
    print("INSIGHTFACE IMAGEFOLDER DATASET PREPARATION")
    print("=" * 80)
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Identities: {len(identity_dirs):,}")
    print(f"Workers: {args.workers}")
    print("=" * 80)
    print()
    
    # Process all identities in parallel
    process_func = partial(process_identity, 
                          input_root=input_root, 
                          output_root=output_root,
                          target_size=target_size)
    
    results = []
    with mp.Pool(args.workers) as pool:
        with tqdm(total=len(identity_dirs), desc="Processing identities", unit="identity") as pbar:
            for result in pool.imap_unordered(process_func, identity_dirs):
                results.append(result)
                pbar.update(1)
    
    # Calculate statistics
    total_identities = len(results)
    total_images = sum(r['total'] for r in results)
    total_success = sum(r['success'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    identities_with_images = sum(1 for r in results if r['success'] > 0)
    
    print()
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total identities processed: {total_identities:,}")
    print(f"Identities with images: {identities_with_images:,}")
    print(f"Total images processed: {total_images:,}")
    print(f"Successfully converted: {total_success:,}")
    print(f"Failed: {total_failed:,}")
    print(f"Success rate: {100.0 * total_success / total_images:.2f}%" if total_images > 0 else "N/A")
    
    # Show identities with most images
    top_identities = sorted(results, key=lambda x: x['success'], reverse=True)[:10]
    print()
    print("Top 10 identities by image count:")
    for i, r in enumerate(top_identities, 1):
        print(f"  {i:2d}. {r['identity']}: {r['success']:,} images")
    
    # Show failed identities
    failed_identities = [r for r in results if r['failed'] > 0]
    if failed_identities:
        print()
        print(f"Identities with errors: {len(failed_identities)}")
        print("First 5 identities with errors:")
        for r in failed_identities[:5]:
            print(f"  - {r['identity']}: {r['failed']} failed")
            for error in r['errors']:
                print(f"    {error}")
    
    print()
    print(f"Dataset ready at: {output_root}")
    print("=" * 80)


if __name__ == '__main__':
    main()
