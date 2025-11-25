#!/usr/bin/env python3
"""
make_rec.py

Convert a directory of images arranged as:

    /path/to/images/
        id000001/
            img1.jpg
            img2.jpg
            ...
        id000002/
            ...
        ...

into an InsightFace-style MXNet RecordIO dataset:

    /data/nvr_faces_rec/
        train.rec
        train.idx
        property

This is compatible with the dataset format expected by InsightFace / arcface_torch
(i.e., MXFaceDataset reading train.rec/train.idx).

Dependencies:
    pip install mxnet pillow tqdm
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import mxnet as mx
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(x, *args, **kwargs):
        return x


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_classes(root: Path) -> Tuple[List[str], dict]:
    """
    Find class subdirectories and assign integer labels.

    Returns:
        classes: sorted list of class folder names
        class_to_idx: mapping from class name to integer label
    """
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_image_list(root: Path, class_to_idx: dict) -> List[Tuple[str, int]]:
    """
    Collect all (image_path, label) pairs from the directory tree.
    """
    samples = []
    for cls_name, label in class_to_idx.items():
        cls_dir = root / cls_name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                samples.append((str(p), int(label)))
    return samples


def write_recordio(
    samples: List[Tuple[str, int]],
    output_dir: Path,
    img_size: int = 112,
    quality: int = 95,
) -> None:
    """
    Write samples to MXNet IndexedRecordIO: train.rec + train.idx
    and create a 'property' file.

    Args:
        samples: list of (image_path, label)
        output_dir: directory to create train.rec/train.idx/property
        img_size: resize images to (img_size, img_size)
        quality: JPEG quality for pack_img
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_path = output_dir / "train.rec"
    idx_path = output_dir / "train.idx"

    print(f"[INFO] Writing RecordIO to: {rec_path}")
    print(f"[INFO] Index file:         {idx_path}")

    writer = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "w")

    num_written = 0
    for idx, (img_path, label) in enumerate(tqdm(samples, desc="Packing images")):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}", file=sys.stderr)
            continue

        if img_size is not None:
            img = img.resize((img_size, img_size), Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)  # HWC, RGB

        # MXNet IRHeader: flag, label, id, id2
        # Use float(label) as in original InsightFace pipeline.
        header = mx.recordio.IRHeader(
            flag=0,
            label=float(label),
            id=idx,
            id2=0,
        )
        packed = mx.recordio.pack_img(
            header,
            img_np,
            quality=quality,
            img_fmt=".jpg",
        )
        writer.write_idx(idx, packed)
        num_written += 1

    print(f"[INFO] Wrote {num_written} RecordIO entries.")

    # Create property file: "<num_classes>,<height>,<width>"
    labels = [label for _, label in samples]
    num_classes = len(set(labels))
    prop_path = output_dir / "property"
    with open(prop_path, "w") as f:
        f.write(f"{num_classes},{img_size},{img_size}\n")

    print(f"[INFO] Property file written to: {prop_path}")
    print(f"[INFO] num_classes={num_classes}, image_size={img_size}x{img_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image directory into an InsightFace-style MXNet RecordIO dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory of images (subfolders per identity).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/nvr_faces_rec",
        help="Output directory for train.rec/train.idx/property.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=112,
        help="Resize images to (img_size, img_size). Default: 112.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sample order before writing RecordIO.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"[ERROR] Input dir does not exist or is not a directory: {input_root}")

    print(f"[INFO] Scanning input directory: {input_root}")
    classes, class_to_idx = find_classes(input_root)
    if not classes:
        raise SystemExit(f"[ERROR] No class subdirectories found in {input_root}")

    print(f"[INFO] Found {len(classes)} classes.")
    samples = make_image_list(input_root, class_to_idx)
    print(f"[INFO] Found {len(samples)} images.")

    if args.shuffle:
        import random

        random.shuffle(samples)
        print("[INFO] Shuffled sample order.")

    write_recordio(samples, output_root, img_size=args.img_size)


if __name__ == "__main__":
    main()
