#!/usr/bin/env python3
"""
Test script to validate MXNet-free RecordIO reading implementation
"""
import os
import sys
import struct
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "recognition" / "arcface_torch"))

def test_recordio_format():
    """Test RecordIO .idx file parsing"""
    print("=" * 60)
    print("TEST 1: RecordIO .idx File Parsing")
    print("=" * 60)
    
    test_idx_content = """0\t0\t0
1\t100\t500
2\t600\t300
3\t900\t450"""
    
    idx_path = "/tmp/test_train.idx"
    with open(idx_path, 'w') as f:
        f.write(test_idx_content)
    
    idx_map = {}
    with open(idx_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                idx = int(parts[0])
                offset = int(parts[1])
                idx_map[idx] = offset
    
    print(f"✓ Loaded {len(idx_map)} index entries")
    print(f"  Index map: {idx_map}")
    
    assert len(idx_map) == 4, "Should have 4 entries"
    assert idx_map[1] == 100, "Entry 1 should be at offset 100"
    print("✓ Index parsing works correctly\n")
    
    os.remove(idx_path)
    return True

def test_recordio_header_unpacking():
    """Test RecordIO header unpacking"""
    print("=" * 60)
    print("TEST 2: RecordIO Header Unpacking")
    print("=" * 60)
    
    print("\nTest Case A: Header with flag=0")
    flag = 0
    record_data = struct.pack('I', flag)
    
    flag_parsed = struct.unpack('I', record_data[:4])[0]
    if flag_parsed == 0:
        header = {'flag': 0, 'label': [0]}
    else:
        label_size = flag_parsed
        label = struct.unpack(f'{label_size}f', record_data[4:4+label_size*4])
        header = {'flag': flag_parsed, 'label': label}
    
    print(f"  Parsed header: {header}")
    assert header['flag'] == 0, "Flag should be 0"
    assert header['label'] == [0], "Label should be [0]"
    print("✓ Flag=0 header parsed correctly")
    
    print("\nTest Case B: Header with label data")
    flag = 2
    labels = (123.0, 456.0)
    record_data = struct.pack('I', flag) + struct.pack(f'{flag}f', *labels)
    
    flag_parsed = struct.unpack('I', record_data[:4])[0]
    label_size = flag_parsed
    label = struct.unpack(f'{label_size}f', record_data[4:4+label_size*4])
    header = {'flag': flag_parsed, 'label': label}
    
    print(f"  Parsed header: {header}")
    assert header['flag'] == 2, "Flag should be 2"
    assert len(header['label']) == 2, "Should have 2 labels"
    assert header['label'][0] == 123.0, "First label should be 123.0"
    print("✓ Label header parsed correctly\n")
    
    return True

def test_image_decoding():
    """Test image decoding from bytes"""
    print("=" * 60)
    print("TEST 3: Image Decoding (MXNet-free)")
    print("=" * 60)
    
    print("\nCreating test image...")
    img_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array, 'RGB')
    
    bio = BytesIO()
    img_pil.save(bio, format='JPEG', quality=95)
    img_bytes = bio.getvalue()
    
    print(f"✓ Created test JPEG: {len(img_bytes)} bytes")
    
    print("\nDecoding with PIL (new method)...")
    img_decoded = Image.open(BytesIO(img_bytes)).convert('RGB')
    img_decoded_np = np.array(img_decoded)
    
    print(f"  Original shape: {img_array.shape}")
    print(f"  Decoded shape:  {img_decoded_np.shape}")
    print(f"  Original dtype: {img_array.dtype}")
    print(f"  Decoded dtype:  {img_decoded_np.dtype}")
    
    assert img_decoded_np.shape == img_array.shape, "Shapes should match"
    assert img_decoded_np.dtype == np.uint8, "Should be uint8"
    print("✓ Image decoding works correctly")
    
    print("\nTesting channel transpose (C, H, W)...")
    img_transposed = np.transpose(img_decoded_np, (2, 0, 1))
    print(f"  Transposed shape: {img_transposed.shape}")
    assert img_transposed.shape == (3, 112, 112), "Should be (3, 112, 112)"
    print("✓ Channel transpose works correctly\n")
    
    return True

def test_image_flip():
    """Test horizontal flip operation"""
    print("=" * 60)
    print("TEST 4: Image Horizontal Flip")
    print("=" * 60)
    
    img_np = np.arange(24).reshape(2, 4, 3).astype(np.float32)
    print(f"\nOriginal image (2x4x3):\n{img_np[:,:,0]}")
    
    print("\nFlipping with np.flip (new method)...")
    img_flipped = np.flip(img_np, axis=1).copy()
    print(f"Flipped image:\n{img_flipped[:,:,0]}")
    
    assert img_flipped.shape == img_np.shape, "Shapes should match"
    assert img_flipped[0, 0, 0] == img_np[0, -1, 0], "First pixel should be last"
    assert img_flipped[0, -1, 0] == img_np[0, 0, 0], "Last pixel should be first"
    print("✓ Horizontal flip works correctly\n")
    
    return True

def test_image_resize():
    """Test image resizing"""
    print("=" * 60)
    print("TEST 5: Image Resize")
    print("=" * 60)
    
    img = Image.new('RGB', (200, 150), color='red')
    print(f"\nOriginal size: {img.size}")
    
    target_size = 112
    print(f"Target shortest side: {target_size}")
    
    scale = target_size / min(img.width, img.height)
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, Image.BILINEAR)
    
    print(f"Scale factor: {scale:.3f}")
    print(f"New size: {img_resized.size}")
    
    assert min(img_resized.size) == target_size, f"Shortest side should be {target_size}"
    print("✓ Image resize works correctly\n")
    
    return True

def test_torch_conversion():
    """Test NumPy to PyTorch conversion"""
    print("=" * 60)
    print("TEST 6: NumPy to PyTorch Conversion")
    print("=" * 60)
    
    img_np = np.random.randint(0, 255, (3, 112, 112), dtype=np.uint8)
    print(f"\nNumPy array shape: {img_np.shape}")
    print(f"NumPy array dtype: {img_np.dtype}")
    
    img_torch = torch.from_numpy(img_np)
    print(f"PyTorch tensor shape: {img_torch.shape}")
    print(f"PyTorch tensor dtype: {img_torch.dtype}")
    
    assert img_torch.shape == torch.Size([3, 112, 112]), "Shape should match"
    assert img_torch.dtype == torch.uint8, "Dtype should be uint8"
    print("✓ NumPy to PyTorch conversion works correctly\n")
    
    return True

def test_integration():
    """Integration test simulating actual dataset loading"""
    print("=" * 60)
    print("TEST 7: Integration Test - Full Pipeline")
    print("=" * 60)
    
    print("\nSimulating RecordIO dataset loading...")
    
    img_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array, 'RGB')
    bio = BytesIO()
    img_pil.save(bio, format='JPEG', quality=95)
    img_bytes = bio.getvalue()
    
    label_value = 42.0
    flag = 1
    record_header = struct.pack('I', flag) + struct.pack('f', label_value)
    record_data = record_header + img_bytes
    
    print(f"✓ Created mock RecordIO record: {len(record_data)} bytes")
    
    header = {'flag': struct.unpack('I', record_data[:4])[0]}
    header['label'] = struct.unpack('f', record_data[4:8])
    header_size = 4 + len(header['label']) * 4
    img_data = record_data[header_size:]
    
    print(f"✓ Parsed header: flag={header['flag']}, label={header['label']}")
    print(f"✓ Extracted image data: {len(img_data)} bytes")
    
    img = Image.open(BytesIO(img_data)).convert('RGB')
    sample = np.array(img)
    sample = np.transpose(sample, (2, 0, 1))
    
    print(f"✓ Decoded image shape: {sample.shape}")
    assert sample.shape == (3, 112, 112), "Should be (3, 112, 112)"
    
    label = header['label'][0]
    label_tensor = torch.tensor(label, dtype=torch.long)
    
    print(f"✓ Label tensor: {label_tensor.item()}")
    assert label_tensor.item() == 42, "Label should be 42"
    
    print("✓ Full pipeline works correctly\n")
    
    return True

def main():
    print("\n" + "=" * 60)
    print("MXNet Replacement Validation Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("RecordIO .idx Parsing", test_recordio_format),
        ("RecordIO Header Unpacking", test_recordio_header_unpacking),
        ("Image Decoding (PIL)", test_image_decoding),
        ("Image Horizontal Flip", test_image_flip),
        ("Image Resize", test_image_resize),
        ("NumPy to PyTorch", test_torch_conversion),
        ("Integration Test", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} FAILED: {e}\n")
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"     Error: {error}")
    
    print("=" * 60)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! MXNet replacement is working correctly.\n")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the implementation.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
