---
description: Repository Information Overview
alwaysApply: true
---

# InsightFace Repository Information

## Summary
InsightFace is an open-source 2D and 3D deep face analysis toolbox, primarily based on Python with C/C++ SDK support. It provides state-of-the-art algorithms for face recognition, detection, alignment, reconstruction, and analysis, optimized for both training and deployment across multiple platforms.

## Repository Structure
**Root-level directories**:
- **python-package/**: Main Python library for face analysis (ONNX-based inference)
- **cpp-package/inspireface/**: Cross-platform C/C++ face recognition SDK
- **detection/**: Face detection models (SCRFD, RetinaFace, BlazeFace)
- **recognition/**: Face recognition training implementations (ArcFace variants)
- **alignment/**: Face alignment and landmark detection
- **reconstruction/**: 3D face reconstruction (PBIDR, gaze tracking, JMLR, OSTEC)
- **attribute/**: Face attribute analysis
- **generation/**: Face generation models
- **parsing/**: Face parsing and segmentation
- **examples/**: Usage examples and demos
- **model_zoo/**: Pre-trained model repository
- **benchmarks/**: Performance benchmarking tools
- **challenges/**: Competition and challenge code
- **web-demos/**: Web-based demonstration applications
- **tools/**: Utility scripts and tools

## Projects

### Python Package (InsightFace Library)
**Configuration**: `python-package/setup.py`, `python-package/pyproject.toml`

#### Language & Runtime
**Language**: Python  
**Version**: Python 3.x  
**Build System**: setuptools  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- numpy, onnx, onnxruntime (or onnxruntime-gpu)
- opencv-python, Pillow, scipy, matplotlib
- scikit-learn, scikit-image
- tqdm, requests, easydict, prettytable, albumentations

**Build Dependencies**:
- Cython >= 0.29.28
- cmake >= 3.22.3
- numpy >= 1.22.3

#### Build & Installation
```bash
pip install insightface
```

**From source**:
```bash
cd python-package
python setup.py install
```

#### Testing
**Framework**: unittest  
**Test Location**: `python-package/insightface/test/`  
**Naming Convention**: `test_*.py`

**Run Command**:
```bash
python -m unittest discover -s test
```

### C++ Package (InspireFace SDK)
**Configuration**: `cpp-package/inspireface/CMakeLists.txt`

#### Language & Runtime
**Language**: C++14  
**Build System**: CMake 3.20+  
**Version**: 1.2.3

#### Dependencies
**Main Dependencies**:
- OpenCV (image processing)
- ONNX Runtime or platform-specific inference backends
- Third-party libraries (auto-cloned from inspireface-3rdparty repository)

#### Build & Installation
**Quick install via Python**:
```bash
pip install -U inspireface
```

**Build from source**:
```bash
cd cpp-package/inspireface
./command/build.sh
```

**Platform-specific builds**:
- Linux: `./command/build_linux_ubuntu18.sh`
- macOS: `./command/build_macos_arm64.sh` or `./command/build_macos_x86.sh`
- Android: `./command/build_android.sh`
- iOS: `./command/build_ios.sh`
- CUDA: `./command/build_linux_cuda.sh`

#### Docker
**Dockerfile Locations**: `cpp-package/inspireface/docker/`
**Images**:
- `Dockerfile.ubuntu18`: Base Ubuntu 18 build
- `Dockerfile.cuda.ubuntu20`: CUDA-enabled Ubuntu 20 build
- `Dockerfile.cuda12_ubuntu22`: CUDA 12 on Ubuntu 22
- `Dockerfile.android`: Android cross-compilation
- `Dockerfile.arm-linux-aarch64`: ARM Linux builds
- `Dockerfile.manylinux2014_x86`: Python wheel builds for x86

**Configuration**: Multi-stage builds with various base images for cross-platform support.

#### Testing
**Framework**: C++ unittest  
**Test Location**: `cpp-package/inspireface/cpp/test/`  
**Configuration**: `cpp-package/inspireface/cpp/test/CMakeLists.txt`

**Python test command**:
```bash
cd cpp-package/inspireface/python
./test.sh
```

### Detection Subprojects

#### SCRFD (Face Detection)
**Configuration**: `detection/scrfd/setup.py`, `detection/scrfd/requirements.txt`

**Language**: Python  
**Framework**: PyTorch, MMDetection  
**Dependencies**: See `detection/scrfd/requirements/` (runtime, build, optional, tests)

**Training**:
```bash
cd detection/scrfd/tools
./dist_train.sh
```

**Testing**:
```bash
cd detection/scrfd/tools
./dist_test.sh
```

#### RetinaFace
**Location**: `detection/retinaface/`  
**Framework**: MXNet  
**Entry Point**: `detection/retinaface/train.py`

#### BlazeFace (PaddlePaddle)
**Location**: `detection/blazeface_paddle/`  
**Framework**: PaddlePaddle

### Recognition Subprojects

#### ArcFace (PyTorch)
**Location**: `recognition/arcface_torch/`  
**Framework**: PyTorch  
**Entry Points**: `train_v2.py`, `train_v3.py`

#### ArcFace (MXNet)
**Location**: `recognition/arcface_mxnet/`  
**Framework**: MXNet  
**Entry Points**: `train.py`, `train_parall.py`

#### ArcFace (OneFlow)
**Location**: `recognition/arcface_oneflow/`  
**Framework**: OneFlow  
**Dependencies**: See `recognition/arcface_oneflow/requirements.txt`

**Training**:
```bash
cd recognition/arcface_oneflow
./train_ddp.sh
```

#### ArcFace (PaddlePaddle)
**Location**: `recognition/arcface_paddle/`  
**Framework**: PaddlePaddle  
**Variants**: dynamic, static

#### Partial FC
**Location**: `recognition/partial_fc/`  
**Framework**: MXNet  
**Purpose**: Distributed face recognition training

### Reconstruction Subprojects

#### PBIDR (Monocular Face Reconstruction)
**Location**: `reconstruction/PBIDR/`  
**Configuration**: `reconstruction/PBIDR/requirements.txt`

**Training**:
```bash
cd reconstruction/PBIDR/code/script
./fast_train.sh
```

#### Gaze Estimation
**Location**: `reconstruction/gaze/`  
**Entry Point**: `reconstruction/gaze/trainer_gaze.py`

#### JMLR (3D Reconstruction)
**Location**: `reconstruction/jmlr/`  
**Entry Point**: `reconstruction/jmlr/train.py`

## Usage & Operations

**Install main package**:
```bash
pip install insightface
```

**Install C++ SDK**:
```bash
pip install -U inspireface
```

**Download models**:
```bash
cd cpp-package/inspireface/command
./download_models_general.sh
```

**Quick example** (Python):
```python
from insightface.app import FaceAnalysis
app = FaceAnalysis()
app.prepare(ctx_id=0)
```

## Key Entry Points

**Main Python Library**: `python-package/insightface/__init__.py`  
**CLI Tool**: `insightface-cli` (console script)  
**Examples**: `examples/demo_analysis.py`, `examples/face_detection/`, `examples/face_recognition/`  
**Face Swapper**: `examples/in_swapper/`

## Validation

**Python Package Tests**: Uses unittest framework with test discovery
**C++ Tests**: CMake-based unit tests
**Integration Tests**: Various test scripts in subproject directories
