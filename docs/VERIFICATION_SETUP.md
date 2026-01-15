# Verification Test Setup

## Reference Implementations

### Primary References
1. **Original 3DGS Implementation** (Python/CUDA)
   - URL: https://github.com/graphdeco-inria/gaussian-splatting
   - Version: Latest stable (as of January 2025)
   - Use: Reference for algorithm behavior, default parameters, validation of output quality

2. **diff-gaussian-rasterization** (CUDA)
   - URL: https://github.com/graphdeco-inria/diff-gaussian-rasterization
   - Version: Latest stable (as of January 2025)
   - Use: Reference rasterizer behavior for cross-validation
   - Note: Not directly used; SplatRs implements its own CPU/GPU rasterizer

3. **gsplat (Nerfstudio)** (Python/CUDA)
   - URL: https://github.com/nerfstudio-project/gsplat
   - Version: Latest stable (as of January 2025)
   - Use: Alternative reference for cross-validation of gradient computations
   - Note: Not directly used; serves as secondary validation source

### Evaluation Libraries (Installed Versions)
- **scikit-image** 0.26.0: SSIM reference implementation
- **lpips** 0.1.4: Perceptual quality metrics
- **torch** 2.9.1: Gradient checking, loss functions
- **torchvision** 0.24.1: Image processing utilities
- **opencv-python** 4.12.0.88: Image I/O and processing
- **matplotlib** 3.10.8: Visualization
- **numpy** 2.2.6: Numerical computations

See `requirements.txt` for minimum versions and `pip list` for currently installed versions.

## Datasets

### Available in `./datasets/`
✅ **Mip-NeRF 360 Scenes** (Real-world unbounded scenes)
- bicycle (outdoor)
- garden (outdoor)
- stump (outdoor)
- bonsai (indoor)
- counter (indoor)
- kitchen (indoor)
- room (indoor)
- dollhouse
- garden_sm (small version)
- dollhouse_sm (small version)

✅ **Tanks and Temples** (in tandt_db/)

### Missing - Need to Download
❌ **NeRF Synthetic (Blender)** - ~400 MB
- Required for: TC-OPT-012 (NeRF Synthetic Benchmark)
- Download: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- Scenes needed: lego (minimum), others optional (chair, drums, ficus, hotdog, materials, mic, ship)
- Target location: `./datasets/nerf_synthetic/`
- **NOTE**: Google Drive link requires manual download or gdown tool

❌ **DTU Dataset** - ~30 GB
- Required for: Advanced benchmarking (optional for initial verification)
- Download: https://roboimagedata.compute.dtu.dk/?page_id=36
- Target location: `./datasets/dtu/`
- **NOTE**: Very large, defer for later

❌ **Cornell Box**
- Required for: Synthetic test (TC-INP-010)
- Download: https://www.graphics.cornell.edu/online/box/data.html
- Target location: `./datasets/synthetic/cornell_box/`
- **NOTE**: Can be created procedurally as alternative

### Synthetic Test Assets (To Generate)
These will be created programmatically for specific tests:
- Single colored cube (TC-COV-001, basic tests)
- Overlapping spheres (TC-RAS-001, TC-RAS-002)
- Checkerboard plane (high-frequency detail)
- Test images with known pixel values (TC-INP-004)
- COLMAP synthetic output (TC-INP-001, TC-INP-002)

## Python Environment Setup

### Current venv packages
- pillow 12.0.0
- pip 25.1.1

### Packages to Install (as needed)
```bash
source venv/bin/activate

# Core evaluation
pip install scikit-image  # SSIM reference
pip install lpips         # Perceptual metrics
pip install opencv-python # Image I/O
pip install matplotlib    # Visualization
pip install numpy         # Numerical computations

# PyTorch (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional
pip install open3d        # Point cloud visualization
pip install piq           # Additional metrics
```

## Download Instructions

### NeRF Synthetic Dataset
Option 1 - Using gdown:
```bash
pip install gdown
cd datasets
gdown --folder https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
mv nerf_synthetic nerf_synthetic  # Adjust based on actual folder name
```

Option 2 - Manual:
1. Visit https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
2. Download the entire folder or individual scenes
3. Extract to `./datasets/nerf_synthetic/`
4. Verify structure: `./datasets/nerf_synthetic/lego/train/`, etc.

### Add datasets to .gitignore
Already configured - datasets should not be committed to git.

## Implementation Notes

### SplatRs vs Reference Implementation

**Architecture**: SplatRs is a **from-scratch pure Rust implementation** of 3D Gaussian Splatting, not a port of the original Python/CUDA codebase.

**Key Deviations**:

1. **Language & Runtime**
   - Pure Rust implementation (CPU + wgpu GPU backend)
   - No Python/CUDA dependencies
   - Different memory layout optimizations (SoA vs AoS)

2. **Rasterizer Implementation**
   - Custom tile-based rasterizer following the paper's algorithm
   - Not a direct port of diff-gaussian-rasterization
   - Validates correctness against reference output, not reference code

3. **Gradient Computation**
   - Custom backward pass implementations in `src/diff/`
   - Verified via finite difference checks and output quality metrics
   - Different numerical precision characteristics than PyTorch/CUDA

4. **Optimization Pipeline**
   - Custom Adam optimizer implementation
   - Same hyperparameters as reference (learning rates, beta values)
   - Adaptive density control follows paper specification

5. **Platform Support**
   - Cross-platform (macOS, Linux, Windows)
   - CPU rendering always available
   - GPU via wgpu (Vulkan/Metal/DX12), not CUDA

**Validation Strategy**:
- Algorithm correctness validated through output quality metrics (PSNR, SSIM, LPIPS)
- Gradient correctness validated via finite difference checks
- Behavior validated against reference implementation outputs
- Not byte-for-byte identical to reference, but functionally equivalent

**Rust Dependencies**:
See `Cargo.toml` for full dependency list. Key libraries:
- nalgebra 0.33: Linear algebra (matrices, vectors, quaternions)
- image 0.25: Image I/O
- rayon 1.10: Parallel iteration
- wgpu 0.19: GPU compute (optional, feature-gated)

**Python Dependencies**:
See `requirements.txt` for minimum versions. Used only for:
- Evaluation metrics (SSIM, LPIPS)
- Test data generation
- Visualization and debugging
- Not used during training/rendering (pure Rust pipeline)
