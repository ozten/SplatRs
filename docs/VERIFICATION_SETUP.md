# Verification Test Setup

## Reference Implementations

### Primary References
1. **Original 3DGS Implementation** (Python/CUDA)
   - URL: https://github.com/graphdeco-inria/gaussian-splatting
   - Use: Reference for algorithm behavior, default parameters

2. **diff-gaussian-rasterization** (CUDA)
   - URL: https://github.com/graphdeco-inria/diff-gaussian-rasterization
   - Use: Reference rasterizer with gradients, validate our rasterizer against

3. **gsplat (Nerfstudio)** (Python/CUDA)
   - URL: https://github.com/nerfstudio-project/gsplat
   - Use: Alternative rasterizer for cross-validation

### Evaluation Libraries
- **scikit-image**: SSIM reference implementation
- **LPIPS**: Perceptual quality metrics
- **PyTorch**: Gradient checking, loss functions
- **NumPy**: Numerical computations

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
