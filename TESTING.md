# Testing Guide

## Test Organization

Tests are organized into two categories:

### 🚀 Fast Tests (Unit + Integration)
- Run by default with `cargo test`
- Complete in < 5 seconds
- No external dependencies required
- **Use these for development/CI**

### 🐢 Slow Tests (E2E)
- Marked with `#[ignore]`
- Require external datasets (calipers project)
- May take minutes to complete
- **Run manually before releases**

---

## Running Tests

### Fast Tests Only (Default)
```bash
# Run all fast tests
cargo test

# Run only library unit tests
cargo test --lib

# Run specific test module
cargo test gradient_check
cargo test adam

# Run with output
cargo test -- --nocapture
```

**Expected time**: < 5 seconds ⚡

### Slow E2E Tests
```bash
# Run all E2E tests (requires external data)
cargo test -- --ignored

# Run specific E2E test
cargo test test_render_calipers_fixed_size -- --ignored

# Run E2E tests with output
cargo test -- --ignored --nocapture
```

**Expected time**: 5-10 minutes 🐢

### All Tests (Fast + Slow)
```bash
# Run everything (for thorough verification)
cargo test -- --include-ignored
```

**Expected time**: 5-10 minutes

---

## Test Categories

### Unit Tests (Fast ⚡)
**Location**: `src/**/*.rs` in `#[cfg(test)]` modules
**Count**: 22 tests

Tests individual components:
- ✅ Camera projection math
- ✅ Gaussian covariance matrices
- ✅ sRGB color conversions
- ✅ Spherical harmonics
- ✅ Adam optimizer
- ✅ Renderer smoke tests

**Run**: `cargo test --lib`

---

### Gradient Checks (Fast ⚡)
**Location**: `tests/gradient_check.rs`
**Count**: 9 tests

Verifies analytical gradients against numerical gradients:
- ✅ Sigmoid & inverse sigmoid
- ✅ Quaternion → rotation matrix
- ✅ Covariance projection
- ✅ 2D Gaussian evaluation
- ✅ Alpha blending
- ✅ SH coefficient gradients
- ✅ Combined projection pipeline

**Run**: `cargo test gradient_check`

**Critical**: These must always pass!

---

### Integration Tests (Fast ⚡)
**Location**: Various `tests/*.rs`
**Count**: ~5 tests

Tests that run without external data:
- ✅ `test_projection_math_simple` - Basic projection math
- ✅ `test_depth_ordering` - Depth sorting
- ✅ `test_m5_sh_view_dependent_color_changes` - SH evaluation
- ✅ Dataset sanity checks (with in-repo datasets)

**Run**: `cargo test --tests` (excludes ignored)

---

### E2E Tests (Slow 🐢)
**Location**: Various `tests/*.rs`, marked with `#[ignore]`
**Count**: ~10 tests

Tests that require external datasets or are slow:

#### M1: COLMAP Loading
- `test_load_calipers_colmap` - Load full dataset
- `test_colmap_camera_details` - Verify camera parsing

#### M2: Point Projection
- `test_project_points_to_images` - Project to images, load PNGs

#### M3: Sphere Rendering
- `test_render_calipers_fixed_size` - **Very slow** (renders 4 viewpoints)

#### M4: Gaussian Rendering
- `test_m4_render_calipers_projected_covariance` - Renders with covariance

#### M7: Training
- `test_m7_overfit_color_only_calipers` - Full training run
- `test_m7_overfit_color_only_tandt_train` - Training on T&T dataset

#### Color Rendering
- `test_render_only_colorful_points` - Render filtered points

**Run**: `cargo test -- --ignored`

**Requirements**:
- Calipers dataset at: `/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/`
- T&T dataset at: `./datasets/tandt_db/tandt/train/`

---

## CI/CD Recommendations

### Pull Request Checks
```bash
# Fast tests only (should complete in < 10 seconds)
cargo test --lib
cargo test gradient_check
cargo test dataset_sanity_check
```

### Pre-Release Checks
```bash
# All tests including E2E
cargo test -- --include-ignored

# Or separately:
cargo test                    # Fast tests
cargo test -- --ignored       # E2E tests
```

---

## Adding New Tests

### Fast Test (Default)
```rust
#[test]
fn test_my_feature() {
    // Your test code
    assert_eq!(2 + 2, 4);
}
```

### Slow E2E Test
```rust
#[test]
#[ignore] // E2E test - brief reason (use `cargo test -- --ignored`)
fn test_my_slow_feature() {
    // Check for external dependencies
    if !path.exists() {
        println!("Skipping - data not found");
        return;
    }

    // Your slow test code
}
```

**Guidelines**:
- Mark as `#[ignore]` if test takes > 1 second
- Mark as `#[ignore]` if test requires external datasets
- Add comment explaining why it's ignored
- Add early return if dependencies missing

---

## Test Performance Benchmarks

### Before Optimization
```
cargo test:              7+ minutes  (includes slow renders)
```

### After Optimization
```
cargo test:              < 5 seconds  ⚡ (only fast tests)
cargo test -- --ignored: 5-10 minutes (E2E tests)
```

**Speedup**: >80x faster for development workflow! 🚀

---

## Troubleshooting

### "Test not found"
If `cargo test test_name` doesn't find your test:
```bash
# List all tests
cargo test -- --list

# Include ignored tests
cargo test -- --ignored --list
```

### "Dataset not found" errors
E2E tests skip gracefully if datasets missing:
```bash
$ cargo test test_load_calipers_colmap -- --ignored
Skipping test - COLMAP data not found
```

This is expected! E2E tests are optional.

### Tests timeout
Some E2E tests (especially `test_render_calipers_fixed_size`) take several minutes:
```bash
# Increase timeout (default is 60s for some test runners)
cargo test test_render_calipers_fixed_size -- --ignored --test-threads=1
```

---

## Test Maintenance

### When to Re-run E2E Tests
- ✅ Before major releases
- ✅ After changing rendering pipeline
- ✅ After modifying COLMAP loading
- ✅ When fixing bugs found in E2E scenarios
- ❌ Not needed for every PR

### Keeping Tests Fast
- Use small synthetic data for integration tests
- Mark any test > 1 second as `#[ignore]`
- Prefer unit tests over E2E tests
- Mock external dependencies when possible

---

## Summary

**Development** (daily):
```bash
cargo test                      # < 5 seconds ⚡
```

**Pre-commit** (optional):
```bash
cargo test gradient_check       # < 1 second
```

**Pre-release** (thorough):
```bash
cargo test -- --include-ignored # 5-10 minutes
```

Happy testing! 🧪✅
