# PLY Binary Format Specification for Gaussian Splatting

**Issue**: hq-5b5
**Status**: Design Specification
**Version**: 1.0
**Date**: 2026-01-21

## Overview

This document specifies the standard PLY (Polygon File Format) binary encoding for 3D Gaussian Splatting models, compatible with nerfstudio (Splatfacto), Polycam, SuperSplat, and other industry-standard tools.

## Goals

1. **Industry Compatibility**: Load models from nerfstudio, Polycam, and other tools
2. **Standard Compliance**: Use widely-adopted PLY binary little-endian format
3. **Full Fidelity**: Preserve all Gaussian parameters (position, scale, rotation, opacity, SH coefficients)
4. **Variable SH Support**: Support SH degrees 0-3 (1, 4, 9, or 16 coefficients)

## File Format

### Container Format

- **Format**: PLY binary little-endian
- **Extension**: `.ply`
- **Compression**: None (users can use external tools like gzip if needed)
- **Endianness**: Little-endian (most common, matches x86/ARM)

### PLY Structure

```
ply
format binary_little_endian 1.0
comment Created by SplatRs
element vertex <N>
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
...
property float f_rest_44
end_header
<binary data>
```

## Property Mapping

### Position (12 bytes)

| PLY Property | Type | Gaussian Field | Notes |
|--------------|------|----------------|-------|
| `x` | float32 | `position.x` | World-space X coordinate |
| `y` | float32 | `position.y` | World-space Y coordinate |
| `z` | float32 | `position.z` | World-space Z coordinate |

### Scale (12 bytes)

| PLY Property | Type | Gaussian Field | Encoding |
|--------------|------|----------------|----------|
| `scale_0` | float32 | `scale.x` | **Log-space**: actual_scale = exp(scale_0) |
| `scale_1` | float32 | `scale.y` | **Log-space**: actual_scale = exp(scale_1) |
| `scale_2` | float32 | `scale.z` | **Log-space**: actual_scale = exp(scale_2) |

**Important**: Scale is stored in **log-space** for numerical stability during optimization.
- PLY files store the raw log values
- To get actual scale: `actual_scale = exp(scale_i)`
- Negative values are valid (exp(negative) gives values < 1)

### Rotation (16 bytes)

| PLY Property | Type | Gaussian Field | Notes |
|--------------|------|----------------|-------|
| `rot_0` | float32 | `rotation.w` | Quaternion W component (scalar part) |
| `rot_1` | float32 | `rotation.i` | Quaternion X component (i) |
| `rot_2` | float32 | `rotation.j` | Quaternion Y component (j) |
| `rot_3` | float32 | `rotation.k` | Quaternion Z component (k) |

**Quaternion Order**: Standard PLY uses `(w, x, y, z)` order (scalar-first)
- This matches our internal `.gs` format
- Some tools may use `(x, y, z, w)` - need to detect and handle on import
- **Must normalize** on load: `q_normalized = q / ||q||`

**Validation on Load**:
```rust
// Read quaternion components
let quat = Quaternion::new(rot_0, rot_1, rot_2, rot_3);
// Normalize to unit quaternion (handles numerical drift)
let rotation = UnitQuaternion::from_quaternion(quat);
```

### Opacity (4 bytes)

| PLY Property | Type | Gaussian Field | Encoding |
|--------------|------|----------------|----------|
| `opacity` | float32 | `opacity` | **Logit-space**: actual_opacity = sigmoid(opacity) |

**Important**: Opacity is stored in **logit-space** (inverse sigmoid)
- Range: (-∞, +∞) in PLY file
- Actual opacity: `sigmoid(opacity) = 1 / (1 + exp(-opacity))`
- This ensures opacity stays in (0, 1) during optimization

### Spherical Harmonics (variable size)

#### DC Components (Base Color) - 12 bytes

| PLY Property | Type | Gaussian Field | Notes |
|--------------|------|----------------|-------|
| `f_dc_0` | float32 | `sh_coeffs[0][0]` | DC coefficient for Red channel |
| `f_dc_1` | float32 | `sh_coeffs[0][1]` | DC coefficient for Green channel |
| `f_dc_2` | float32 | `sh_coeffs[0][2]` | DC coefficient for Blue channel |

**Color Conversion**: The DC coefficients are NOT RGB values directly
```rust
// To get approximate base color:
const SH_C0: f32 = 0.28209479177387814; // sqrt(1/(4*pi))
let rgb_r = 0.5 + SH_C0 * f_dc_0;
let rgb_g = 0.5 + SH_C0 * f_dc_1;
let rgb_b = 0.5 + SH_C0 * f_dc_2;
```

#### Higher-Order SH Coefficients - Variable size

| SH Degree | Total Coefficients | f_rest Properties | Bytes |
|-----------|-------------------|-------------------|-------|
| 0 | 1 | None (DC only) | 0 |
| 1 | 4 | `f_rest_0` to `f_rest_8` (9 floats) | 36 |
| 2 | 9 | `f_rest_0` to `f_rest_23` (24 floats) | 96 |
| 3 | 16 | `f_rest_0` to `f_rest_44` (45 floats) | 180 |

**Coefficient Layout** (for degree 3):
```
Index | Coefficient | RGB Interleaved
------|-------------|----------------
0     | DC          | f_dc_0, f_dc_1, f_dc_2
1     | SH1_-1      | f_rest_0, f_rest_1, f_rest_2
2     | SH1_0       | f_rest_3, f_rest_4, f_rest_5
3     | SH1_+1      | f_rest_6, f_rest_7, f_rest_8
4     | SH2_-2      | f_rest_9, f_rest_10, f_rest_11
5     | SH2_-1      | f_rest_12, f_rest_13, f_rest_14
6     | SH2_0       | f_rest_15, f_rest_16, f_rest_17
7     | SH2_+1      | f_rest_18, f_rest_19, f_rest_20
8     | SH2_+2      | f_rest_21, f_rest_22, f_rest_23
9     | SH3_-3      | f_rest_24, f_rest_25, f_rest_26
10    | SH3_-2      | f_rest_27, f_rest_28, f_rest_29
11    | SH3_-1      | f_rest_30, f_rest_31, f_rest_32
12    | SH3_0       | f_rest_33, f_rest_34, f_rest_35
13    | SH3_+1      | f_rest_36, f_rest_37, f_rest_38
14    | SH3_+2      | f_rest_39, f_rest_40, f_rest_41
15    | SH3_+3      | f_rest_42, f_rest_43, f_rest_44
```

**Mapping to Internal Representation**:
```rust
// Our internal format: sh_coeffs[[f32; 3]; 16]
// Index 0 is DC (stored in f_dc_*)
sh_coeffs[0] = [f_dc_0, f_dc_1, f_dc_2];

// Indices 1-15 are higher-order (stored in f_rest_*)
for i in 1..num_coeffs {
    let base_idx = (i - 1) * 3;
    sh_coeffs[i][0] = f_rest[base_idx + 0]; // R
    sh_coeffs[i][1] = f_rest[base_idx + 1]; // G
    sh_coeffs[i][2] = f_rest[base_idx + 2]; // B
}
```

## Size Calculations

### Per-Gaussian Size (bytes)

| Component | Size | Notes |
|-----------|------|-------|
| Position (x, y, z) | 12 | 3 × float32 |
| Scale (scale_0-2) | 12 | 3 × float32 |
| Rotation (rot_0-3) | 16 | 4 × float32 |
| Opacity | 4 | 1 × float32 |
| SH DC (f_dc_0-2) | 12 | 3 × float32 |
| **Subtotal (fixed)** | **56** | |
| SH Rest (degree 0) | 0 | No higher-order |
| SH Rest (degree 1) | 36 | 9 × float32 |
| SH Rest (degree 2) | 96 | 24 × float32 |
| SH Rest (degree 3) | 180 | 45 × float32 |
| **Total (degree 3)** | **236** | Same as .gs format |

### File Size Estimates

| Gaussians | Header | Data (deg 3) | Total |
|-----------|--------|--------------|-------|
| 1K | ~500 bytes | 236 KB | ~237 KB |
| 10K | ~500 bytes | 2.36 MB | ~2.36 MB |
| 100K | ~500 bytes | 23.6 MB | ~23.6 MB |
| 1M | ~500 bytes | 236 MB | ~236 MB |

**Note**: Binary PLY files are **not compressed**. For comparison:
- .gs with LZ4: typically 5-10× smaller
- .ply.gz (external): typically 3-5× smaller

## Implementation Notes

### Writing PLY Files

1. **Header Construction**:
   ```rust
   fn write_header(num_gaussians: usize, sh_degree: u32) -> String {
       let num_rest_coeffs = match sh_degree {
           0 => 0,
           1 => 9,
           2 => 24,
           3 => 45,
           _ => panic!("Unsupported SH degree"),
       };

       let mut header = String::from("ply\n");
       header.push_str("format binary_little_endian 1.0\n");
       header.push_str(&format!("comment Created by SplatRs\n"));
       header.push_str(&format!("element vertex {}\n", num_gaussians));

       // Position
       header.push_str("property float x\n");
       header.push_str("property float y\n");
       header.push_str("property float z\n");

       // Scale (log-space)
       header.push_str("property float scale_0\n");
       header.push_str("property float scale_1\n");
       header.push_str("property float scale_2\n");

       // Rotation (quaternion w,x,y,z)
       header.push_str("property float rot_0\n");
       header.push_str("property float rot_1\n");
       header.push_str("property float rot_2\n");
       header.push_str("property float rot_3\n");

       // Opacity (logit-space)
       header.push_str("property float opacity\n");

       // SH DC
       header.push_str("property float f_dc_0\n");
       header.push_str("property float f_dc_1\n");
       header.push_str("property float f_dc_2\n");

       // SH higher-order
       for i in 0..num_rest_coeffs {
           header.push_str(&format!("property float f_rest_{}\n", i));
       }

       header.push_str("end_header\n");
       header
   }
   ```

2. **Binary Data Writing**:
   ```rust
   fn write_gaussian_binary<W: Write>(
       writer: &mut W,
       gaussian: &Gaussian,
       sh_degree: u32,
   ) -> io::Result<()> {
       // Position (3 × f32 = 12 bytes)
       writer.write_all(&gaussian.position.x.to_le_bytes())?;
       writer.write_all(&gaussian.position.y.to_le_bytes())?;
       writer.write_all(&gaussian.position.z.to_le_bytes())?;

       // Scale - already in log-space (3 × f32 = 12 bytes)
       writer.write_all(&gaussian.scale.x.to_le_bytes())?;
       writer.write_all(&gaussian.scale.y.to_le_bytes())?;
       writer.write_all(&gaussian.scale.z.to_le_bytes())?;

       // Rotation (4 × f32 = 16 bytes)
       let q = gaussian.rotation.quaternion();
       writer.write_all(&q.w.to_le_bytes())?; // rot_0
       writer.write_all(&q.i.to_le_bytes())?; // rot_1
       writer.write_all(&q.j.to_le_bytes())?; // rot_2
       writer.write_all(&q.k.to_le_bytes())?; // rot_3

       // Opacity - already in logit-space (1 × f32 = 4 bytes)
       writer.write_all(&gaussian.opacity.to_le_bytes())?;

       // SH DC (f_dc_0-2) - 3 × f32 = 12 bytes
       writer.write_all(&gaussian.sh_coeffs[0][0].to_le_bytes())?;
       writer.write_all(&gaussian.sh_coeffs[0][1].to_le_bytes())?;
       writer.write_all(&gaussian.sh_coeffs[0][2].to_le_bytes())?;

       // SH higher-order (f_rest_*)
       let num_coeffs = match sh_degree {
           0 => 1,
           1 => 4,
           2 => 9,
           3 => 16,
           _ => panic!("Unsupported SH degree"),
       };

       for i in 1..num_coeffs {
           writer.write_all(&gaussian.sh_coeffs[i][0].to_le_bytes())?;
           writer.write_all(&gaussian.sh_coeffs[i][1].to_le_bytes())?;
           writer.write_all(&gaussian.sh_coeffs[i][2].to_le_bytes())?;
       }

       Ok(())
   }
   ```

### Reading PLY Files

1. **Header Parsing**:
   - Parse header line-by-line until "end_header"
   - Extract `element vertex <N>` to get count
   - Detect SH degree from number of `f_rest_*` properties
   - Validate format is `binary_little_endian`

2. **SH Degree Detection**:
   ```rust
   fn detect_sh_degree(f_rest_count: usize) -> u32 {
       match f_rest_count {
           0 => 0,   // DC only
           9 => 1,   // DC + 3 coeffs
           24 => 2,  // DC + 8 coeffs
           45 => 3,  // DC + 15 coeffs
           _ => panic!("Invalid f_rest count: {}", f_rest_count),
       }
   }
   ```

3. **Binary Data Reading**:
   - Read fixed 56 bytes (position, scale, rotation, opacity, f_dc)
   - Read variable f_rest bytes based on detected SH degree
   - Normalize quaternion after reading
   - Zero-fill unused SH coefficients if degree < 3

### Compatibility Considerations

1. **Quaternion Order**:
   - Standard: `(w, x, y, z)` - scalar first
   - Some tools use: `(x, y, z, w)` - scalar last
   - **Detection**: Check if quaternion norm ≈ 1.0 after reading
   - If not, try swapping order and check again

2. **Property Order**:
   - The order of properties in the header MUST match binary data order
   - Some tools may write properties in different order
   - Parse header to build property map, then read accordingly

3. **Missing Properties**:
   - If file has fewer SH coefficients than degree 3, zero-fill the rest
   - If file has no opacity, default to 0.0 (logit-space, = 50% actual)
   - If file has no scale, default to 0.0 (log-space, = 1.0 actual)

## Validation Checklist

When implementing, validate:

- [ ] Header format is exactly "binary_little_endian 1.0"
- [ ] All properties are type "float" (not "double" or "uchar")
- [ ] Property names match standard: x,y,z,scale_0-2,rot_0-3,opacity,f_dc_0-2,f_rest_*
- [ ] Quaternion is normalized after reading (||q|| ≈ 1.0)
- [ ] Scale values are in log-space (can be negative)
- [ ] Opacity values are in logit-space (can be any value)
- [ ] File size matches: header_size + (num_gaussians × bytes_per_gaussian)
- [ ] Can round-trip: save→load→save produces identical files

## References

1. **PLY Format**: [Stanford PLY Specification](http://paulbourke.net/dataformats/ply/)
2. **Gaussian Splatting Paper**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
3. **Nerfstudio Splatfacto**: [docs.nerf.studio](https://docs.nerf.studio/nerfology/methods/splat.html)
4. **PlayCanvas PLY Guide**: [developer.playcanvas.com](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/)

## Version History

- **v1.0** (2026-01-21): Initial specification
  - Binary little-endian format
  - Support for SH degrees 0-3
  - Standard property names
  - Quaternion (w,x,y,z) order
