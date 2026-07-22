//! PLY format I/O for Gaussian clouds and meshes.
//!
//! PLY (Polygon File Format) is used to:
//! - Export Gaussian clouds for visualization (M1-M2)
//! - Save trained models (M10)
//! - Export extracted meshes (M12)

use crate::core::{Gaussian, GaussianCloud};
use crate::io::{colmap::Point3D, LoadError};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Save COLMAP 3D points to PLY format (for M1 visualization).
///
/// This exports a simple point cloud with positions and colors.
pub fn save_colmap_points_ply(points: &[Point3D], path: &Path) -> Result<(), LoadError> {
    let mut file = File::create(path)?;

    // Write PLY header
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", points.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property uchar red")?;
    writeln!(file, "property uchar green")?;
    writeln!(file, "property uchar blue")?;
    writeln!(file, "end_header")?;

    // Write vertex data
    for point in points {
        writeln!(
            file,
            "{} {} {} {} {} {}",
            point.position.x,
            point.position.y,
            point.position.z,
            point.color[0],
            point.color[1],
            point.color[2]
        )?;
    }

    Ok(())
}

/// Save a Gaussian cloud to PLY format.
///
/// Saves in standard binary PLY format compatible with nerfstudio, Polycam, etc.
/// Supports variable SH degrees (0-3).
///
/// Format: Binary little-endian with properties:
/// - Position: x, y, z
/// - Scale: scale_0, scale_1, scale_2 (log-space)
/// - Rotation: rot_0, rot_1, rot_2, rot_3 (quaternion w,x,y,z)
/// - Opacity: opacity (logit-space)
/// - SH: f_dc_0, f_dc_1, f_dc_2 + f_rest_0..f_rest_N
///
/// # Arguments
/// * `cloud` - The Gaussian cloud to save
/// * `path` - Output file path
/// * `sh_degree` - Spherical harmonics degree (0-3)
pub fn save_ply(cloud: &GaussianCloud, path: &Path, sh_degree: u32) -> Result<(), LoadError> {
    if sh_degree > 3 {
        return Err(LoadError::InvalidFormat(format!(
            "Invalid SH degree: {}. Must be 0-3.",
            sh_degree
        )));
    }

    let mut file = File::create(path)?;

    // Write ASCII header
    write_ply_header(&mut file, cloud.len(), sh_degree)?;

    // Write binary data for each Gaussian
    for gaussian in cloud.as_slice() {
        write_gaussian_binary(&mut file, gaussian, sh_degree)?;
    }

    Ok(())
}

/// Write PLY ASCII header
fn write_ply_header(file: &mut File, num_gaussians: usize, sh_degree: u32) -> Result<(), LoadError> {
    // Calculate number of f_rest coefficients based on SH degree
    let num_rest_coeffs = match sh_degree {
        0 => 0,   // DC only (1 coefficient total)
        1 => 9,   // DC + 3 bands (4 coefficients total)
        2 => 24,  // DC + 8 bands (9 coefficients total)
        3 => 45,  // DC + 15 bands (16 coefficients total)
        _ => unreachable!(),
    };

    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "comment Created by SplatRs")?;
    writeln!(file, "element vertex {}", num_gaussians)?;

    // Position (3 floats)
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;

    // Scale (3 floats, log-space)
    writeln!(file, "property float scale_0")?;
    writeln!(file, "property float scale_1")?;
    writeln!(file, "property float scale_2")?;

    // Rotation (4 floats, quaternion w,x,y,z)
    writeln!(file, "property float rot_0")?;
    writeln!(file, "property float rot_1")?;
    writeln!(file, "property float rot_2")?;
    writeln!(file, "property float rot_3")?;

    // Opacity (1 float, logit-space)
    writeln!(file, "property float opacity")?;

    // SH DC coefficients (3 floats)
    writeln!(file, "property float f_dc_0")?;
    writeln!(file, "property float f_dc_1")?;
    writeln!(file, "property float f_dc_2")?;

    // SH higher-order coefficients
    for i in 0..num_rest_coeffs {
        writeln!(file, "property float f_rest_{}", i)?;
    }

    writeln!(file, "end_header")?;

    Ok(())
}

/// Write a single Gaussian as binary data
fn write_gaussian_binary(file: &mut File, gaussian: &crate::core::Gaussian, sh_degree: u32) -> Result<(), LoadError> {
    use std::io::Write;

    // Position (3 × f32 = 12 bytes)
    file.write_all(&gaussian.position.x.to_le_bytes())?;
    file.write_all(&gaussian.position.y.to_le_bytes())?;
    file.write_all(&gaussian.position.z.to_le_bytes())?;

    // Scale - already in log-space (3 × f32 = 12 bytes)
    file.write_all(&gaussian.scale.x.to_le_bytes())?;
    file.write_all(&gaussian.scale.y.to_le_bytes())?;
    file.write_all(&gaussian.scale.z.to_le_bytes())?;

    // Rotation quaternion (4 × f32 = 16 bytes)
    // Order: w, x, y, z (scalar-first, standard PLY)
    let q = gaussian.rotation.quaternion();
    file.write_all(&q.w.to_le_bytes())?;  // rot_0
    file.write_all(&q.i.to_le_bytes())?;  // rot_1
    file.write_all(&q.j.to_le_bytes())?;  // rot_2
    file.write_all(&q.k.to_le_bytes())?;  // rot_3

    // Opacity - already in logit-space (1 × f32 = 4 bytes)
    file.write_all(&gaussian.opacity.to_le_bytes())?;

    // SH DC coefficients (f_dc_0, f_dc_1, f_dc_2) - 3 × f32 = 12 bytes
    file.write_all(&gaussian.sh_coeffs[0][0].to_le_bytes())?;
    file.write_all(&gaussian.sh_coeffs[0][1].to_le_bytes())?;
    file.write_all(&gaussian.sh_coeffs[0][2].to_le_bytes())?;

    // SH higher-order coefficients (f_rest_*)
    let num_coeffs = match sh_degree {
        0 => 1,   // DC only
        1 => 4,   // DC + 3 coeffs
        2 => 9,   // DC + 8 coeffs
        3 => 16,  // DC + 15 coeffs
        _ => unreachable!(),
    };

    // Write f_rest_* (starting from index 1, since index 0 is DC)
    for i in 1..num_coeffs {
        file.write_all(&gaussian.sh_coeffs[i][0].to_le_bytes())?;  // R
        file.write_all(&gaussian.sh_coeffs[i][1].to_le_bytes())?;  // G
        file.write_all(&gaussian.sh_coeffs[i][2].to_le_bytes())?;  // B
    }

    Ok(())
}

/// Scalar property types that can appear in a PLY header.
#[derive(Clone, Copy)]
enum PlyType {
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
}

impl PlyType {
    fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "float" | "float32" => PlyType::F32,
            "double" | "float64" => PlyType::F64,
            "uchar" | "uint8" => PlyType::U8,
            "char" | "int8" => PlyType::I8,
            "ushort" | "uint16" => PlyType::U16,
            "short" | "int16" => PlyType::I16,
            "uint" | "uint32" => PlyType::U32,
            "int" | "int32" => PlyType::I32,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            PlyType::F32 | PlyType::U32 | PlyType::I32 => 4,
            PlyType::F64 => 8,
            PlyType::U8 | PlyType::I8 => 1,
            PlyType::U16 | PlyType::I16 => 2,
        }
    }

    /// Read a little-endian value at `buf` and return it as f32.
    fn read_le(self, buf: &[u8]) -> f32 {
        match self {
            PlyType::F32 => f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            PlyType::F64 => {
                f64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]])
                    as f32
            }
            PlyType::U8 => buf[0] as f32,
            PlyType::I8 => (buf[0] as i8) as f32,
            PlyType::U16 => u16::from_le_bytes([buf[0], buf[1]]) as f32,
            PlyType::I16 => i16::from_le_bytes([buf[0], buf[1]]) as f32,
            PlyType::U32 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
            PlyType::I32 => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
        }
    }
}

/// Load a Gaussian cloud from a trained 3DGS PLY (INRIA / Brush / standard format).
///
/// This reads the widely-used binary-little-endian layout produced by the reference
/// 3D Gaussian Splatting implementations, with per-vertex properties:
/// `x y z scale_0..2 opacity rot_0..3 f_dc_0..2 f_rest_0..N`.
///
/// Convention notes (these match SplatRs's internal representation, so values are copied
/// directly with no re-parameterization):
/// - `scale_*` are **log-space** scales (SplatRs stores log-space).
/// - `opacity` is a **logit** (inverse-sigmoid); SplatRs stores logit-space opacity.
/// - `rot_*` is a quaternion in `(w, x, y, z)` order, stored unnormalized; we normalize it.
/// - Spherical-harmonics `f_rest` are laid out **channel-major**: all coefficients of R,
///   then G, then B (i.e. `f_rest[c * n_per_channel + k]`).
pub fn load_ply(path: &Path) -> Result<GaussianCloud, LoadError> {
    let bytes = std::fs::read(path)?;

    // Locate the end of the ASCII header.
    let marker = b"end_header";
    let header_end = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .ok_or_else(|| LoadError::InvalidFormat("PLY: no end_header found".into()))?;
    // Advance past "end_header" and the following newline.
    let mut body_start = header_end + marker.len();
    while body_start < bytes.len() && bytes[body_start] != b'\n' {
        body_start += 1;
    }
    body_start += 1; // skip the '\n'

    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|e| LoadError::InvalidFormat(format!("PLY: header not UTF-8: {}", e)))?;

    let mut vertex_count: Option<usize> = None;
    let mut props: Vec<(String, PlyType)> = Vec::new();
    let mut is_binary_le = false;
    let mut in_vertex_element = false;

    for line in header.lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("format") => {
                if let Some(fmt) = it.next() {
                    is_binary_le = fmt == "binary_little_endian";
                }
            }
            Some("element") => {
                let name = it.next().unwrap_or("");
                let count: usize = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| LoadError::InvalidFormat("PLY: bad element count".into()))?;
                in_vertex_element = name == "vertex";
                if in_vertex_element {
                    vertex_count = Some(count);
                }
            }
            Some("property") if in_vertex_element => {
                // property <type> <name>  (we only support scalar properties, not lists)
                let ty = it.next().unwrap_or("");
                if ty == "list" {
                    return Err(LoadError::InvalidFormat(
                        "PLY: list properties on vertex element not supported".into(),
                    ));
                }
                let pty = PlyType::from_name(ty).ok_or_else(|| {
                    LoadError::InvalidFormat(format!("PLY: unknown property type '{}'", ty))
                })?;
                let name = it.next().unwrap_or("").to_string();
                props.push((name, pty));
            }
            _ => {}
        }
    }

    if !is_binary_le {
        return Err(LoadError::InvalidFormat(
            "PLY: only binary_little_endian is supported".into(),
        ));
    }
    let vertex_count = vertex_count
        .ok_or_else(|| LoadError::InvalidFormat("PLY: no vertex element".into()))?;

    // Compute per-vertex stride and each property's byte offset within a vertex record.
    let mut offsets: HashMap<String, (usize, PlyType)> = HashMap::new();
    let mut stride = 0usize;
    for (name, pty) in &props {
        offsets.insert(name.clone(), (stride, *pty));
        stride += pty.size();
    }

    let body = &bytes[body_start..];
    if body.len() < stride * vertex_count {
        return Err(LoadError::InvalidFormat(format!(
            "PLY: body too short: have {} bytes, need {} ({} verts × {} stride)",
            body.len(),
            stride * vertex_count,
            vertex_count,
            stride
        )));
    }

    // Determine the number of f_rest coefficients (per channel) present.
    let n_rest_total = props
        .iter()
        .filter(|(n, _)| n.starts_with("f_rest_"))
        .count();
    let n_rest_per_channel = n_rest_total / 3; // channel-major layout (R..,G..,B..)

    let get = |vert: &[u8], name: &str| -> Option<f32> {
        offsets.get(name).map(|(off, pty)| pty.read_le(&vert[*off..]))
    };

    let mut gaussians = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let vert = &body[i * stride..(i + 1) * stride];

        let position = Vector3::new(
            get(vert, "x").unwrap_or(0.0),
            get(vert, "y").unwrap_or(0.0),
            get(vert, "z").unwrap_or(0.0),
        );

        let scale = Vector3::new(
            get(vert, "scale_0").unwrap_or(0.0),
            get(vert, "scale_1").unwrap_or(0.0),
            get(vert, "scale_2").unwrap_or(0.0),
        );

        let opacity = get(vert, "opacity").unwrap_or(0.0);

        // Quaternion stored (w, x, y, z), unnormalized.
        let qw = get(vert, "rot_0").unwrap_or(1.0);
        let qx = get(vert, "rot_1").unwrap_or(0.0);
        let qy = get(vert, "rot_2").unwrap_or(0.0);
        let qz = get(vert, "rot_3").unwrap_or(0.0);
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));

        // Spherical harmonics: DC (coeff 0) from f_dc, rest from f_rest (channel-major).
        let mut sh_coeffs = [[0.0f32; 3]; 16];
        sh_coeffs[0] = [
            get(vert, "f_dc_0").unwrap_or(0.0),
            get(vert, "f_dc_1").unwrap_or(0.0),
            get(vert, "f_dc_2").unwrap_or(0.0),
        ];
        let max_k = n_rest_per_channel.min(15); // 15 higher-order coeffs fit in slots 1..=15
        for k in 0..max_k {
            for c in 0..3 {
                let idx = c * n_rest_per_channel + k;
                sh_coeffs[1 + k][c] = get(vert, &format!("f_rest_{}", idx)).unwrap_or(0.0);
            }
        }

        gaussians.push(Gaussian::new(position, scale, rotation, opacity, sh_coeffs));
    }

    Ok(GaussianCloud::from_gaussians(gaussians))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Gaussian;
    use nalgebra::{UnitQuaternion, Vector3};
    use std::io::{BufRead, BufReader};
    use tempfile::NamedTempFile;

    // Helper function to read only the ASCII header from a PLY file
    fn read_ply_header(path: &std::path::Path) -> Vec<String> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        // Find end of header
        let header_str = String::from_utf8_lossy(&buffer);
        let header_end_marker = "end_header\n";
        let header_end = header_str.find(header_end_marker).unwrap() + header_end_marker.len();

        // Extract and split header into lines
        let header_only = &buffer[..header_end];
        let header_text = String::from_utf8(header_only.to_vec()).unwrap();
        header_text.lines().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_save_ply_basic() {
        // Create a simple cloud with one Gaussian
        let mut cloud = GaussianCloud::new();
        let gaussian = Gaussian::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3), // log-space scale
            UnitQuaternion::identity(),
            0.5, // logit-space opacity
            [[0.1; 3]; 16],
        );
        cloud.push(gaussian);

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 3).unwrap();

        // Verify file exists and has content
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert!(metadata.len() > 0);

        // Read and verify header
        let lines = read_ply_header(temp_file.path());

        // Check header format
        assert_eq!(lines[0], "ply");
        assert_eq!(lines[1], "format binary_little_endian 1.0");
        assert_eq!(lines[2], "comment Created by SplatRs");
        assert_eq!(lines[3], "element vertex 1");

        // Find end_header
        let end_header_idx = lines.iter().position(|l| l == "end_header").unwrap();

        // Count properties (should be 59 for SH degree 3)
        // 3 (pos) + 3 (scale) + 4 (rot) + 1 (opacity) + 3 (f_dc) + 45 (f_rest) = 59 properties
        let property_lines: Vec<_> = lines[4..end_header_idx]
            .iter()
            .filter(|l| l.starts_with("property"))
            .collect();
        assert_eq!(property_lines.len(), 59);
    }

    #[test]
    fn test_save_ply_sh_degree_0() {
        // Test with SH degree 0 (DC only)
        let mut cloud = GaussianCloud::new();
        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            0.0,
            [[0.0; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 0).unwrap();

        // For degree 0: 3 (pos) + 3 (scale) + 4 (rot) + 1 (opacity) + 3 (f_dc) = 14 properties
        // No f_rest_* properties
        let lines = read_ply_header(temp_file.path());

        let property_count = lines.iter().filter(|l| l.starts_with("property")).count();
        assert_eq!(property_count, 14);

        // Verify no f_rest properties
        let has_f_rest = lines.iter().any(|l| l.contains("f_rest"));
        assert!(!has_f_rest);
    }

    #[test]
    fn test_save_ply_sh_degree_1() {
        // Test with SH degree 1
        let mut cloud = GaussianCloud::new();
        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            0.0,
            [[0.0; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 1).unwrap();

        // For degree 1: 14 + 9 (f_rest_0 to f_rest_8) = 23 properties
        let lines = read_ply_header(temp_file.path());

        let property_count = lines.iter().filter(|l| l.starts_with("property")).count();
        assert_eq!(property_count, 23);
    }

    #[test]
    fn test_save_ply_invalid_sh_degree() {
        let cloud = GaussianCloud::new();
        let temp_file = NamedTempFile::new().unwrap();

        // Should fail for SH degree > 3
        let result = save_ply(&cloud, temp_file.path(), 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_ply_multiple_gaussians() {
        // Test with multiple Gaussians
        let mut cloud = GaussianCloud::new();
        for i in 0..10 {
            cloud.push(Gaussian::new(
                Vector3::new(i as f32, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
                UnitQuaternion::identity(),
                0.0,
                [[0.5; 3]; 16],
            ));
        }

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 3).unwrap();

        // Read header and verify vertex count
        let lines = read_ply_header(temp_file.path());

        assert!(lines.iter().any(|l| l == "element vertex 10"));

        // Verify file size
        // Header is variable size, but each Gaussian with degree 3 is 236 bytes
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        // Should have header + (10 × 236 bytes) = header + 2360 bytes
        assert!(metadata.len() > 2360);
    }

    #[test]
    fn test_ply_header_format() {
        // Verify exact header format for compatibility
        let mut cloud = GaussianCloud::new();
        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            0.0,
            [[0.0; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 3).unwrap();

        let lines = read_ply_header(temp_file.path());

        // Verify exact property order (important for binary format)
        let mut prop_idx = 4; // Start after ply, format, comment, element vertex

        // Position
        assert_eq!(lines[prop_idx], "property float x");
        assert_eq!(lines[prop_idx + 1], "property float y");
        assert_eq!(lines[prop_idx + 2], "property float z");
        prop_idx += 3;

        // Scale
        assert_eq!(lines[prop_idx], "property float scale_0");
        assert_eq!(lines[prop_idx + 1], "property float scale_1");
        assert_eq!(lines[prop_idx + 2], "property float scale_2");
        prop_idx += 3;

        // Rotation
        assert_eq!(lines[prop_idx], "property float rot_0");
        assert_eq!(lines[prop_idx + 1], "property float rot_1");
        assert_eq!(lines[prop_idx + 2], "property float rot_2");
        assert_eq!(lines[prop_idx + 3], "property float rot_3");
        prop_idx += 4;

        // Opacity
        assert_eq!(lines[prop_idx], "property float opacity");
        prop_idx += 1;

        // SH DC
        assert_eq!(lines[prop_idx], "property float f_dc_0");
        assert_eq!(lines[prop_idx + 1], "property float f_dc_1");
        assert_eq!(lines[prop_idx + 2], "property float f_dc_2");
        prop_idx += 3;

        // SH rest (should start with f_rest_0)
        assert_eq!(lines[prop_idx], "property float f_rest_0");
    }

    #[test]
    fn test_binary_data_size() {
        // Verify binary data size matches spec
        let mut cloud = GaussianCloud::new();
        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            0.0,
            [[0.0; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 3).unwrap();

        // Read file and find header end
        let file_content = std::fs::read(temp_file.path()).unwrap();
        let header_str = String::from_utf8_lossy(&file_content);
        let header_end = header_str.find("end_header\n").unwrap() + "end_header\n".len();

        // Binary data size should be 236 bytes for 1 Gaussian with degree 3
        let binary_size = file_content.len() - header_end;
        assert_eq!(binary_size, 236);
    }

    #[test]
    fn test_load_ply_basic() {
        use approx::assert_relative_eq;

        // Create and save a test cloud
        let mut original_cloud = GaussianCloud::new();
        let gaussian = Gaussian::new(
            Vector3::new(1.5, 2.5, 3.5),
            Vector3::new(0.1, 0.2, 0.3), // log-space scale
            UnitQuaternion::identity(),
            0.75, // logit-space opacity
            [[0.5; 3]; 16],
        );
        original_cloud.push(gaussian);

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&original_cloud, temp_file.path(), 3).unwrap();

        // Load it back
        let loaded_cloud = load_ply(temp_file.path()).unwrap();

        // Verify
        assert_eq!(loaded_cloud.len(), 1);
        let loaded = &loaded_cloud.as_slice()[0];
        let orig = &original_cloud.as_slice()[0];

        assert_relative_eq!(loaded.position.x, orig.position.x, epsilon = 1e-6);
        assert_relative_eq!(loaded.position.y, orig.position.y, epsilon = 1e-6);
        assert_relative_eq!(loaded.position.z, orig.position.z, epsilon = 1e-6);

        assert_relative_eq!(loaded.scale.x, orig.scale.x, epsilon = 1e-6);
        assert_relative_eq!(loaded.scale.y, orig.scale.y, epsilon = 1e-6);
        assert_relative_eq!(loaded.scale.z, orig.scale.z, epsilon = 1e-6);

        assert_relative_eq!(loaded.opacity, orig.opacity, epsilon = 1e-6);

        // Check SH coefficients
        for i in 0..16 {
            for j in 0..3 {
                assert_relative_eq!(loaded.sh_coeffs[i][j], orig.sh_coeffs[i][j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_roundtrip_sh_degree_3() {
        use approx::assert_relative_eq;

        // Create a cloud with 5 Gaussians, various values
        let mut original_cloud = GaussianCloud::new();
        for i in 0..5 {
            let mut sh_coeffs = [[0.0f32; 3]; 16];
            for j in 0..16 {
                sh_coeffs[j] = [i as f32 + 0.1, i as f32 + 0.2, i as f32 + 0.3];
            }

            original_cloud.push(Gaussian::new(
                Vector3::new(i as f32, i as f32 * 2.0, i as f32 * 3.0),
                Vector3::new(-0.5 + i as f32 * 0.1, 0.0, 0.5),
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), i as f32 * 0.1),
                i as f32 - 2.0, // Various logit values
                sh_coeffs,
            ));
        }

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&original_cloud, temp_file.path(), 3).unwrap();
        let loaded_cloud = load_ply(temp_file.path()).unwrap();

        assert_eq!(loaded_cloud.len(), original_cloud.len());

        for i in 0..5 {
            let orig = &original_cloud.as_slice()[i];
            let loaded = &loaded_cloud.as_slice()[i];

            assert_relative_eq!(loaded.position.x, orig.position.x, epsilon = 1e-5);
            assert_relative_eq!(loaded.position.y, orig.position.y, epsilon = 1e-5);
            assert_relative_eq!(loaded.position.z, orig.position.z, epsilon = 1e-5);

            assert_relative_eq!(loaded.scale.x, orig.scale.x, epsilon = 1e-5);
            assert_relative_eq!(loaded.scale.y, orig.scale.y, epsilon = 1e-5);
            assert_relative_eq!(loaded.scale.z, orig.scale.z, epsilon = 1e-5);

            assert_relative_eq!(loaded.opacity, orig.opacity, epsilon = 1e-5);

            // Quaternions should be normalized and equal
            let q_orig = orig.rotation.quaternion();
            let q_loaded = loaded.rotation.quaternion();
            assert_relative_eq!(q_loaded.w, q_orig.w, epsilon = 1e-5);
            assert_relative_eq!(q_loaded.i, q_orig.i, epsilon = 1e-5);
            assert_relative_eq!(q_loaded.j, q_orig.j, epsilon = 1e-5);
            assert_relative_eq!(q_loaded.k, q_orig.k, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_roundtrip_sh_degree_0() {
        use approx::assert_relative_eq;

        // Test with SH degree 0 (DC only)
        let mut cloud = GaussianCloud::new();
        cloud.push(Gaussian::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            UnitQuaternion::identity(),
            0.5,
            [[0.25; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 0).unwrap();
        let loaded_cloud = load_ply(temp_file.path()).unwrap();

        assert_eq!(loaded_cloud.len(), 1);
        let loaded = &loaded_cloud.as_slice()[0];
        let orig = &cloud.as_slice()[0];

        // DC coefficients should match
        assert_relative_eq!(loaded.sh_coeffs[0][0], orig.sh_coeffs[0][0], epsilon = 1e-6);
        assert_relative_eq!(loaded.sh_coeffs[0][1], orig.sh_coeffs[0][1], epsilon = 1e-6);
        assert_relative_eq!(loaded.sh_coeffs[0][2], orig.sh_coeffs[0][2], epsilon = 1e-6);

        // Higher-order coefficients should be zero (not saved/loaded for degree 0)
        for i in 1..16 {
            assert_eq!(loaded.sh_coeffs[i][0], 0.0);
            assert_eq!(loaded.sh_coeffs[i][1], 0.0);
            assert_eq!(loaded.sh_coeffs[i][2], 0.0);
        }
    }

    #[test]
    fn test_roundtrip_sh_degree_1() {
        use approx::assert_relative_eq;

        let mut cloud = GaussianCloud::new();
        let mut sh_coeffs = [[0.0f32; 3]; 16];
        // Set DC + first 3 higher-order
        for i in 0..4 {
            sh_coeffs[i] = [i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3];
        }

        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
            0.0,
            sh_coeffs,
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 1).unwrap();
        let loaded_cloud = load_ply(temp_file.path()).unwrap();

        let loaded = &loaded_cloud.as_slice()[0];

        // First 4 coefficients should match
        for i in 0..4 {
            assert_relative_eq!(loaded.sh_coeffs[i][0], sh_coeffs[i][0], epsilon = 1e-6);
            assert_relative_eq!(loaded.sh_coeffs[i][1], sh_coeffs[i][1], epsilon = 1e-6);
            assert_relative_eq!(loaded.sh_coeffs[i][2], sh_coeffs[i][2], epsilon = 1e-6);
        }

        // Rest should be zero
        for i in 4..16 {
            assert_eq!(loaded.sh_coeffs[i][0], 0.0);
            assert_eq!(loaded.sh_coeffs[i][1], 0.0);
            assert_eq!(loaded.sh_coeffs[i][2], 0.0);
        }
    }

    #[test]
    fn test_load_ply_invalid_format() {
        use std::io::Write;

        // Test with ASCII format (not supported)
        let temp_file = NamedTempFile::new().unwrap();
        let mut file = std::fs::File::create(temp_file.path()).unwrap();
        writeln!(file, "ply").unwrap();
        writeln!(file, "format ascii 1.0").unwrap();
        writeln!(file, "element vertex 1").unwrap();
        writeln!(file, "end_header").unwrap();

        let result = load_ply(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_ply_missing_magic() {
        use std::io::Write;

        let temp_file = NamedTempFile::new().unwrap();
        let mut file = std::fs::File::create(temp_file.path()).unwrap();
        writeln!(file, "not_ply").unwrap();
        writeln!(file, "format binary_little_endian 1.0").unwrap();

        let result = load_ply(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_quaternion_normalization() {
        use approx::assert_relative_eq;

        // Create a Gaussian with a non-normalized quaternion
        let mut cloud = GaussianCloud::new();
        // Use from_axis_angle which gives a normalized quaternion
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.5);

        cloud.push(Gaussian::new(
            Vector3::zeros(),
            Vector3::zeros(),
            rotation,
            0.0,
            [[0.0; 3]; 16],
        ));

        let temp_file = NamedTempFile::new().unwrap();
        save_ply(&cloud, temp_file.path(), 3).unwrap();
        let loaded_cloud = load_ply(temp_file.path()).unwrap();

        // Loaded quaternion should be normalized
        let q = loaded_cloud.as_slice()[0].rotation.quaternion();
        let norm = (q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}
