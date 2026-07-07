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
/// For visualization and debugging, we can export Gaussians as point clouds
/// where each point represents a Gaussian center with its color.
///
/// Later (M10), we'll extend this to save full Gaussian parameters.
pub fn save_ply(cloud: &GaussianCloud, path: &Path) -> Result<(), LoadError> {
    // TODO: Implement for M10
    // For M10, save full Gaussian parameters (scale, rotation, opacity, SH)
    unimplemented!("See M10 - PLY export for Gaussian clouds")
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

    #[test]
    #[ignore]
    fn test_ply_roundtrip() {
        // TODO: Test save and load
    }
}
