//! PLY format I/O for Gaussian clouds and meshes.
//!
//! PLY (Polygon File Format) is used to:
//! - Export Gaussian clouds for visualization (M1-M2)
//! - Save trained models (M10)
//! - Export extracted meshes (M12)

use crate::core::GaussianCloud;
use crate::io::{colmap::Point3D, LoadError};
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

/// Load a Gaussian cloud from PLY format.
///
/// Loads from standard binary PLY format compatible with nerfstudio, Polycam, etc.
/// Automatically detects SH degree from header.
///
/// # Arguments
/// * `path` - Input file path
///
/// # Returns
/// * `GaussianCloud` with loaded Gaussians
pub fn load_ply(path: &Path) -> Result<GaussianCloud, LoadError> {
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut content = Vec::new();
    file.read_to_end(&mut content)?;

    // Parse header
    let (header_end, num_gaussians, sh_degree) = parse_ply_header(&content)?;

    // Read binary data
    let binary_data = &content[header_end..];
    let gaussians = read_ply_gaussians(binary_data, num_gaussians, sh_degree)?;

    Ok(GaussianCloud::from_gaussians(gaussians))
}

/// Parse PLY header and return (header_end_offset, num_gaussians, sh_degree)
fn parse_ply_header(content: &[u8]) -> Result<(usize, usize, u32), LoadError> {
    // Find end of header
    let header_str = String::from_utf8_lossy(content);
    let header_end_marker = "end_header\n";
    let header_end = header_str
        .find(header_end_marker)
        .ok_or_else(|| LoadError::InvalidFormat("Missing end_header".to_string()))?
        + header_end_marker.len();

    // Extract header text
    let header_text = String::from_utf8(content[..header_end].to_vec())
        .map_err(|_| LoadError::InvalidFormat("Header is not valid UTF-8".to_string()))?;

    let lines: Vec<&str> = header_text.lines().collect();

    // Validate PLY magic
    if lines.is_empty() || lines[0] != "ply" {
        return Err(LoadError::InvalidFormat(
            "Not a PLY file (missing 'ply' magic)".to_string(),
        ));
    }

    // Validate format
    let format_line = lines
        .iter()
        .find(|l| l.starts_with("format "))
        .ok_or_else(|| LoadError::InvalidFormat("Missing format line".to_string()))?;

    if !format_line.contains("binary_little_endian") {
        return Err(LoadError::InvalidFormat(
            "Only binary_little_endian format is supported".to_string(),
        ));
    }

    // Extract vertex count
    let vertex_line = lines
        .iter()
        .find(|l| l.starts_with("element vertex "))
        .ok_or_else(|| LoadError::InvalidFormat("Missing 'element vertex' line".to_string()))?;

    let num_gaussians = vertex_line
        .split_whitespace()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| LoadError::InvalidFormat("Invalid vertex count".to_string()))?;

    // Detect SH degree from f_rest_* properties
    let f_rest_count = lines
        .iter()
        .filter(|l| l.contains("f_rest_"))
        .count();

    let sh_degree = match f_rest_count {
        0 => 0,   // DC only
        9 => 1,   // DC + 3 coeffs
        24 => 2,  // DC + 8 coeffs
        45 => 3,  // DC + 15 coeffs
        _ => {
            return Err(LoadError::InvalidFormat(format!(
                "Invalid f_rest count: {}. Expected 0, 9, 24, or 45",
                f_rest_count
            )))
        }
    };

    Ok((header_end, num_gaussians, sh_degree))
}

/// Read Gaussian data from binary PLY content
fn read_ply_gaussians(
    data: &[u8],
    num_gaussians: usize,
    sh_degree: u32,
) -> Result<Vec<crate::core::Gaussian>, LoadError> {
    // Calculate bytes per Gaussian
    let bytes_per_gaussian = match sh_degree {
        0 => 56,       // Fixed only (no f_rest)
        1 => 56 + 36,  // Fixed + 9 f_rest
        2 => 56 + 96,  // Fixed + 24 f_rest
        3 => 56 + 180, // Fixed + 45 f_rest
        _ => unreachable!(),
    };

    // Validate data size
    let expected_size = num_gaussians * bytes_per_gaussian;
    if data.len() < expected_size {
        return Err(LoadError::InvalidFormat(format!(
            "Insufficient binary data: expected {} bytes, got {}",
            expected_size,
            data.len()
        )));
    }

    let mut gaussians = Vec::with_capacity(num_gaussians);
    let mut offset = 0;

    for _ in 0..num_gaussians {
        let gaussian = read_single_gaussian(&data[offset..], sh_degree)?;
        gaussians.push(gaussian);
        offset += bytes_per_gaussian;
    }

    Ok(gaussians)
}

/// Read a single Gaussian from binary data
fn read_single_gaussian(data: &[u8], sh_degree: u32) -> Result<crate::core::Gaussian, LoadError> {
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    if data.len() < 56 {
        return Err(LoadError::InvalidFormat(
            "Insufficient data for Gaussian".to_string(),
        ));
    }

    let mut offset = 0;

    // Helper to read f32
    let read_f32 = |off: &mut usize| -> f32 {
        let bytes = [data[*off], data[*off + 1], data[*off + 2], data[*off + 3]];
        *off += 4;
        f32::from_le_bytes(bytes)
    };

    // Position (12 bytes)
    let px = read_f32(&mut offset);
    let py = read_f32(&mut offset);
    let pz = read_f32(&mut offset);
    let position = Vector3::new(px, py, pz);

    // Scale - in log-space (12 bytes)
    let sx = read_f32(&mut offset);
    let sy = read_f32(&mut offset);
    let sz = read_f32(&mut offset);
    let scale = Vector3::new(sx, sy, sz);

    // Rotation quaternion (w, x, y, z) - 16 bytes
    let qw = read_f32(&mut offset);
    let qx = read_f32(&mut offset);
    let qy = read_f32(&mut offset);
    let qz = read_f32(&mut offset);

    // Normalize quaternion (important for numerical stability)
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));

    // Opacity - in logit-space (4 bytes)
    let opacity = read_f32(&mut offset);

    // SH coefficients
    let mut sh_coeffs = [[0.0f32; 3]; 16];

    // DC coefficients (f_dc_0, f_dc_1, f_dc_2) - 12 bytes
    sh_coeffs[0][0] = read_f32(&mut offset);
    sh_coeffs[0][1] = read_f32(&mut offset);
    sh_coeffs[0][2] = read_f32(&mut offset);

    // Higher-order coefficients (f_rest_*)
    // Read f_rest into indices 1..num_coeffs
    let num_coeffs = match sh_degree {
        0 => 1,
        1 => 4,
        2 => 9,
        3 => 16,
        _ => unreachable!(),
    };

    for i in 1..num_coeffs {
        sh_coeffs[i][0] = read_f32(&mut offset); // R
        sh_coeffs[i][1] = read_f32(&mut offset); // G
        sh_coeffs[i][2] = read_f32(&mut offset); // B
    }

    // Remaining coefficients are already zero-filled

    Ok(crate::core::Gaussian::new(
        position, scale, rotation, opacity, sh_coeffs,
    ))
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
