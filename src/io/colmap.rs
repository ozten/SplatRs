//! COLMAP binary format parser.
//!
//! COLMAP stores sparse reconstruction in binary files:
//! - cameras.bin: Camera intrinsics
//! - images.bin: Camera poses (extrinsics) + 2D keypoints
//! - points3D.bin: 3D points from structure-from-motion
//!
//! Format spec: https://colmap.github.io/format.html

use crate::core::Camera;
use byteorder::{LittleEndian, ReadBytesExt};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur when loading COLMAP data.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid COLMAP binary format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported camera model: {0}")]
    UnsupportedCameraModel(i32),
}

/// A complete COLMAP scene with cameras, images, and 3D points.
#[derive(Debug, Clone)]
pub struct ColmapScene {
    /// Camera parameters (intrinsics), indexed by camera_id
    pub cameras: HashMap<u32, Camera>,

    /// Image metadata (file paths, camera poses)
    pub images: Vec<ImageInfo>,

    /// 3D points from sparse reconstruction
    pub points: Vec<Point3D>,
}

/// Information about a single image in the COLMAP reconstruction.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Image ID
    pub id: u32,

    /// Camera ID (index into cameras array)
    pub camera_id: u32,

    /// Image file name
    pub name: String,

    /// Camera pose (rotation + translation to transform world → camera)
    pub rotation: nalgebra::UnitQuaternion<f32>,
    pub translation: Vector3<f32>,
}

/// A 3D point from COLMAP sparse reconstruction.
#[derive(Debug, Clone)]
pub struct Point3D {
    /// Point ID
    pub id: u64,

    /// 3D position
    pub position: Vector3<f32>,

    /// RGB color (0-255)
    pub color: [u8; 3],

    /// Reprojection error
    pub error: f32,
}

/// Load a complete COLMAP scene from a directory containing sparse reconstruction.
///
/// Expected directory structure:
/// ```text
/// sparse/0/
///   cameras.bin
///   images.bin
///   points3D.bin
/// ```
pub fn load_colmap_scene(sparse_dir: &Path) -> Result<ColmapScene, LoadError> {
    let cameras_path = sparse_dir.join("cameras.bin");
    let images_path = sparse_dir.join("images.bin");
    let points_path = sparse_dir.join("points3D.bin");

    // Parse each binary file
    let cameras = read_cameras_bin(&cameras_path)?;
    let images = read_images_bin(&images_path)?;
    let points = read_points3d_bin(&points_path)?;

    Ok(ColmapScene {
        cameras,
        images,
        points,
    })
}

/// Read cameras.bin file.
///
/// Binary format (little-endian):
/// - num_cameras: u64
/// - For each camera:
///   - camera_id: u32
///   - model_id: i32 (0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, etc.)
///   - width: u64
///   - height: u64
///   - params: [f64; N] (N depends on model, typically 4 for PINHOLE)
fn read_cameras_bin(path: &Path) -> Result<HashMap<u32, Camera>, LoadError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let num_cameras = reader.read_u64::<LittleEndian>()?;
    let mut cameras = HashMap::with_capacity(num_cameras as usize);
    let mut has_distortion_models = false; // Track if any cameras have distortion

    for _ in 0..num_cameras {
        let camera_id = reader.read_u32::<LittleEndian>()?;
        let model_id = reader.read_i32::<LittleEndian>()?;
        let width = reader.read_u64::<LittleEndian>()?;
        let height = reader.read_u64::<LittleEndian>()?;

        // Parse parameters based on camera model
        // Model IDs: 0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 3=RADIAL, etc.
        let camera = match model_id {
            0 => {
                // SIMPLE_PINHOLE: f, cx, cy
                let f = reader.read_f64::<LittleEndian>()? as f32;
                let cx = reader.read_f64::<LittleEndian>()? as f32;
                let cy = reader.read_f64::<LittleEndian>()? as f32;

                Camera::new(
                    f,
                    f, // fx = fy for simple pinhole
                    cx,
                    cy,
                    width as u32,
                    height as u32,
                    Matrix3::identity(), // Will be filled from images.bin
                    Vector3::zeros(),
                )
            }
            1 => {
                // PINHOLE: fx, fy, cx, cy
                let fx = reader.read_f64::<LittleEndian>()? as f32;
                let fy = reader.read_f64::<LittleEndian>()? as f32;
                let cx = reader.read_f64::<LittleEndian>()? as f32;
                let cy = reader.read_f64::<LittleEndian>()? as f32;

                Camera::new(
                    fx,
                    fy,
                    cx,
                    cy,
                    width as u32,
                    height as u32,
                    Matrix3::identity(),
                    Vector3::zeros(),
                )
            }
            2 | 3 => {
                // SIMPLE_RADIAL / RADIAL: f, cx, cy, k...
                has_distortion_models = true;
                let f = reader.read_f64::<LittleEndian>()? as f32;
                let cx = reader.read_f64::<LittleEndian>()? as f32;
                let cy = reader.read_f64::<LittleEndian>()? as f32;

                // Skip distortion parameters
                let num_distortion = if model_id == 2 { 1 } else { 2 };
                for _ in 0..num_distortion {
                    reader.read_f64::<LittleEndian>()?;
                }

                Camera::new(
                    f,
                    f, // fy = fx for radial models
                    cx,
                    cy,
                    width as u32,
                    height as u32,
                    Matrix3::identity(),
                    Vector3::zeros(),
                )
            }
            4 => {
                // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                has_distortion_models = true;
                let fx = reader.read_f64::<LittleEndian>()? as f32;
                let fy = reader.read_f64::<LittleEndian>()? as f32;
                let cx = reader.read_f64::<LittleEndian>()? as f32;
                let cy = reader.read_f64::<LittleEndian>()? as f32;

                // Skip distortion: k1, k2, p1, p2
                for _ in 0..4 {
                    reader.read_f64::<LittleEndian>()?;
                }

                Camera::new(
                    fx,
                    fy,
                    cx,
                    cy,
                    width as u32,
                    height as u32,
                    Matrix3::identity(),
                    Vector3::zeros(),
                )
            }
            5 => {
                // OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
                has_distortion_models = true;
                let fx = reader.read_f64::<LittleEndian>()? as f32;
                let fy = reader.read_f64::<LittleEndian>()? as f32;
                let cx = reader.read_f64::<LittleEndian>()? as f32;
                let cy = reader.read_f64::<LittleEndian>()? as f32;

                // Skip distortion: k1, k2, k3, k4 (4 params, not 12!)
                for _ in 0..4 {
                    reader.read_f64::<LittleEndian>()?;
                }

                Camera::new(
                    fx,
                    fy,
                    cx,
                    cy,
                    width as u32,
                    height as u32,
                    Matrix3::identity(),
                    Vector3::zeros(),
                )
            }
            _ => return Err(LoadError::UnsupportedCameraModel(model_id)),
        };

        cameras.insert(camera_id, camera);
    }

    // Warn if distortion parameters were ignored
    if has_distortion_models {
        eprintln!();
        eprintln!("⚠️  WARNING: COLMAP distortion parameters ignored");
        eprintln!("   Detected non-pinhole camera models (RADIAL, OPENCV, or FISHEYE).");
        eprintln!("   This implementation assumes images are already undistorted.");
        eprintln!();
        eprintln!("   If using raw distorted images, projections will be systematically wrong.");
        eprintln!("   To fix: Use COLMAP's image_undistorter to preprocess your dataset:");
        eprintln!("     colmap image_undistorter \\");
        eprintln!("       --image_path /path/to/images \\");
        eprintln!("       --input_path /path/to/sparse/0 \\");
        eprintln!("       --output_path /path/to/undistorted");
        eprintln!();
    }

    Ok(cameras)
}

/// Read images.bin file.
///
/// Binary format (little-endian):
/// - num_images: u64
/// - For each image:
///   - image_id: u32
///   - qw, qx, qy, qz: f64 (rotation quaternion)
///   - tx, ty, tz: f64 (translation)
///   - camera_id: u32
///   - name: null-terminated string
///   - (keypoints data - we can skip for now)
fn read_images_bin(path: &Path) -> Result<Vec<ImageInfo>, LoadError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let num_images = reader.read_u64::<LittleEndian>()?;
    let mut images = Vec::with_capacity(num_images as usize);

    for _ in 0..num_images {
        let image_id = reader.read_u32::<LittleEndian>()?;

        // Read quaternion (qw, qx, qy, qz)
        let qw = reader.read_f64::<LittleEndian>()? as f32;
        let qx = reader.read_f64::<LittleEndian>()? as f32;
        let qy = reader.read_f64::<LittleEndian>()? as f32;
        let qz = reader.read_f64::<LittleEndian>()? as f32;

        // Read translation (tx, ty, tz)
        let tx = reader.read_f64::<LittleEndian>()? as f32;
        let ty = reader.read_f64::<LittleEndian>()? as f32;
        let tz = reader.read_f64::<LittleEndian>()? as f32;

        let camera_id = reader.read_u32::<LittleEndian>()?;

        // Read null-terminated image name
        let mut name_bytes = Vec::new();
        loop {
            let byte = reader.read_u8()?;
            if byte == 0 {
                break;
            }
            name_bytes.push(byte);
        }
        let name = String::from_utf8(name_bytes)
            .map_err(|e| LoadError::InvalidFormat(format!("Invalid UTF-8 in image name: {}", e)))?;

        // Read 2D points (we skip these for now, but need to read them to advance the file pointer)
        let num_points2d = reader.read_u64::<LittleEndian>()?;
        // Each 2D point is: x (f64), y (f64), point3d_id (u64) = 24 bytes
        for _ in 0..num_points2d {
            reader.read_f64::<LittleEndian>()?; // x
            reader.read_f64::<LittleEndian>()?; // y
            reader.read_u64::<LittleEndian>()?; // point3d_id
        }

        // Create quaternion (nalgebra uses (w, x, y, z) order internally)
        let rotation =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz).normalize());

        let translation = Vector3::new(tx, ty, tz);

        images.push(ImageInfo {
            id: image_id,
            camera_id,
            name,
            rotation,
            translation,
        });
    }

    Ok(images)
}

/// Read points3D.bin file.
///
/// Binary format (little-endian):
/// - num_points: u64
/// - For each point:
///   - point_id: u64
///   - x, y, z: f64 (position)
///   - r, g, b: u8 (color)
///   - error: f64 (reprojection error)
///   - (track data - we can skip for now)
fn read_points3d_bin(path: &Path) -> Result<Vec<Point3D>, LoadError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let num_points = reader.read_u64::<LittleEndian>()?;
    let mut points = Vec::with_capacity(num_points as usize);

    for _ in 0..num_points {
        let point_id = reader.read_u64::<LittleEndian>()?;

        // Read position (x, y, z)
        let x = reader.read_f64::<LittleEndian>()? as f32;
        let y = reader.read_f64::<LittleEndian>()? as f32;
        let z = reader.read_f64::<LittleEndian>()? as f32;

        // Read color (r, g, b)
        let r = reader.read_u8()?;
        let g = reader.read_u8()?;
        let b = reader.read_u8()?;

        // Read error
        let error = reader.read_f64::<LittleEndian>()? as f32;

        // Read track (list of image observations)
        let track_length = reader.read_u64::<LittleEndian>()?;
        // Each track element is: image_id (u32), point2d_idx (u32) = 8 bytes
        for _ in 0..track_length {
            reader.read_u32::<LittleEndian>()?; // image_id
            reader.read_u32::<LittleEndian>()?; // point2d_idx
        }

        points.push(Point3D {
            id: point_id,
            position: Vector3::new(x, y, z),
            color: [r, g, b],
            error,
        });
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Remove when implementing
    fn test_load_colmap_scene() {
        // TODO: Add test with fixture data
        // See tests/fixtures/ directory
    }
}
