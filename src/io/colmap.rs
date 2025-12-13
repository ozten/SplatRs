//! COLMAP binary format parser.
//!
//! COLMAP stores sparse reconstruction in binary files:
//! - cameras.bin: Camera intrinsics
//! - images.bin: Camera poses (extrinsics) + 2D keypoints
//! - points3D.bin: 3D points from structure-from-motion
//!
//! Format spec: https://colmap.github.io/format.html

use crate::core::Camera;
use nalgebra::Vector3;
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
    /// Camera parameters (intrinsics)
    pub cameras: Vec<Camera>,

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
    // TODO: Implement for M1
    // This is the first real milestone!
    //
    // Steps:
    // 1. Read cameras.bin (parse camera intrinsics)
    // 2. Read images.bin (parse camera poses)
    // 3. Read points3D.bin (parse 3D points)
    // 4. Match images to cameras by camera_id
    // 5. Return ColmapScene

    unimplemented!("See M1 - COLMAP binary parser")
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
fn read_cameras_bin(path: &Path) -> Result<Vec<Camera>, LoadError> {
    // TODO: Implement for M1
    unimplemented!("See M1 - cameras.bin parser")
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
    // TODO: Implement for M1
    unimplemented!("See M1 - images.bin parser")
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
    // TODO: Implement for M1
    unimplemented!("See M1 - points3D.bin parser")
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
