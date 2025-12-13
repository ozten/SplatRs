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
/// For visualization and debugging, we can export Gaussians as point clouds
/// where each point represents a Gaussian center with its color.
///
/// Later (M10), we'll extend this to save full Gaussian parameters.
pub fn save_ply(cloud: &GaussianCloud, path: &Path) -> Result<(), LoadError> {
    // TODO: Implement for M10
    // For M10, save full Gaussian parameters (scale, rotation, opacity, SH)
    unimplemented!("See M10 - PLY export for Gaussian clouds")
}

/// Load a Gaussian cloud from PLY format.
///
/// This will be needed for:
/// - Loading trained models
/// - Loading initial point clouds
pub fn load_ply(path: &Path) -> Result<GaussianCloud, LoadError> {
    // TODO: Implement for M10
    unimplemented!("See M10 - PLY import")
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
