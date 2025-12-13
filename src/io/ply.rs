//! PLY format I/O for Gaussian clouds and meshes.
//!
//! PLY (Polygon File Format) is used to:
//! - Export Gaussian clouds for visualization (M1-M2)
//! - Save trained models (M10)
//! - Export extracted meshes (M12)

use crate::core::GaussianCloud;
use crate::io::LoadError;
use std::path::Path;

/// Save a Gaussian cloud to PLY format.
///
/// For visualization and debugging, we can export Gaussians as point clouds
/// where each point represents a Gaussian center with its color.
///
/// Later (M10), we'll extend this to save full Gaussian parameters.
pub fn save_ply(cloud: &GaussianCloud, path: &Path) -> Result<(), LoadError> {
    // TODO: Implement for M1
    // For M1, just save positions and DC colors as a point cloud
    // For M10, save full Gaussian parameters (scale, rotation, opacity, SH)
    unimplemented!("See M1 - PLY export for visualization")
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
