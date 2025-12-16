//! I/O operations for loading and saving data.
//!
//! This module handles all file format parsing and export:
//! - COLMAP binary format (cameras, images, points3D)
//! - PLY format (Gaussian clouds and meshes)
//! - OBJ format (mesh export)
//! - Checkpoints (training state)

mod colmap;
mod obj;
mod ply;

// Re-export public types and functions
pub use colmap::{load_colmap_scene, ColmapScene, ImageInfo, LoadError, Point3D};
pub use obj::save_obj;
pub use ply::{load_ply, save_colmap_points_ply, save_ply};
