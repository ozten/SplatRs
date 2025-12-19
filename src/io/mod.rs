//! I/O operations for loading and saving data.
//!
//! This module handles all file format parsing and export:
//! - COLMAP binary format (cameras, images, points3D)
//! - PLY format (Gaussian clouds and meshes)
//! - OBJ format (mesh export)
//! - Model format (.gs files for trained Gaussian Splatting models)
//! - Checkpoints (training state)

mod colmap;
mod color_management;
mod model;
mod obj;
mod ply;

// Re-export public types and functions
pub use colmap::{load_colmap_scene, ColmapScene, ImageInfo, LoadError, Point3D};
pub use color_management::load_image_to_srgb;
pub use model::{
    compute_bounds, load_model, save_model, Compression, ModelError, ModelMetadata,
};
pub use obj::save_obj;
pub use ply::{load_ply, save_colmap_points_ply, save_ply};
