//! I/O operations for loading and saving data.
//!
//! This module handles all file format parsing and export:
//! - COLMAP binary format (cameras, images, points3D)
//! - PLY format (Gaussian clouds and meshes)
//! - OBJ format (mesh export)
//! - Checkpoints (training state)

mod colmap;
mod ply;
mod obj;

// Re-export public types and functions
pub use colmap::{ColmapScene, load_colmap_scene, LoadError};
pub use ply::{save_ply, load_ply};
pub use obj::save_obj;
