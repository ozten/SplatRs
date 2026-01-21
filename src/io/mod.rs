//! I/O operations for loading and saving data.
//!
//! This module handles all file format parsing and export:
//! - COLMAP binary format (cameras, images, points3D)
//! - PLY format (Gaussian clouds and meshes)
//! - OBJ format (mesh export)
//! - Model format (.gs files for trained Gaussian Splatting models)
//! - Checkpoints (training state)
//! - Training metadata (JSONL checkpoint logs)

mod colmap;
mod color_management;
mod model;
mod obj;
mod ply;
mod training_metadata;

// Re-export public types and functions
pub use colmap::{load_colmap_scene, ColmapScene, ImageInfo, LoadError, Point3D};
pub use color_management::load_image_to_srgb;
pub use model::{
    compute_bounds, load_model, save_model, Compression, ModelError, ModelMetadata,
};
pub use obj::save_obj;
pub use ply::{load_ply, save_colmap_points_ply, save_ply};
pub use training_metadata::{
    filter_by_iteration, get_latest_checkpoint, read_training_history, CheckpointRecord,
    MetadataError, SceneBounds,
};
