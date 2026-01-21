//! Training metadata logging to JSONL format.
//!
//! This module provides functionality to write checkpoint metadata to an append-only
//! JSONL (JSON Lines) file for tracking training progress over time.

use fs2::FileExt;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

/// Metadata for a single training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// ISO 8601 timestamp when checkpoint was created
    pub timestamp: String,

    /// Training iteration number
    pub iteration: u64,

    /// Filename of the saved PLY file
    pub ply_filename: String,

    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr: f32,

    /// Scene bounding box minimum corner
    pub scene_bounds_min: [f32; 3],

    /// Scene bounding box maximum corner
    pub scene_bounds_max: [f32; 3],

    /// Training image width (after downsampling)
    pub training_width: u32,

    /// Training image height (after downsampling)
    pub training_height: u32,

    /// Downsample factor applied during training (e.g., 0.25 = 25% of original)
    pub downsample_factor: f32,

    /// Path to the dataset used for training
    pub dataset_path: String,
}

impl CheckpointMetadata {
    /// Create a new CheckpointMetadata with the current timestamp
    pub fn new(
        iteration: u64,
        ply_filename: String,
        psnr: f32,
        scene_bounds_min: Vector3<f32>,
        scene_bounds_max: Vector3<f32>,
        training_width: u32,
        training_height: u32,
        downsample_factor: f32,
        dataset_path: String,
    ) -> Self {
        use chrono::Utc;
        let timestamp = Utc::now().to_rfc3339();

        Self {
            timestamp,
            iteration,
            ply_filename,
            psnr,
            scene_bounds_min: [scene_bounds_min.x, scene_bounds_min.y, scene_bounds_min.z],
            scene_bounds_max: [scene_bounds_max.x, scene_bounds_max.y, scene_bounds_max.z],
            training_width,
            training_height,
            downsample_factor,
            dataset_path,
        }
    }
}

/// Error type for metadata operations
#[derive(Debug)]
pub enum MetadataError {
    Io(io::Error),
    Serialization(serde_json::Error),
}

impl From<io::Error> for MetadataError {
    fn from(e: io::Error) -> Self {
        MetadataError::Io(e)
    }
}

impl From<serde_json::Error> for MetadataError {
    fn from(e: serde_json::Error) -> Self {
        MetadataError::Serialization(e)
    }
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::Io(e) => write!(f, "I/O error: {}", e),
            MetadataError::Serialization(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for MetadataError {}

/// Append checkpoint metadata to the training history JSONL file.
///
/// Creates the file if it doesn't exist. Uses file locking to safely
/// handle concurrent writes from multiple processes.
///
/// # Arguments
///
/// * `path` - Path to the JSONL file (typically "training_history.jsonl")
/// * `metadata` - The checkpoint metadata to append
///
/// # Returns
///
/// `Ok(())` on success, or a `MetadataError` on failure.
///
/// # Example
///
/// ```no_run
/// use sugar_rs::io::metadata::{CheckpointMetadata, append_checkpoint_metadata};
/// use nalgebra::Vector3;
///
/// let metadata = CheckpointMetadata::new(
///     1000,
///     "checkpoint_1000.ply".to_string(),
///     28.5,
///     Vector3::new(-1.0, -1.0, -1.0),
///     Vector3::new(1.0, 1.0, 1.0),
///     512,
///     512,
///     0.5,
///     "data/scene".to_string(),
/// );
///
/// append_checkpoint_metadata("training_history.jsonl", &metadata).unwrap();
/// ```
pub fn append_checkpoint_metadata<P: AsRef<Path>>(
    path: P,
    metadata: &CheckpointMetadata,
) -> Result<(), MetadataError> {
    // Serialize metadata to JSON
    let json_line = serde_json::to_string(metadata)?;

    // Open file in append mode, create if it doesn't exist
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    // Lock the file exclusively for writing (cross-platform via fs2)
    file.lock_exclusive()?;

    // Write the JSON line followed by newline
    // Use the file as a Write trait object
    let mut writer = &file;
    writeln!(writer, "{}", json_line)?;

    // Explicitly sync to ensure data is written
    file.sync_all()?;

    // Unlock the file (happens automatically on drop, but we can do it explicitly)
    file.unlock()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use tempfile::NamedTempFile;

    #[test]
    fn test_append_checkpoint_metadata() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create and append first metadata entry
        let metadata1 = CheckpointMetadata::new(
            1000,
            "checkpoint_1000.ply".to_string(),
            28.5,
            Vector3::new(-1.0, -2.0, -3.0),
            Vector3::new(1.0, 2.0, 3.0),
            512,
            512,
            0.5,
            "data/scene1".to_string(),
        );

        append_checkpoint_metadata(path, &metadata1).unwrap();

        // Append second metadata entry
        let metadata2 = CheckpointMetadata::new(
            2000,
            "checkpoint_2000.ply".to_string(),
            30.2,
            Vector3::new(-1.5, -2.5, -3.5),
            Vector3::new(1.5, 2.5, 3.5),
            1024,
            1024,
            0.25,
            "data/scene2".to_string(),
        );

        append_checkpoint_metadata(path, &metadata2).unwrap();

        // Read back and verify
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>().unwrap();

        assert_eq!(lines.len(), 2);

        // Parse first line
        let parsed1: CheckpointMetadata = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed1.iteration, 1000);
        assert_eq!(parsed1.ply_filename, "checkpoint_1000.ply");
        assert_eq!(parsed1.psnr, 28.5);
        assert_eq!(parsed1.training_width, 512);
        assert_eq!(parsed1.training_height, 512);
        assert_eq!(parsed1.downsample_factor, 0.5);
        assert_eq!(parsed1.dataset_path, "data/scene1");

        // Parse second line
        let parsed2: CheckpointMetadata = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(parsed2.iteration, 2000);
        assert_eq!(parsed2.ply_filename, "checkpoint_2000.ply");
        assert_eq!(parsed2.psnr, 30.2);
        assert_eq!(parsed2.training_width, 1024);
        assert_eq!(parsed2.training_height, 1024);
        assert_eq!(parsed2.downsample_factor, 0.25);
        assert_eq!(parsed2.dataset_path, "data/scene2");
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = CheckpointMetadata::new(
            5000,
            "test.ply".to_string(),
            32.1,
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(5.0, 5.0, 5.0),
            256,
            256,
            1.0,
            "data/test".to_string(),
        );

        // Serialize to JSON
        let json = serde_json::to_string(&metadata).unwrap();

        // Deserialize back
        let parsed: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.iteration, 5000);
        assert_eq!(parsed.ply_filename, "test.ply");
        assert_eq!(parsed.psnr, 32.1);
        assert_eq!(parsed.scene_bounds_min, [0.0, 0.0, 0.0]);
        assert_eq!(parsed.scene_bounds_max, [5.0, 5.0, 5.0]);
    }
}
