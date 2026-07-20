//! Training metadata reader for JSONL checkpoint logs.
//!
//! This module provides functions to read and query training_history.jsonl files
//! that track checkpoint data during Gaussian Splatting training.
//!
//! File format: `training_history.jsonl` (append-only JSONL)
//!
//! Each line is a JSON object with per-checkpoint fields:
//! - timestamp: ISO 8601 timestamp string
//! - iteration: training iteration number
//! - ply_filename: path to the saved .ply or .gs file
//! - psnr: Peak Signal-to-Noise Ratio (dB)
//! - scene_bounds: object with min/max arrays [x, y, z]
//! - training_width: image width used for training
//! - training_height: image height used for training
//! - downsample_factor: resolution scale factor (e.g., 0.25 = 25%)
//! - dataset_path: path to the training dataset

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Scene bounding box
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl SceneBounds {
    pub fn new(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        Self {
            min: [min.x, min.y, min.z],
            max: [max.x, max.y, max.z],
        }
    }

    pub fn min_vector(&self) -> Vector3<f32> {
        Vector3::new(self.min[0], self.min[1], self.min[2])
    }

    pub fn max_vector(&self) -> Vector3<f32> {
        Vector3::new(self.max[0], self.max[1], self.max[2])
    }
}

/// Training checkpoint record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub timestamp: String,
    pub iteration: u64,
    pub ply_filename: String,
    pub psnr: f32,
    pub scene_bounds: SceneBounds,
    pub training_width: u32,
    pub training_height: u32,
    pub downsample_factor: f32,
    pub dataset_path: String,
}

/// Error type for training metadata operations
#[derive(Debug)]
pub enum MetadataError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotFound(String),
}

impl From<std::io::Error> for MetadataError {
    fn from(e: std::io::Error) -> Self {
        MetadataError::Io(e)
    }
}

impl From<serde_json::Error> for MetadataError {
    fn from(e: serde_json::Error) -> Self {
        MetadataError::Json(e)
    }
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::Io(e) => write!(f, "I/O error: {}", e),
            MetadataError::Json(e) => write!(f, "JSON parse error: {}", e),
            MetadataError::NotFound(msg) => write!(f, "Not found: {}", msg),
        }
    }
}

impl std::error::Error for MetadataError {}

/// Read all checkpoint records from a training_history.jsonl file
///
/// # Arguments
/// * `path` - Path to the training_history.jsonl file
///
/// # Returns
/// Vector of checkpoint records in chronological order
///
/// # Example
/// ```no_run
/// use sugar_rs::io::read_training_history;
///
/// let records = read_training_history("output/training_history.jsonl").unwrap();
/// println!("Found {} checkpoints", records.len());
/// ```
pub fn read_training_history<P: AsRef<Path>>(path: P) -> Result<Vec<CheckpointRecord>, MetadataError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        let line = line?;
        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }
        let record: CheckpointRecord = serde_json::from_str(&line)?;
        records.push(record);
    }

    Ok(records)
}

/// Get the most recent checkpoint from a training_history.jsonl file
///
/// # Arguments
/// * `path` - Path to the training_history.jsonl file
///
/// # Returns
/// The checkpoint record with the highest iteration number
///
/// # Example
/// ```no_run
/// use sugar_rs::io::get_latest_checkpoint;
///
/// let latest = get_latest_checkpoint("output/training_history.jsonl").unwrap();
/// println!("Latest checkpoint at iteration {}", latest.iteration);
/// ```
pub fn get_latest_checkpoint<P: AsRef<Path>>(path: P) -> Result<CheckpointRecord, MetadataError> {
    let records = read_training_history(path)?;

    records
        .into_iter()
        .max_by_key(|r| r.iteration)
        .ok_or_else(|| MetadataError::NotFound("No checkpoints found in file".to_string()))
}

/// Filter checkpoints by iteration number
///
/// # Arguments
/// * `path` - Path to the training_history.jsonl file
/// * `min_iteration` - Minimum iteration (inclusive), or None for no minimum
/// * `max_iteration` - Maximum iteration (inclusive), or None for no maximum
///
/// # Returns
/// Vector of checkpoint records matching the filter criteria
///
/// # Example
/// ```no_run
/// use sugar_rs::io::filter_by_iteration;
///
/// // Get checkpoints between iterations 1000 and 5000
/// let records = filter_by_iteration("output/training_history.jsonl", Some(1000), Some(5000)).unwrap();
/// println!("Found {} checkpoints in range", records.len());
/// ```
pub fn filter_by_iteration<P: AsRef<Path>>(
    path: P,
    min_iteration: Option<u64>,
    max_iteration: Option<u64>,
) -> Result<Vec<CheckpointRecord>, MetadataError> {
    let records = read_training_history(path)?;

    Ok(records
        .into_iter()
        .filter(|r| {
            let above_min = min_iteration.map_or(true, |min| r.iteration >= min);
            let below_max = max_iteration.map_or(true, |max| r.iteration <= max);
            above_min && below_max
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_jsonl() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        // Write test records
        let records = vec![
            CheckpointRecord {
                timestamp: "2024-01-15T10:30:00Z".to_string(),
                iteration: 1000,
                ply_filename: "checkpoint_1000.gs".to_string(),
                psnr: 22.5,
                scene_bounds: SceneBounds {
                    min: [-1.0, -1.0, -1.0],
                    max: [1.0, 1.0, 1.0],
                },
                training_width: 256,
                training_height: 256,
                downsample_factor: 0.25,
                dataset_path: "data/scene1".to_string(),
            },
            CheckpointRecord {
                timestamp: "2024-01-15T10:35:00Z".to_string(),
                iteration: 2000,
                ply_filename: "checkpoint_2000.gs".to_string(),
                psnr: 25.3,
                scene_bounds: SceneBounds {
                    min: [-1.0, -1.0, -1.0],
                    max: [1.0, 1.0, 1.0],
                },
                training_width: 256,
                training_height: 256,
                downsample_factor: 0.25,
                dataset_path: "data/scene1".to_string(),
            },
            CheckpointRecord {
                timestamp: "2024-01-15T10:40:00Z".to_string(),
                iteration: 3000,
                ply_filename: "checkpoint_3000.gs".to_string(),
                psnr: 27.1,
                scene_bounds: SceneBounds {
                    min: [-1.0, -1.0, -1.0],
                    max: [1.0, 1.0, 1.0],
                },
                training_width: 256,
                training_height: 256,
                downsample_factor: 0.25,
                dataset_path: "data/scene1".to_string(),
            },
        ];

        for record in records {
            let json = serde_json::to_string(&record).unwrap();
            writeln!(file, "{}", json).unwrap();
        }

        file.flush().unwrap();
        file
    }

    #[test]
    fn test_read_training_history() {
        let file = create_test_jsonl();
        let records = read_training_history(file.path()).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].iteration, 1000);
        assert_eq!(records[1].iteration, 2000);
        assert_eq!(records[2].iteration, 3000);

        assert_eq!(records[0].psnr, 22.5);
        assert_eq!(records[1].psnr, 25.3);
        assert_eq!(records[2].psnr, 27.1);
    }

    #[test]
    fn test_get_latest_checkpoint() {
        let file = create_test_jsonl();
        let latest = get_latest_checkpoint(file.path()).unwrap();

        assert_eq!(latest.iteration, 3000);
        assert_eq!(latest.psnr, 27.1);
        assert_eq!(latest.ply_filename, "checkpoint_3000.gs");
    }

    #[test]
    fn test_filter_by_iteration_both_bounds() {
        let file = create_test_jsonl();
        let filtered = filter_by_iteration(file.path(), Some(1500), Some(2500)).unwrap();

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].iteration, 2000);
    }

    #[test]
    fn test_filter_by_iteration_min_only() {
        let file = create_test_jsonl();
        let filtered = filter_by_iteration(file.path(), Some(2000), None).unwrap();

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].iteration, 2000);
        assert_eq!(filtered[1].iteration, 3000);
    }

    #[test]
    fn test_filter_by_iteration_max_only() {
        let file = create_test_jsonl();
        let filtered = filter_by_iteration(file.path(), None, Some(2000)).unwrap();

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].iteration, 1000);
        assert_eq!(filtered[1].iteration, 2000);
    }

    #[test]
    fn test_filter_by_iteration_no_bounds() {
        let file = create_test_jsonl();
        let filtered = filter_by_iteration(file.path(), None, None).unwrap();

        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_scene_bounds_conversion() {
        let min = Vector3::new(-2.0, -3.0, -4.0);
        let max = Vector3::new(5.0, 6.0, 7.0);
        let bounds = SceneBounds::new(min, max);

        assert_eq!(bounds.min, [-2.0, -3.0, -4.0]);
        assert_eq!(bounds.max, [5.0, 6.0, 7.0]);

        let min_vec = bounds.min_vector();
        let max_vec = bounds.max_vector();

        assert_eq!(min_vec, min);
        assert_eq!(max_vec, max);
    }
}
