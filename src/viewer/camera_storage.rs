//! Camera position storage for the viewer.
//!
//! Allows users to save and reload camera positions for any loaded model.
//! Camera data is stored in JSON format next to the model file.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedCamera {
    pub id: u32,
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_deg: f32,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub timestamp: u64, // Unix timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CameraStorage {
    pub cameras: Vec<SavedCamera>,
    #[serde(default)]
    pub next_id: u32,
}

impl CameraStorage {
    /// Get the camera storage file path for a given model file.
    ///
    /// For example: "path/to/model.gs" → "path/to/model.gs.cameras.json"
    fn get_storage_path(model_path: &str) -> PathBuf {
        PathBuf::from(format!("{}.cameras.json", model_path))
    }

    /// Load camera storage from disk for a given model file.
    ///
    /// Returns empty storage if file doesn't exist.
    pub fn load(model_path: &str) -> Result<Self, String> {
        let path = Self::get_storage_path(model_path);

        if !path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read camera storage: {}", e))?;

        serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse camera storage: {}", e))
    }

    /// Save camera storage to disk for a given model file.
    pub fn save(&self, model_path: &str) -> Result<(), String> {
        let path = Self::get_storage_path(model_path);

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize camera storage: {}", e))?;

        fs::write(&path, json)
            .map_err(|e| format!("Failed to write camera storage: {}", e))?;

        Ok(())
    }

    /// Add a new camera to storage.
    ///
    /// Returns the assigned ID.
    pub fn add_camera(
        &mut self,
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        fov_y_deg: f32,
        width: u32,
        height: u32,
        name: Option<String>,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.cameras.push(SavedCamera {
            id,
            position,
            yaw,
            pitch,
            fov_y_deg,
            width,
            height,
            name,
            timestamp,
        });

        id
    }

    /// Get a camera by ID.
    pub fn get_camera(&self, id: u32) -> Option<&SavedCamera> {
        self.cameras.iter().find(|c| c.id == id)
    }

    /// Delete a camera by ID.
    pub fn delete_camera(&mut self, id: u32) -> bool {
        if let Some(pos) = self.cameras.iter().position(|c| c.id == id) {
            self.cameras.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get all cameras sorted by ID.
    pub fn list_cameras(&self) -> Vec<&SavedCamera> {
        let mut cameras: Vec<&SavedCamera> = self.cameras.iter().collect();
        cameras.sort_by_key(|c| c.id);
        cameras
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_storage_empty() {
        let storage = CameraStorage::default();
        assert_eq!(storage.cameras.len(), 0);
        assert_eq!(storage.next_id, 0);
    }

    #[test]
    fn test_add_and_get_camera() {
        let mut storage = CameraStorage::default();

        let id = storage.add_camera(
            [1.0, 2.0, 3.0],
            0.5,
            0.3,
            60.0,
            640,
            480,
            Some("Test Camera".to_string()),
        );

        assert_eq!(id, 0);
        assert_eq!(storage.next_id, 1);

        let camera = storage.get_camera(id).unwrap();
        assert_eq!(camera.position, [1.0, 2.0, 3.0]);
        assert_eq!(camera.yaw, 0.5);
        assert_eq!(camera.pitch, 0.3);
        assert_eq!(camera.fov_y_deg, 60.0);
        assert_eq!(camera.name.as_deref(), Some("Test Camera"));
    }

    #[test]
    fn test_delete_camera() {
        let mut storage = CameraStorage::default();

        let id1 = storage.add_camera([0.0, 0.0, 0.0], 0.0, 0.0, 60.0, 640, 480, None);
        let id2 = storage.add_camera([1.0, 1.0, 1.0], 0.5, 0.5, 60.0, 640, 480, None);

        assert_eq!(storage.cameras.len(), 2);

        assert!(storage.delete_camera(id1));
        assert_eq!(storage.cameras.len(), 1);

        assert_eq!(storage.get_camera(id1), None);
        assert!(storage.get_camera(id2).is_some());

        assert!(!storage.delete_camera(id1)); // Already deleted
    }

    #[test]
    fn test_list_cameras_sorted() {
        let mut storage = CameraStorage::default();

        let id2 = storage.add_camera([2.0, 0.0, 0.0], 0.0, 0.0, 60.0, 640, 480, None);
        let id0 = storage.add_camera([0.0, 0.0, 0.0], 0.0, 0.0, 60.0, 640, 480, None);
        let id1 = storage.add_camera([1.0, 0.0, 0.0], 0.0, 0.0, 60.0, 640, 480, None);

        // Delete and re-add to create non-sequential IDs
        storage.delete_camera(id0);
        let id3 = storage.add_camera([0.0, 0.0, 0.0], 0.0, 0.0, 60.0, 640, 480, None);

        let cameras = storage.list_cameras();
        let ids: Vec<u32> = cameras.iter().map(|c| c.id).collect();

        // Should be sorted: [1, 2, 3]
        assert_eq!(ids, vec![id2, id1, id3]);
    }

    #[test]
    fn test_storage_path_generation() {
        let path = CameraStorage::get_storage_path("/path/to/model.gs");
        assert_eq!(path, PathBuf::from("/path/to/model.gs.cameras.json"));

        let path = CameraStorage::get_storage_path("runs/output/model.gs");
        assert_eq!(path, PathBuf::from("runs/output/model.gs.cameras.json"));
    }
}
