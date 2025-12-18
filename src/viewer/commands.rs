use crate::core::{Camera, quaternion_to_matrix};
use crate::io::{self, load_colmap_scene};
use crate::viewer::state::AppState;
use image::ImageEncoder;
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::State;

#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    #[serde(skip)]
    pub created_timestamp: u64, // Unix timestamp for sorting (not sent to frontend)
}

#[derive(Serialize)]
pub struct ModelMetadataResponse {
    pub num_gaussians: usize,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub center: [f32; 3],
    pub suggested_camera_distance: f32,
    pub training_width: u32,
    pub training_height: u32,
    pub dataset_path: String,
}

#[derive(Deserialize)]
pub struct CameraParams {
    pub position: [f32; 3],
    pub rotation: [[f32; 3]; 3], // Row-major rotation matrix
    pub width: u32,
    pub height: u32,
    pub fov_y_deg: f32,
}

#[tauri::command]
pub async fn log_to_stdout(message: String) {
    println!("[JS] {}", message);
}

/// Recursively scan a directory for .gs files
fn scan_directory_for_models(dir: &PathBuf, models: &mut Vec<ModelInfo>) {
    if !dir.exists() {
        return;
    }

    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Recursively scan subdirectories
            scan_directory_for_models(&path, models);
        } else if path.extension().and_then(|s| s.to_str()) == Some("gs") {
            // Get file metadata for timestamp
            let timestamp = if let Ok(metadata) = fs::metadata(&path) {
                // Try created time first, fall back to modified time
                metadata
                    .created()
                    .or_else(|_| metadata.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            } else {
                0
            };

            // Create a friendly display name (show parent dir if from runs/)
            let name = if let Some(parent) = path.parent() {
                if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
                    // If parent looks like a timestamped run directory, include it
                    if parent_name.contains('_') && parent.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) == Some("runs") {
                        format!("{}/{}",
                            parent_name,
                            path.file_name().and_then(|n| n.to_str()).unwrap_or("model.gs")
                        )
                    } else {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("model.gs")
                            .to_string()
                    }
                } else {
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("model.gs")
                        .to_string()
                }
            } else {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model.gs")
                    .to_string()
            };

            models.push(ModelInfo {
                name,
                path: path.to_str().unwrap_or("").to_string(),
                created_timestamp: timestamp,
            });
        }
    }
}

#[tauri::command]
pub async fn list_models() -> Result<Vec<ModelInfo>, String> {
    println!("[VIEWER] list_models called");

    let mut models = vec![];

    // Scan runs/ directory (contains timestamped subdirectories)
    let runs_dir = PathBuf::from("runs");
    if runs_dir.exists() {
        println!("[VIEWER] Scanning runs/ directory...");
        scan_directory_for_models(&runs_dir, &mut models);
    }

    // Scan test_output/ directory (legacy location)
    let test_output = PathBuf::from("test_output");
    if test_output.exists() {
        println!("[VIEWER] Scanning test_output/ directory...");
        scan_directory_for_models(&test_output, &mut models);
    }

    // Sort by timestamp descending (newest first)
    models.sort_by(|a, b| b.created_timestamp.cmp(&a.created_timestamp));

    println!("[VIEWER] Found {} models", models.len());
    Ok(models)
}

#[tauri::command]
pub async fn load_model(
    path: String,
    state: State<'_, AppState>,
) -> Result<ModelMetadataResponse, String> {
    let (cloud, metadata) = io::load_model(&path).map_err(|e| e.to_string())?;

    // Compute bounding box for auto-framing
    let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

    for g in &cloud.gaussians {
        min.x = min.x.min(g.position.x);
        min.y = min.y.min(g.position.y);
        min.z = min.z.min(g.position.z);
        max.x = max.x.max(g.position.x);
        max.y = max.y.max(g.position.y);
        max.z = max.z.max(g.position.z);
    }

    let center = (min + max) * 0.5;

    let num_gaussians = cloud.gaussians.len();

    // Calculate suggested camera distance (diagonal of bounding box)
    let dx = max.x - min.x;
    let dy = max.y - min.y;
    let dz = max.z - min.z;
    let diagonal = (dx * dx + dy * dy + dz * dz).sqrt();
    let suggested_distance = diagonal * 1.5; // 1.5x for padding

    println!("[VIEWER] Loaded model: {} gaussians", num_gaussians);
    println!("[VIEWER] Bounds min: [{}, {}, {}]", min.x, min.y, min.z);
    println!("[VIEWER] Bounds max: [{}, {}, {}]", max.x, max.y, max.z);
    println!("[VIEWER] Center: [{}, {}, {}]", center.x, center.y, center.z);
    println!("[VIEWER] Suggested distance: {}", suggested_distance);

    // Extract training resolution and dataset path before moving metadata
    let training_width = metadata.training_width;
    let training_height = metadata.training_height;
    let dataset_path = metadata.dataset_path.clone();

    *state.model.lock().unwrap() = Some((cloud, metadata));

    Ok(ModelMetadataResponse {
        num_gaussians,
        bounds_min: [min.x, min.y, min.z],
        bounds_max: [max.x, max.y, max.z],
        center: [center.x, center.y, center.z],
        suggested_camera_distance: suggested_distance,
        training_width,
        training_height,
        dataset_path,
    })
}

#[tauri::command]
pub async fn render_frame(
    camera_params: CameraParams,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let model_guard = state.model.lock().unwrap();
    let (cloud, _) = model_guard
        .as_ref()
        .ok_or("No model loaded")?;

    // Log camera parameters (first few frames)
    static FRAME_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let frame = FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame < 3 {
        println!("[VIEWER] Frame {} render_frame called", frame);
        println!("  position: [{}, {}, {}]", camera_params.position[0], camera_params.position[1], camera_params.position[2]);
        println!("  rotation: {:?}", camera_params.rotation);
        println!("  resolution: {}x{}", camera_params.width, camera_params.height);
        println!("  fov_y: {}°", camera_params.fov_y_deg);
    }

    // Build Camera from frontend parameters
    let fov_y_rad = camera_params.fov_y_deg.to_radians();
    let fy = camera_params.height as f32 / (2.0 * (fov_y_rad / 2.0).tan());
    let fx = fy; // Assume square pixels
    let cx = camera_params.width as f32 / 2.0;
    let cy = camera_params.height as f32 / 2.0;

    // Convert row-major rotation matrix
    let rotation = Matrix3::from_rows(&[
        camera_params.rotation[0].into(),
        camera_params.rotation[1].into(),
        camera_params.rotation[2].into(),
    ]);

    // Translation: t = -R * camera_position
    let cam_pos = Vector3::from(camera_params.position);
    let translation = -rotation * cam_pos;

    let camera = Camera {
        fx,
        fy,
        cx,
        cy,
        width: camera_params.width,
        height: camera_params.height,
        rotation,
        translation,
    };

    // Render on GPU
    let mut renderer = state.renderer.lock().unwrap();
    let background = Vector3::new(0.0, 0.0, 0.0); // Black background
    let pixels = renderer.render(&cloud.gaussians, &camera, &background);

    if frame < 3 {
        println!("[VIEWER] Frame {} rendered {} pixels", frame, pixels.len());
    }

    // Convert linear RGB to sRGB and encode to PNG
    let mut img_buf = image::RgbImage::new(camera.width, camera.height);
    for (i, pixel) in pixels.iter().enumerate() {
        let x = (i % camera.width as usize) as u32;
        let y = (i / camera.width as usize) as u32;

        // Gamma correction: linear -> sRGB
        let r = (pixel.x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        let g = (pixel.y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        let b = (pixel.z.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;

        img_buf.put_pixel(x, y, image::Rgb([r, g, b]));
    }

    // Encode to PNG in memory
    let mut png_bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png_bytes)
        .write_image(
            img_buf.as_raw(),
            camera.width,
            camera.height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| e.to_string())?;

    // Base64 encode for transfer
    use base64::Engine;
    let encoded = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png_bytes)
    );

    if frame < 3 {
        println!("[VIEWER] Frame {} encoded to {} bytes", frame, encoded.len());
    }

    Ok(encoded)
}

#[derive(Serialize)]
pub struct CameraInfo {
    pub position: [f32; 3],
    pub rotation: [[f32; 3]; 3], // Row-major rotation matrix
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

#[tauri::command]
pub async fn get_camera_by_id(
    camera_id: usize,
    dataset_root: String,
) -> Result<CameraInfo, String> {
    println!("[VIEWER] get_camera_by_id called: id={}, root={}", camera_id, dataset_root);

    let dataset_path = PathBuf::from(&dataset_root);
    let sparse_dir = dataset_path.join("sparse/0");

    if !sparse_dir.exists() {
        return Err(format!("COLMAP sparse directory not found: {:?}", sparse_dir));
    }

    // Load COLMAP scene
    let scene = load_colmap_scene(&sparse_dir).map_err(|e| e.to_string())?;

    // Check if camera_id is valid
    if camera_id >= scene.images.len() {
        return Err(format!(
            "Camera ID {} out of range (max: {})",
            camera_id,
            scene.images.len() - 1
        ));
    }

    // Get image info for this camera
    let image_info = &scene.images[camera_id];

    // Get camera intrinsics
    let camera_intrinsics = scene
        .cameras
        .get(&image_info.camera_id)
        .ok_or_else(|| format!("Camera intrinsics not found for ID {}", image_info.camera_id))?;

    // Convert quaternion to rotation matrix
    let rotation_matrix = quaternion_to_matrix(&image_info.rotation);

    // Camera position: try positive sign (COLMAP may use opposite convention)
    let position = rotation_matrix.transpose() * image_info.translation;

    // Convert rotation matrix to row-major format for frontend
    let rotation = [
        [rotation_matrix[(0, 0)], rotation_matrix[(0, 1)], rotation_matrix[(0, 2)]],
        [rotation_matrix[(1, 0)], rotation_matrix[(1, 1)], rotation_matrix[(1, 2)]],
        [rotation_matrix[(2, 0)], rotation_matrix[(2, 1)], rotation_matrix[(2, 2)]],
    ];

    println!(
        "[VIEWER] Camera {} found: pos=[{}, {}, {}]",
        camera_id, position.x, position.y, position.z
    );

    Ok(CameraInfo {
        position: [position.x, position.y, position.z],
        rotation,
        fx: camera_intrinsics.fx,
        fy: camera_intrinsics.fy,
        cx: camera_intrinsics.cx,
        cy: camera_intrinsics.cy,
        width: camera_intrinsics.width,
        height: camera_intrinsics.height,
    })
}
