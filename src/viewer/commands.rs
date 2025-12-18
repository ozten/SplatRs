use crate::core::Camera;
use crate::io;
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
}

#[derive(Serialize)]
pub struct ModelMetadataResponse {
    pub num_gaussians: usize,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub center: [f32; 3],
    pub suggested_camera_distance: f32,
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

#[tauri::command]
pub async fn list_models() -> Result<Vec<ModelInfo>, String> {
    println!("[VIEWER] list_models called");
    let test_output = PathBuf::from("test_output");
    if !test_output.exists() {
        println!("[VIEWER] test_output directory does not exist");
        return Ok(vec![]);
    }

    let mut models = vec![];
    for entry in fs::read_dir(test_output).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gs") {
            models.push(ModelInfo {
                name: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                path: path.to_str().unwrap_or("").to_string(),
            });
        }
    }
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

    *state.model.lock().unwrap() = Some((cloud, metadata));

    Ok(ModelMetadataResponse {
        num_gaussians,
        bounds_min: [min.x, min.y, min.z],
        bounds_max: [max.x, max.y, max.z],
        center: [center.x, center.y, center.z],
        suggested_camera_distance: suggested_distance,
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
