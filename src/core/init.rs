//! Initialization utilities for Gaussian clouds.
//!
//! Functions to initialize Gaussians from various sources:
//! - COLMAP point clouds
//! - Random initialization
//! - Custom point clouds

use crate::core::color::srgb_u8_to_linear_f32;
use crate::core::{Camera, Gaussian, GaussianCloud};
use crate::io::Point3D;
use nalgebra::{UnitQuaternion, Vector3};

fn gaussian_from_colmap_point(point: &Point3D) -> Gaussian {
    // Position
    let position = point.position;

    // Small uniform scale (log-space, so exp(scale) = actual size)
    // Start with scale that gives ~0.01 unit radius
    let scale = Vector3::new(-4.6, -4.6, -4.6); // exp(-4.6) ≈ 0.01

    // Identity rotation
    let rotation = UnitQuaternion::identity();

    // Full opacity in logit space: inverse_sigmoid(0.9) ≈ 2.2
    let opacity = 2.2;

    // Convert RGB color (0-255) to SH DC coefficient (0-1)
    // COLMAP colors are in sRGB space, so we must convert to linear before
    // storing in SH coefficients (which represent linear radiance).
    // For spherical harmonics, the DC coefficient is color / Y_0^0
    // where Y_0^0 = 0.28209479
    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0] = [
        srgb_u8_to_linear_f32(point.color[0]) / 0.28209479,
        srgb_u8_to_linear_f32(point.color[1]) / 0.28209479,
        srgb_u8_to_linear_f32(point.color[2]) / 0.28209479,
    ];

    Gaussian::new(position, scale, rotation, opacity, sh_coeffs)
}

/// Initialize Gaussians from COLMAP 3D points.
///
/// Creates one Gaussian per point with:
/// - Position from point location
/// - Color from point RGB (stored in DC SH coefficient)
/// - Identity rotation
/// - Small uniform scale
/// - Full opacity
pub fn init_from_colmap_points(points: &[Point3D]) -> GaussianCloud {
    let gaussians: Vec<Gaussian> = points.iter().map(gaussian_from_colmap_point).collect();

    GaussianCloud::from_gaussians(gaussians)
}

/// Initialize Gaussians from COLMAP 3D points, but keep only points that are
/// inside the given camera's image bounds.
///
/// To avoid "all the Gaussians end up in one part of the image", this uses a
/// simple *screen-space stratified sampling* strategy:
/// - Project all points into the view.
/// - Bin them into tiles in pixel space.
/// - Pick points round-robin across tiles, preferring nearer points within a tile.
///
/// This is useful for early single-image debugging (M7), where we want coverage
/// across the whole frame and we don't yet optimize geometry.
pub fn init_from_colmap_points_visible_stratified(
    points: &[Point3D],
    camera: &Camera,
    max_gaussians: usize,
    tile_size_px: u32,
) -> GaussianCloud {
    let tile_size_px = tile_size_px.max(1) as f32;
    let tiles_x = ((camera.width as f32) / tile_size_px).ceil().max(1.0) as usize;
    let tiles_y = ((camera.height as f32) / tile_size_px).ceil().max(1.0) as usize;
    let tile_count = tiles_x * tiles_y;

    #[derive(Clone)]
    struct Candidate<'a> {
        depth: f32,
        point: &'a Point3D,
    }

    let mut tiles: Vec<Vec<Candidate<'_>>> = (0..tile_count).map(|_| Vec::new()).collect();

    for point in points {
        let p_cam = camera.world_to_camera(&point.position);
        if p_cam.z <= 0.0 {
            continue;
        }
        let Some(px) = camera.project(&p_cam) else {
            continue;
        };
        if px.x < 0.0 || px.x >= camera.width as f32 || px.y < 0.0 || px.y >= camera.height as f32 {
            continue;
        }

        let tx = (px.x / tile_size_px).floor() as usize;
        let ty = (px.y / tile_size_px).floor() as usize;
        let tx = tx.min(tiles_x - 1);
        let ty = ty.min(tiles_y - 1);
        let tid = ty * tiles_x + tx;
        tiles[tid].push(Candidate {
            depth: p_cam.z,
            point,
        });
    }

    // Sort each tile by depth (front-to-back). This is mostly to avoid selecting
    // far background points if a tile is crowded.
    for t in &mut tiles {
        t.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Round-robin across tiles to get a roughly uniform screen-space distribution.
    let mut tile_indices = vec![0usize; tile_count];
    let mut selected_points: Vec<&Point3D> = Vec::with_capacity(max_gaussians.min(points.len()));
    loop {
        if selected_points.len() >= max_gaussians {
            break;
        }
        let mut added_this_round = 0usize;
        for tid in 0..tile_count {
            if selected_points.len() >= max_gaussians {
                break;
            }
            let idx = tile_indices[tid];
            if idx < tiles[tid].len() {
                selected_points.push(tiles[tid][idx].point);
                tile_indices[tid] = idx + 1;
                added_this_round += 1;
            }
        }
        if added_this_round == 0 {
            break;
        }
    }

    let gaussians: Vec<Gaussian> = selected_points
        .into_iter()
        .map(gaussian_from_colmap_point)
        .collect();

    GaussianCloud::from_gaussians(gaussians)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_from_points() {
        let points = vec![
            Point3D {
                id: 0,
                position: Vector3::new(1.0, 2.0, 3.0),
                color: [255, 128, 64],
                error: 0.1,
            },
            Point3D {
                id: 1,
                position: Vector3::new(4.0, 5.0, 6.0),
                color: [100, 200, 50],
                error: 0.2,
            },
        ];

        let cloud = init_from_colmap_points(&points);

        assert_eq!(cloud.len(), 2);

        // Check first Gaussian
        let g0 = &cloud.gaussians[0];
        assert_eq!(g0.position, Vector3::new(1.0, 2.0, 3.0));
        assert!(g0.scale.x < 0.0); // Should be in log-space

        // Check SH DC component is set
        assert!(g0.sh_coeffs[0][0] > 0.0);
    }

    #[test]
    fn test_init_visible_stratified_filters_out_of_frame() {
        let camera = Camera::new(
            100.0,
            100.0,
            50.0,
            50.0,
            100,
            100,
            nalgebra::Matrix3::identity(),
            Vector3::new(0.0, 0.0, 0.0),
        );

        // One point in front of camera, one behind.
        let points = vec![
            Point3D {
                id: 0,
                position: Vector3::new(0.0, 0.0, 2.0),
                color: [255, 0, 0],
                error: 0.1,
            },
            Point3D {
                id: 1,
                position: Vector3::new(0.0, 0.0, -2.0),
                color: [0, 255, 0],
                error: 0.1,
            },
        ];

        let cloud = init_from_colmap_points_visible_stratified(&points, &camera, 10, 16);
        assert_eq!(cloud.len(), 1);
    }
}
