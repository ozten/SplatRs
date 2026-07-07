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
use rayon::prelude::*;
use std::collections::HashMap;

fn gaussian_from_colmap_point(point: &Point3D) -> Gaussian {
    // Position
    let position = point.position;

    // Small uniform scale (log-space, so exp(scale) = actual size)
    // Start with scale that gives ~0.01 unit radius
    let scale = Vector3::new(-4.6, -4.6, -4.6); // exp(-4.6) ≈ 0.01

    // Identity rotation
    let rotation = UnitQuaternion::identity();

    // Initial opacity ~0.1 in logit space (reference 3DGS convention: inverse_sigmoid(0.1) ≈ -2.197).
    // Starting near-transparent lets the optimizer add opacity only where the scene needs it, and
    // lets gradient/densification signal reach Gaussians occluded behind the first surface.
    // (Was 2.2 ≈ sigmoid 0.9 — near-opaque, which saturated alpha-compositing from iteration 0.)
    let opacity = crate::core::inverse_sigmoid(0.1);

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

/// Per-point mean squared distance to the `k` nearest neighbors.
///
/// This is the quantity reference 3DGS (`simple-knn` / `distCUDA2`, k = 3) uses to set each
/// Gaussian's initial scale: dense regions get small splats, sparse regions larger ones.
/// Implemented as a uniform voxel grid with an expanding Chebyshev-ring search, so no
/// external KD-tree dependency is needed. O(n) grid build, near-O(1) query per point for
/// realistic COLMAP clouds.
pub fn mean_sq_dist_knn(positions: &[Vector3<f32>], k: usize) -> Vec<f32> {
    let n = positions.len();
    if n <= 1 || k == 0 {
        return vec![0.0; n];
    }
    let k = k.min(n - 1);

    let mut min = positions[0];
    let mut max = positions[0];
    for p in positions {
        min = min.inf(p);
        max = max.sup(p);
    }
    let extent = max - min;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1e-6);

    // Cell size: aim for a few points per occupied cell. COLMAP clouds are mostly
    // surface-like, so size cells off the largest bbox axis rather than a volumetric
    // estimate (which would under-size cells for flat scenes).
    let cells_per_axis = ((n as f32 / 4.0).cbrt().ceil() as i64).max(1);
    let cell = max_extent / cells_per_axis as f32;

    let cell_of = |p: &Vector3<f32>| -> (i64, i64, i64) {
        (
            ((p.x - min.x) / cell).floor() as i64,
            ((p.y - min.y) / cell).floor() as i64,
            ((p.z - min.z) / cell).floor() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        grid.entry(cell_of(p)).or_default().push(i as u32);
    }

    // Searching rings up to this radius is guaranteed to have visited every occupied cell.
    let max_ring = cells_per_axis + 1;

    positions
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            // Best-k squared distances, kept sorted ascending (k is tiny, so sort-on-insert is fine).
            let mut best: Vec<f32> = Vec::with_capacity(k + 1);
            let (cx, cy, cz) = cell_of(p);
            let mut r: i64 = 0;
            loop {
                // Visit cells on the Chebyshev shell at radius r.
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                continue;
                            }
                            let Some(cell_pts) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                                continue;
                            };
                            for &j in cell_pts {
                                if j as usize == i {
                                    continue;
                                }
                                let d2 = (positions[j as usize] - p).norm_squared();
                                if best.len() < k {
                                    best.push(d2);
                                    best.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                } else if d2 < best[k - 1] {
                                    best[k - 1] = d2;
                                    best.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                }
                            }
                        }
                    }
                }
                // Any point in an unvisited cell (Chebyshev distance > r) is at least r·cell
                // away, so once the current k-th best is within that bound we are done.
                let searched = r as f32 * cell;
                if best.len() == k && best[k - 1] <= searched * searched {
                    break;
                }
                if r > max_ring {
                    break;
                }
                r += 1;
            }
            if best.is_empty() {
                0.0
            } else {
                best.iter().sum::<f32>() / best.len() as f32
            }
        })
        .collect()
}

/// C1: density-adaptive initial scale (reference 3DGS): per point, isotropic
/// `σ = sqrt(mean sq dist to 3 nearest neighbors)`, stored in log space. Replaces the old
/// depth heuristic (`1.5·z/f`), which sized Gaussians by distance from one camera instead of
/// local point density, making dense regions start too large and over-split during
/// densification. `min_sigma`/`max_sigma` bound σ in world units.
pub fn apply_knn_init_scales(gaussians: &mut [Gaussian], min_sigma: f32, max_sigma: f32) {
    let positions: Vec<Vector3<f32>> = gaussians.iter().map(|g| g.position).collect();
    let d2 = mean_sq_dist_knn(&positions, 3);
    for (g, d2) in gaussians.iter_mut().zip(d2) {
        let sigma = d2.max(1e-7).sqrt().clamp(min_sigma, max_sigma);
        g.scale = Vector3::from_element(sigma.ln());
    }
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

    fn brute_force_mean_sq_dist_knn(positions: &[Vector3<f32>], k: usize) -> Vec<f32> {
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut d2: Vec<f32> = positions
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, q)| (q - p).norm_squared())
                    .collect();
                d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let k = k.min(d2.len());
                if k == 0 {
                    0.0
                } else {
                    d2[..k].iter().sum::<f32>() / k as f32
                }
            })
            .collect()
    }

    #[test]
    fn test_mean_sq_dist_knn_line() {
        // Points on a line, spacing 1: 3-NN sq dists at an interior point are 1, 1, 4 → mean 2.
        let positions: Vec<Vector3<f32>> = (0..10)
            .map(|i| Vector3::new(i as f32, 0.0, 0.0))
            .collect();
        let d2 = mean_sq_dist_knn(&positions, 3);
        for i in 2..8 {
            assert!((d2[i] - 2.0).abs() < 1e-5, "interior point {}: {}", i, d2[i]);
        }
        // Endpoint: neighbors at 1, 2, 3 → (1 + 4 + 9)/3
        assert!((d2[0] - 14.0 / 3.0).abs() < 1e-5, "endpoint: {}", d2[0]);
    }

    #[test]
    fn test_mean_sq_dist_knn_matches_brute_force() {
        // Deterministic pseudo-random cloud, non-uniform density (cluster + sparse shell).
        let mut positions = Vec::new();
        let mut state = 0x12345678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        for i in 0..500 {
            let s = if i % 5 == 0 { 10.0 } else { 1.0 };
            positions.push(Vector3::new(s * next(), s * next(), s * next()));
        }
        let fast = mean_sq_dist_knn(&positions, 3);
        let slow = brute_force_mean_sq_dist_knn(&positions, 3);
        for i in 0..positions.len() {
            assert!(
                (fast[i] - slow[i]).abs() <= 1e-4 * slow[i].max(1e-6),
                "point {}: grid={} brute={}",
                i,
                fast[i],
                slow[i]
            );
        }
    }

    #[test]
    fn test_mean_sq_dist_knn_degenerate() {
        // Empty, single point, duplicates: must not panic, must return finite values.
        assert!(mean_sq_dist_knn(&[], 3).is_empty());
        assert_eq!(mean_sq_dist_knn(&[Vector3::zeros()], 3), vec![0.0]);
        let dup = vec![Vector3::new(1.0, 2.0, 3.0); 4];
        for d2 in mean_sq_dist_knn(&dup, 3) {
            assert_eq!(d2, 0.0);
        }
    }

    #[test]
    fn test_apply_knn_init_scales() {
        let points: Vec<Point3D> = (0..5)
            .map(|i| Point3D {
                id: i,
                position: Vector3::new(i as f32 * 0.5, 0.0, 0.0),
                color: [128, 128, 128],
                error: 0.1,
            })
            .collect();
        let mut gaussians = init_from_colmap_points(&points).gaussians;
        apply_knn_init_scales(&mut gaussians, 1e-4, 1.0);
        for g in &gaussians {
            // Interior spacing 0.5 → σ ∈ [0.5, 1.0]-ish; all finite, isotropic, within clamps.
            assert!(g.scale.x.is_finite());
            assert_eq!(g.scale.x, g.scale.y);
            assert_eq!(g.scale.x, g.scale.z);
            let sigma = g.scale.x.exp();
            assert!((1e-4..=1.0).contains(&sigma), "sigma = {}", sigma);
        }
    }
}
