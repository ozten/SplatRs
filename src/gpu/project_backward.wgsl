// WGSL shader for projection backward pass (2D→3D gradient computation).
//
// This shader converts 2D gradients (d_mean_px, d_cov_2d) from the rasterization
// backward pass into 3D gradients (d_position, d_scale, d_rotation) for the
// Gaussian parameters.
//
// Ported from: src/gpu/gradients.rs::chain_2d_to_3d_gradients()

// ============================================================================
// Data Structures
// ============================================================================

// Camera parameters for projection math
struct Camera {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    pad1: u32,
    pad2: u32,
    rotation: mat3x3<f32>,       // World→camera rotation (3x3 = 36 bytes)
    translation: vec3<f32>,      // Camera translation (= -R * camera_pos)
    pad3: f32,                   // Padding for alignment
}

// Input: 2D gradients from rasterization backward
struct Gradient2D {
    d_color: vec4<f32>,              // dL/d(color) + padding
    d_opacity_logit_pad: vec4<f32>,  // dL/d(opacity_logit) + padding
    d_mean_px: vec4<f32>,            // dL/d(mean_px) + padding
    d_cov_2d: vec4<f32>,             // dL/d(cov_2d) as (xx, xy, yy, pad)
}

// 3D Gaussian input (for cached forward data)
struct Gaussian3D {
    position: vec4<f32>,         // xyz + padding
    log_scale: vec4<f32>,        // log(sx, sy, sz) + padding
    rotation: vec4<f32>,         // Quaternion (w,x,y,z) - NOTE: raw, not normalized
    opacity_logit_pad: vec4<f32>,
    sh: array<vec4<f32>, 4>,     // SH coefficients (16 coeffs as 4 vec4)
}

// Output: 3D gradients
struct Gradient3D {
    d_position: vec4<f32>,       // dL/d(position) + padding
    d_log_scale: vec4<f32>,      // dL/d(log_scale) + padding
    d_rotation: vec4<f32>,       // dL/d(rotation) (quaternion gradient, w,x,y,z)
    d_sh: array<vec4<f32>, 4>,   // dL/d(SH coefficients)
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> gaussians_3d: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read> gradients_2d: array<Gradient2D>;
@group(0) @binding(3) var<storage, read_write> gradients_3d: array<Gradient3D>;

// ============================================================================
// Math Helper Functions
// ============================================================================

// Convert quaternion to rotation matrix
// Quaternion order: (w, x, y, z)
// NOTE: Input q_raw is NOT normalized - we normalize it first
fn quaternion_to_matrix(q_raw: vec4<f32>) -> mat3x3<f32> {
    let n = length(q_raw);
    let q = q_raw / n;

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
    );
}

// Transform world point to camera space
fn world_to_camera(pos_world: vec3<f32>, cam_rot: mat3x3<f32>, cam_trans: vec3<f32>) -> vec3<f32> {
    return cam_rot * pos_world + cam_trans;
}

// Compute perspective projection Jacobian
// J = ∂[u,v]/∂[x,y,z] where [u,v] = [fx*x/z + cx, fy*y/z + cy]
// Returns mat3x2: 3 columns (for x,y,z) × 2 rows (for u,v)
fn perspective_jacobian(point_cam: vec3<f32>, fx: f32, fy: f32) -> mat3x2<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;

    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;

    // J = | fx/z    0      -fx*x/z² |
    //     |  0     fy/z    -fy*y/z² |
    // In WGSL column-major: 3 columns of vec2
    return mat3x2<f32>(
        vec2<f32>(fx * z_inv, 0.0),           // Column 0 (∂u/∂x, ∂v/∂x)
        vec2<f32>(0.0, fy * z_inv),           // Column 1 (∂u/∂y, ∂v/∂y)
        vec2<f32>(-fx * x * z_inv2, -fy * y * z_inv2)  // Column 2 (∂u/∂z, ∂v/∂z)
    );
}

// ============================================================================
// Gradient Functions (Backward Pass Math)
// ============================================================================

// Gradient of projection w.r.t. camera-space point
// Given d_uv = dL/d[u,v], returns dL/d[x,y,z]
fn project_point_grad_point_cam(
    point_cam: vec3<f32>,
    fx: f32,
    fy: f32,
    d_uv: vec2<f32>
) -> vec3<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;

    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;

    // Jacobian elements
    let du_dx = fx * z_inv;
    let du_dz = -fx * x * z_inv2;
    let dv_dy = fy * z_inv;
    let dv_dz = -fy * y * z_inv2;

    // Chain rule
    let d_x = d_uv.x * du_dx;
    let d_y = d_uv.y * dv_dy;
    let d_z = d_uv.x * du_dz + d_uv.y * dv_dz;

    return vec3<f32>(d_x, d_y, d_z);
}

// Gradient of 2D covariance projection w.r.t. camera-space point
// This is complex - involves the Jacobian dependence on the point
fn project_covariance_2d_grad_point_cam(
    point_cam: vec3<f32>,
    fx: f32,
    fy: f32,
    cam_rot: mat3x3<f32>,
    gaussian_rot: mat3x3<f32>,
    log_scale: vec3<f32>,
    d_sigma2d: mat2x2<f32>
) -> vec3<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;

    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;
    let z_inv3 = z_inv2 * z_inv;

    // Construct 3D covariance from scale and rotation
    let v = vec3<f32>(
        exp(2.0 * log_scale.x),
        exp(2.0 * log_scale.y),
        exp(2.0 * log_scale.z)
    );
    let d_mat = mat3x3<f32>(
        vec3<f32>(v.x, 0.0, 0.0),
        vec3<f32>(0.0, v.y, 0.0),
        vec3<f32>(0.0, 0.0, v.z)
    );
    let sigma = gaussian_rot * d_mat * transpose(gaussian_rot);
    let sigma_cam = cam_rot * sigma * transpose(cam_rot);

    // Gradient through Jacobian
    // Σ₂d = J Σ_cam Jᵀ, so d_J = (G + Gᵀ) J Σ_cam where G = d_sigma2d
    let sym_g = d_sigma2d + transpose(d_sigma2d);
    let j = perspective_jacobian(point_cam, fx, fy);
    // Break into steps to avoid potential chained multiplication issues
    let temp = sym_g * j;  // mat2x2 * mat3x2 = mat3x2
    let d_j = temp * sigma_cam;  // mat3x2 * mat3x3 = mat3x2

    // Gradient of Jacobian w.r.t. point_cam
    // J = | fx/z    0      -fx*x/z² |
    //     |  0     fy/z    -fy*y/z² |
    // d_j is mat3x2 (3 columns of vec2), so access as d_j[col][row]

    let d_j00 = d_j[0][0];  // Element (row=0, col=0)
    let d_j02 = d_j[2][0];  // Element (row=0, col=2)
    let d_j11 = d_j[1][1];  // Element (row=1, col=1)
    let d_j12 = d_j[2][1];  // Element (row=1, col=2)

    let d_x = d_j02 * (-fx * z_inv2);
    let d_y = d_j12 * (-fy * z_inv2);
    let d_z = d_j00 * (-fx * z_inv2)
        + d_j02 * (2.0 * fx * x * z_inv3)
        + d_j11 * (-fy * z_inv2)
        + d_j12 * (2.0 * fy * y * z_inv3);

    return vec3<f32>(d_x, d_y, d_z);
}

// Gradient of 2D covariance projection w.r.t. log-scales
// Returns dL/d(log_scale)
fn project_covariance_2d_grad_log_scale(
    cam_rot: mat3x3<f32>,
    jacobian: mat3x2<f32>,  // 3 columns of vec2 (represents 2×3 math matrix)
    gaussian_rot: mat3x3<f32>,
    log_scale: vec3<f32>,
    d_sigma2d: mat2x2<f32>
) -> vec3<f32> {
    // Backprop through covariance projection
    // Σ₂d = J Σ_cam Jᵀ
    // dL/dΣ_cam = Jᵀ dL/dΣ₂d J
    // Force right-to-left evaluation to avoid WGSL matrix mult issues
    let g_j = d_sigma2d * jacobian;  // (2×2) * (2×3) = (2×3) as mat3x2
    let d_sigma_cam = transpose(jacobian) * g_j;  // (3×2) * (2×3) = (3×3) as mat3x3

    // Σ_cam = W Σ Wᵀ
    // dL/dΣ = Wᵀ dL/dΣ_cam W
    let d_sigma_cam_w = d_sigma_cam * cam_rot;  // (3×3) * (3×3) = (3×3)
    let d_sigma = transpose(cam_rot) * d_sigma_cam_w;  // (3×3) * (3×3) = (3×3)

    // Σ = R D Rᵀ where D = diag(v) and v_i = exp(2 * log_scale_i)
    // M = Rᵀ dL/dΣ R
    let d_sigma_r = d_sigma * gaussian_rot;  // (3×3) * (3×3) = (3×3)
    let m = transpose(gaussian_rot) * d_sigma_r;  // (3×3) * (3×3) = (3×3)

    // dL/dv_i = M_ii
    // dL/d(log_scale_i) = dL/dv_i * dv_i/d(log_scale_i) = M_ii * 2 * exp(2 * log_scale_i)
    let v = vec3<f32>(
        exp(2.0 * log_scale.x),
        exp(2.0 * log_scale.y),
        exp(2.0 * log_scale.z)
    );

    return vec3<f32>(
        m[0][0] * 2.0 * v.x,
        m[1][1] * 2.0 * v.y,
        m[2][2] * 2.0 * v.z
    );
}

// Gradient of 2D covariance projection w.r.t. rotation (as SO(3) vector at identity)
// Returns dL/d(rotation_vector) evaluated at current rotation
fn project_covariance_2d_grad_rotation_vector(
    cam_rot: mat3x3<f32>,
    jacobian: mat3x2<f32>,  // 3 columns of vec2 (represents 2×3 math matrix)
    gaussian_rot: mat3x3<f32>,
    log_scale: vec3<f32>,
    d_sigma2d: mat2x2<f32>
) -> vec3<f32> {
    let v = vec3<f32>(
        exp(2.0 * log_scale.x),
        exp(2.0 * log_scale.y),
        exp(2.0 * log_scale.z)
    );
    let d_mat = mat3x3<f32>(
        vec3<f32>(v.x, 0.0, 0.0),
        vec3<f32>(0.0, v.y, 0.0),
        vec3<f32>(0.0, 0.0, v.z)
    );

    // Backprop to dL/dΣ
    let g_j = d_sigma2d * jacobian;  // (2×2) * (2×3) = (2×3)
    let d_sigma_cam = transpose(jacobian) * g_j;  // (3×2) * (2×3) = (3×3)
    let d_sigma_cam_w = d_sigma_cam * cam_rot;  // (3×3) * (3×3) = (3×3)
    let d_sigma = transpose(cam_rot) * d_sigma_cam_w;  // (3×3) * (3×3) = (3×3)

    // Gradient w.r.t. R: dL/dR = (G + Gᵀ) R D
    let g = d_sigma;
    let g_sym = g + transpose(g);  // Symmetrize first
    let g_r_temp = g_sym * gaussian_rot;  // (3×3) * (3×3) = (3×3)
    let g_r = g_r_temp * d_mat;  // (3×3) * (3×3) = (3×3)

    // Skew-symmetric basis matrices for SO(3)
    let kx = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0),
        vec3<f32>(0.0, 1.0, 0.0)
    );
    let ky = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(-1.0, 0.0, 0.0)
    );
    let kz = mat3x3<f32>(
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0)
    );

    // Directional derivatives
    let d_r_x = kx * gaussian_rot;
    let d_r_y = ky * gaussian_rot;
    let d_r_z = kz * gaussian_rot;

    // Inner product (Frobenius)
    let grad_x = dot(g_r[0], d_r_x[0]) + dot(g_r[1], d_r_x[1]) + dot(g_r[2], d_r_x[2]);
    let grad_y = dot(g_r[0], d_r_y[0]) + dot(g_r[1], d_r_y[1]) + dot(g_r[2], d_r_y[2]);
    let grad_z = dot(g_r[0], d_r_z[0]) + dot(g_r[1], d_r_z[1]) + dot(g_r[2], d_r_z[2]);

    return vec3<f32>(grad_x, grad_y, grad_z);
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute @workgroup_size(256)
fn project_backward(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gaussian_idx = global_id.x;
    if (gaussian_idx >= arrayLength(&gaussians_3d)) {
        return;
    }

    let g = gaussians_3d[gaussian_idx];
    let g2d = gradients_2d[gaussian_idx];

    // Initialize 3D gradients
    var d_position = vec3<f32>(0.0);
    var d_log_scale = vec3<f32>(0.0);
    var d_rot_vec = vec3<f32>(0.0);

    // Compute camera-space position (needed for both gradients)
    let point_cam = world_to_camera(g.position.xyz, camera.rotation, camera.translation);

    // 1. Chain d_mean_px -> d_point_cam -> d_position
    let d_mean_px = g2d.d_mean_px.xy;
    if (dot(d_mean_px, d_mean_px) > 0.0) {
        let d_point_cam_from_mean = project_point_grad_point_cam(
            point_cam,
            camera.fx,
            camera.fy,
            d_mean_px
        );

        // Transform to world space: dL/d(pos_world) = R^T * dL/d(point_cam)
        d_position += transpose(camera.rotation) * d_point_cam_from_mean;
    }

    // 2. Chain d_cov_2d -> d_point_cam, d_log_scale, d_rot_vec
    let d_cov_2d = g2d.d_cov_2d.xyz;  // (xx, xy, yy)
    if (dot(d_cov_2d, d_cov_2d) > 0.0) {
        let gaussian_rot = quaternion_to_matrix(g.rotation);
        let d_sigma2d = mat2x2<f32>(
            vec2<f32>(d_cov_2d.x, d_cov_2d.y),
            vec2<f32>(d_cov_2d.y, d_cov_2d.z)
        );

        // d_cov_2d -> d_point_cam (via Jacobian dependence)
        let d_point_cam_from_cov = project_covariance_2d_grad_point_cam(
            point_cam,
            camera.fx,
            camera.fy,
            camera.rotation,
            gaussian_rot,
            g.log_scale.xyz,
            d_sigma2d
        );
        d_position += transpose(camera.rotation) * d_point_cam_from_cov;

        // d_cov_2d -> d_log_scale
        let j = perspective_jacobian(point_cam, camera.fx, camera.fy);
        d_log_scale = project_covariance_2d_grad_log_scale(
            camera.rotation,
            j,
            gaussian_rot,
            g.log_scale.xyz,
            d_sigma2d
        );

        // d_cov_2d -> d_rot_vec
        d_rot_vec = project_covariance_2d_grad_rotation_vector(
            camera.rotation,
            j,
            gaussian_rot,
            g.log_scale.xyz,
            d_sigma2d
        );
    }

    // 3. Store results (SH gradient is just a copy from d_color for now)
    gradients_3d[gaussian_idx] = Gradient3D(
        vec4<f32>(d_position, 0.0),
        vec4<f32>(d_log_scale, 0.0),
        vec4<f32>(d_rot_vec, 0.0),  // NOTE: This is SO(3) vector, not quaternion yet!
        array<vec4<f32>, 4>(
            g2d.d_color,  // SH gradients = color gradients (for now, SH not implemented)
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0)
        )
    );
}
