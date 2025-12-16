//! Adam optimizer (minimal).
//!
//! For M7 we start with a small, focused optimizer that can update per-Gaussian
//! RGB vectors (e.g. SH DC coefficients) on CPU.

use nalgebra::{UnitQuaternion, Vector3};

pub struct AdamF32 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamF32 {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn ensure_len(&mut self, len: usize) {
        if self.m.len() != len {
            self.m.resize(len, 0.0);
            self.v.resize(len, 0.0);
        }
    }

    /// Reset moment estimates while keeping the global timestep.
    ///
    /// This is useful when the parameter vector is re-built (e.g. densification/pruning),
    /// because any per-index mapping to momentum state becomes invalid.
    pub fn reset_moments_keep_t(&mut self, len: usize) {
        self.m.clear();
        self.v.clear();
        self.m.resize(len, 0.0);
        self.v.resize(len, 0.0);
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        self.ensure_len(params.len());

        self.t += 1;
        let t = self.t as f32;
        let b1 = self.beta1;
        let b2 = self.beta2;

        let bias1 = 1.0 - b1.powf(t);
        let bias2 = 1.0 - b2.powf(t);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.m[i] * b1 + g * (1.0 - b1);
            self.v[i] = self.v[i] * b2 + g * g * (1.0 - b2);

            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

pub struct AdamVec3 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u32,
    m: Vec<Vector3<f32>>,
    v: Vec<Vector3<f32>>,
}

impl AdamVec3 {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn ensure_len(&mut self, len: usize) {
        if self.m.len() != len {
            // Resize to new length, preserving existing state and zeroing new elements
            self.m.resize(len, Vector3::zeros());
            self.v.resize(len, Vector3::zeros());
            // Don't reset t! Keep the current timestep for proper bias correction.
            // New parameters start with zero momentum, which is correct.
        }
    }

    /// Reset moment estimates while keeping the global timestep.
    ///
    /// This is useful when the parameter vector is re-built (e.g. densification/pruning),
    /// because any per-index mapping to momentum state becomes invalid.
    pub fn reset_moments_keep_t(&mut self, len: usize) {
        self.m.clear();
        self.v.clear();
        self.m.resize(len, Vector3::zeros());
        self.v.resize(len, Vector3::zeros());
    }

    pub fn step(&mut self, params: &mut [Vector3<f32>], grads: &[Vector3<f32>]) {
        assert_eq!(params.len(), grads.len());
        self.ensure_len(params.len());

        self.t += 1;
        let t = self.t as f32;
        let b1 = self.beta1;
        let b2 = self.beta2;

        let bias1 = 1.0 - b1.powf(t);
        let bias2 = 1.0 - b2.powf(t);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.m[i] * b1 + g * (1.0 - b1);
            self.v[i] = self.v[i] * b2 + g.component_mul(&g) * (1.0 - b2);

            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;

            // elementwise update
            params[i].x -= self.lr * m_hat.x / (v_hat.x.sqrt() + self.eps);
            params[i].y -= self.lr * m_hat.y / (v_hat.y.sqrt() + self.eps);
            params[i].z -= self.lr * m_hat.z / (v_hat.z.sqrt() + self.eps);
        }
    }
}

pub struct AdamSo3 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u32,
    m: Vec<Vector3<f32>>,
    v: Vec<Vector3<f32>>,
}

impl AdamSo3 {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn ensure_len(&mut self, len: usize) {
        if self.m.len() != len {
            self.m.resize(len, Vector3::zeros());
            self.v.resize(len, Vector3::zeros());
        }
    }

    pub fn reset_moments_keep_t(&mut self, len: usize) {
        self.m.clear();
        self.v.clear();
        self.m.resize(len, Vector3::zeros());
        self.v.resize(len, Vector3::zeros());
    }

    pub fn step(&mut self, rotations: &mut [UnitQuaternion<f32>], grads: &[Vector3<f32>]) {
        assert_eq!(rotations.len(), grads.len());
        self.ensure_len(rotations.len());

        self.t += 1;
        let t = self.t as f32;
        let b1 = self.beta1;
        let b2 = self.beta2;

        let bias1 = 1.0 - b1.powf(t);
        let bias2 = 1.0 - b2.powf(t);

        for i in 0..rotations.len() {
            let g = grads[i];
            self.m[i] = self.m[i] * b1 + g * (1.0 - b1);
            self.v[i] = self.v[i] * b2 + g.component_mul(&g) * (1.0 - b2);

            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;

            let mut step_vec = Vector3::new(
                -self.lr * m_hat.x / (v_hat.x.sqrt() + self.eps),
                -self.lr * m_hat.y / (v_hat.y.sqrt() + self.eps),
                -self.lr * m_hat.z / (v_hat.z.sqrt() + self.eps),
            );

            // Keep updates small; otherwise a single bad gradient can flip rotations.
            let max_step_rad = 0.2f32;
            let n = step_vec.norm();
            if n.is_finite() && n > max_step_rad {
                step_vec *= max_step_rad / n;
            }

            let dq = UnitQuaternion::from_scaled_axis(step_vec);
            rotations[i] = dq * rotations[i];
        }
    }
}

pub struct AdamSh16 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u32,
    m: Vec<[Vector3<f32>; 16]>,
    v: Vec<[Vector3<f32>; 16]>,
}

impl AdamSh16 {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn ensure_len(&mut self, len: usize) {
        if self.m.len() != len {
            self.m.resize(len, [Vector3::zeros(); 16]);
            self.v.resize(len, [Vector3::zeros(); 16]);
        }
    }

    pub fn reset_moments_keep_t(&mut self, len: usize) {
        self.m.clear();
        self.v.clear();
        self.m.resize(len, [Vector3::zeros(); 16]);
        self.v.resize(len, [Vector3::zeros(); 16]);
    }

    pub fn step(&mut self, params: &mut [[Vector3<f32>; 16]], grads: &[[Vector3<f32>; 16]]) {
        assert_eq!(params.len(), grads.len());
        self.ensure_len(params.len());

        self.t += 1;
        let t = self.t as f32;
        let b1 = self.beta1;
        let b2 = self.beta2;

        let bias1 = 1.0 - b1.powf(t);
        let bias2 = 1.0 - b2.powf(t);

        for i in 0..params.len() {
            for k in 0..16 {
                let g = grads[i][k];
                self.m[i][k] = self.m[i][k] * b1 + g * (1.0 - b1);
                self.v[i][k] = self.v[i][k] * b2 + g.component_mul(&g) * (1.0 - b2);

                let m_hat = self.m[i][k] / bias1;
                let v_hat = self.v[i][k] / bias2;

                params[i][k].x -= self.lr * m_hat.x / (v_hat.x.sqrt() + self.eps);
                params[i][k].y -= self.lr * m_hat.y / (v_hat.y.sqrt() + self.eps);
                params[i][k].z -= self.lr * m_hat.z / (v_hat.z.sqrt() + self.eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_preserves_timestep_on_resize() {
        let mut opt = AdamVec3::new(0.001, 0.9, 0.999, 1e-8);

        // Start with 2 parameters
        let mut params = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)];
        let grads = vec![Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.4, 0.5, 0.6)];

        // Run a few steps
        opt.step(&mut params, &grads);
        opt.step(&mut params, &grads);
        opt.step(&mut params, &grads);

        assert_eq!(opt.t, 3, "Should have timestep 3 after 3 steps");

        // Now resize to 3 parameters (simulating adding a Gaussian)
        let mut params3 = vec![
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(7.0, 8.0, 9.0),
        ];
        let grads3 = vec![
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
        ];

        // Take another step with new size
        opt.step(&mut params3, &grads3);

        // Timestep should NOT be reset!
        assert_eq!(
            opt.t, 4,
            "Timestep should be 4 (not reset to 1) after resize"
        );

        // Momentum vectors should be the right size
        assert_eq!(opt.m.len(), 3, "Momentum vectors should have length 3");
        assert_eq!(opt.v.len(), 3, "Velocity vectors should have length 3");

        // First two elements should have non-zero momentum (from previous steps)
        assert_ne!(
            opt.m[0],
            Vector3::zeros(),
            "First parameter should have momentum"
        );
        assert_ne!(
            opt.m[1],
            Vector3::zeros(),
            "Second parameter should have momentum"
        );

        // Third element should have accumulated some momentum from the last step
        assert_ne!(
            opt.m[2],
            Vector3::zeros(),
            "Third parameter should have momentum after one step"
        );
    }

    #[test]
    fn test_adam_basic_update() {
        let mut opt = AdamVec3::new(0.01, 0.9, 0.999, 1e-8);

        let mut params = vec![Vector3::new(1.0, 1.0, 1.0)];
        let grads = vec![Vector3::new(1.0, 1.0, 1.0)];

        let initial = params[0];
        opt.step(&mut params, &grads);

        // Parameters should have moved in the opposite direction of gradient
        assert!(
            params[0].x < initial.x,
            "Parameter should decrease with positive gradient"
        );
        assert!(
            params[0].y < initial.y,
            "Parameter should decrease with positive gradient"
        );
        assert!(
            params[0].z < initial.z,
            "Parameter should decrease with positive gradient"
        );
    }

    #[test]
    fn test_adam_f32_basic_update() {
        let mut opt = AdamF32::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0f32];
        let grads = vec![1.0f32];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }
}
