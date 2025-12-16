//! Adam optimizer (minimal).
//!
//! For M7 we start with a small, focused optimizer that can update per-Gaussian
//! RGB vectors (e.g. SH DC coefficients) on CPU.

use nalgebra::Vector3;

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
            self.m = vec![Vector3::zeros(); len];
            self.v = vec![Vector3::zeros(); len];
            self.t = 0;
        }
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
