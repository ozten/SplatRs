use crate::core::GaussianCloud;
use crate::gpu::GpuRenderer;
use crate::io::ModelMetadata;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct AppState {
    pub model: Arc<Mutex<Option<(GaussianCloud, ModelMetadata)>>>,
    pub renderer: Arc<Mutex<GpuRenderer>>,
    /// When true, only DC term is used for color (view-independent rendering)
    pub disable_sh: AtomicBool,
}

impl AppState {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let renderer = GpuRenderer::new().map_err(|e| e.to_string())?;
        Ok(Self {
            model: Arc::new(Mutex::new(None)),
            renderer: Arc::new(Mutex::new(renderer)),
            disable_sh: AtomicBool::new(false),
        })
    }

    pub fn get_disable_sh(&self) -> bool {
        self.disable_sh.load(Ordering::Relaxed)
    }

    pub fn set_disable_sh(&self, value: bool) {
        self.disable_sh.store(value, Ordering::Relaxed);
    }
}
