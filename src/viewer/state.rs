use crate::core::GaussianCloud;
use crate::gpu::GpuRenderer;
use crate::io::ModelMetadata;
use std::sync::{Arc, Mutex};

pub struct AppState {
    pub model: Arc<Mutex<Option<(GaussianCloud, ModelMetadata)>>>,
    pub renderer: Arc<Mutex<GpuRenderer>>,
}

impl AppState {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let renderer = GpuRenderer::new().map_err(|e| e.to_string())?;
        Ok(Self {
            model: Arc::new(Mutex::new(None)),
            renderer: Arc::new(Mutex::new(renderer)),
        })
    }
}
