//! GPU-availability skip helper shared by the golden_* GPU-gated tests
//! (docs/TILE_RASTER_PLAN.md Part A).
//!
//! Convention (matches tests/gpu_deterministic_rendering.rs): `GpuRenderer::new()`
//! returning `Err` means no adapter/device is available (e.g. CI without a GPU, or a
//! probe-limits failure per the 2026-07-23 watchdog false-positive fix) — tests should
//! log and skip rather than fail the whole suite.
//!
//! Only compiled when `mod`-included with `#[cfg(feature = "gpu")]` at the call site
//! (files under a tests/ subdirectory aren't auto-discovered as their own test target,
//! so this cfg lives on the `mod` declaration, not duplicated here).

#![allow(dead_code)]

use sugar_rs::gpu::GpuRenderer;

/// Try to create a `GpuRenderer`. Returns `None` (after an `eprintln!`) if no
/// adapter/device is available, so callers can bail out early with:
/// `let Some(gpu) = gpu_skip::try_gpu_renderer() else { return; };`
pub fn try_gpu_renderer() -> Option<GpuRenderer> {
    match GpuRenderer::new() {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("Skipping GPU-gated golden test (no adapter/device): {e}");
            None
        }
    }
}
