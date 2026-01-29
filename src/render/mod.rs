pub mod cpu;
pub mod gpu;

use image::{ImageBuffer, Rgb};
use std::time::Duration;
use crate::types::{Scene, Camera};

/// Result of rendering
pub struct RenderResult {
    pub image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub samples: u32,
    pub backend: &'static str,
}

/// Trait for render backends
pub trait Renderer {
    fn render(&self, scene: &Scene, camera: &Camera, time_limit: Duration) -> RenderResult;
    fn render_samples(&self, scene: &Scene, camera: &Camera, num_samples: u32) -> RenderResult;
    fn name(&self) -> &'static str;
}
