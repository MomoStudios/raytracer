mod types;
mod render;

use std::fs;
use std::env;
use std::time::{Duration, Instant};
use types::{Scene, RenderBackend};
use render::{Renderer, cpu::CpuRenderer, gpu::GpuRenderer};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: raytracer <scene.json> <output.png> [time_ms] [--cpu|--gpu]");
        eprintln!("  --cpu    Force CPU rendering");
        eprintln!("  --gpu    Force GPU rendering (default if available)");
        std::process::exit(1);
    }

    let scene_file = &args[1];
    let output_file = &args[2];
    
    // Parse optional arguments
    let mut time_ms: u64 = 0;
    let mut backend = None;
    
    for arg in args.iter().skip(3) {
        if arg == "--cpu" {
            backend = Some(RenderBackend::Cpu);
        } else if arg == "--gpu" {
            backend = Some(RenderBackend::Gpu);
        } else if let Ok(t) = arg.parse::<u64>() {
            time_ms = t;
        }
    }

    let scene_json = fs::read_to_string(scene_file)
        .expect("Failed to read scene file");
    let scene: Scene = serde_json::from_str(&scene_json)
        .expect("Failed to parse scene JSON");

    let render_time = if time_ms > 0 { time_ms } else { scene.render_time_ms };
    let time_limit = Duration::from_millis(render_time);

    // Select renderer
    let renderer: Box<dyn Renderer> = match backend {
        Some(RenderBackend::Cpu) => {
            println!("Using CPU renderer (forced)");
            Box::new(CpuRenderer::new())
        }
        Some(RenderBackend::Gpu) => {
            match GpuRenderer::new() {
                Some(r) => {
                    println!("Using GPU renderer (forced)");
                    Box::new(r)
                }
                None => {
                    eprintln!("GPU renderer requested but not available, falling back to CPU");
                    Box::new(CpuRenderer::new())
                }
            }
        }
        None => {
            // Default: try GPU, fall back to CPU
            match GpuRenderer::new() {
                Some(r) => {
                    println!("Using GPU renderer (auto-detected)");
                    Box::new(r)
                }
                None => {
                    println!("GPU not available, using CPU renderer");
                    Box::new(CpuRenderer::new())
                }
            }
        }
    };

    let geom_count = scene.spheres.len() + scene.triangles.len() + scene.polygons.len();
    println!("Rendering {}x{} with {} objects, {} lights, {} max bounces, {}ms time limit...",
             scene.width, scene.height, geom_count, scene.lights.len(), scene.max_bounces, render_time);

    let start = Instant::now();
    let result = renderer.render(&scene, time_limit);
    let elapsed = start.elapsed();

    result.image.save(output_file).expect("Failed to save image");
    println!("Rendered {} samples in {:.2}s using {} -> {}", 
             result.samples, elapsed.as_secs_f64(), result.backend, output_file);
}
