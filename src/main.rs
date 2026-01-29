mod types;
mod render;

use std::fs;
use std::env;
use std::time::{Duration, Instant};
use types::{Scene, Camera, Vec3, RenderBackend};
use render::{Renderer, cpu::CpuRenderer, gpu::GpuRenderer};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: raytracer <scene.json> <output.png> [time_ms] [--cpu|--gpu]");
        eprintln!("       raytracer <scene.json> <output_dir> --animate <seconds> <fps> [samples_per_frame]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --cpu         Force CPU rendering");
        eprintln!("  --gpu         Force GPU rendering (default if available)");
        eprintln!("  --animate     Render animation with camera orbiting scene center");
        std::process::exit(1);
    }

    let scene_file = &args[1];
    let output = &args[2];
    
    let scene_json = fs::read_to_string(scene_file)
        .expect("Failed to read scene file");
    let scene: Scene = serde_json::from_str(&scene_json)
        .expect("Failed to parse scene JSON");

    // Check for animation mode
    if args.iter().any(|a| a == "--animate") {
        run_animation(&args, &scene, output);
    } else {
        run_single_frame(&args, &scene, output);
    }
}

fn run_single_frame(args: &[String], scene: &Scene, output_file: &str) {
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

    let render_time = if time_ms > 0 { time_ms } else { scene.render_time_ms };
    let time_limit = Duration::from_millis(render_time);

    let renderer: Box<dyn Renderer> = select_renderer(backend);
    let camera = scene.get_camera();

    let geom_count = scene.spheres.len() + scene.triangles.len() + scene.polygons.len();
    println!("Rendering {}x{} with {} objects, {} lights, {} max bounces, {}ms time limit...",
             scene.width, scene.height, geom_count, scene.lights.len(), scene.max_bounces, render_time);

    let start = Instant::now();
    let result = renderer.render(scene, &camera, time_limit);
    let elapsed = start.elapsed();

    result.image.save(output_file).expect("Failed to save image");
    println!("Rendered {} samples in {:.2}s using {} -> {}", 
             result.samples, elapsed.as_secs_f64(), result.backend, output_file);
}

fn run_animation(args: &[String], scene: &Scene, output_dir: &str) {
    // Parse animation args: --animate <seconds> <fps> [samples_per_frame]
    let animate_idx = args.iter().position(|a| a == "--animate").unwrap();
    
    let seconds: f64 = args.get(animate_idx + 1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let fps: u32 = args.get(animate_idx + 2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let samples_per_frame: u32 = args.get(animate_idx + 3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    
    let total_frames = (seconds * fps as f64) as u32;
    
    // Determine backend
    let backend = if args.iter().any(|a| a == "--cpu") {
        Some(RenderBackend::Cpu)
    } else if args.iter().any(|a| a == "--gpu") {
        Some(RenderBackend::Gpu)
    } else {
        None
    };
    
    let renderer: Box<dyn Renderer> = select_renderer(backend);
    
    // Create output directory
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    // Calculate orbit parameters
    let base_camera = scene.get_camera();
    let center = base_camera.look_at.clone();
    let initial_offset = base_camera.position.sub(&center);
    let orbit_radius = initial_offset.length();
    let orbit_height = initial_offset.y;
    
    // Calculate initial angle
    let initial_angle = initial_offset.z.atan2(initial_offset.x);
    
    println!("Rendering {} frames at {}fps ({:.1}s) with {} samples per frame",
             total_frames, fps, seconds, samples_per_frame);
    println!("Output directory: {}", output_dir);
    println!("Using {} renderer", renderer.name());
    println!();
    
    let total_start = Instant::now();
    
    for frame in 0..total_frames {
        let t = frame as f64 / total_frames as f64;
        let angle = initial_angle + t * 2.0 * std::f64::consts::PI; // Full rotation
        
        // Calculate camera position on orbit
        let cam_x = center.x + orbit_radius * angle.cos();
        let cam_z = center.z + orbit_radius * angle.sin();
        
        let camera = Camera {
            position: Vec3::new(cam_x, center.y + orbit_height, cam_z),
            look_at: center.clone(),
            up: base_camera.up.clone(),
            fov: base_camera.fov,
        };
        
        let frame_start = Instant::now();
        let result = renderer.render_samples(scene, &camera, samples_per_frame);
        let frame_time = frame_start.elapsed();
        
        let filename = format!("{}/frame_{:05}.png", output_dir, frame);
        result.image.save(&filename).expect("Failed to save frame");
        
        let elapsed = total_start.elapsed().as_secs_f64();
        let eta = if frame > 0 {
            elapsed / frame as f64 * (total_frames - frame) as f64
        } else {
            0.0
        };
        
        println!("Frame {}/{} - {:.2}s - ETA: {:.0}s", 
                 frame + 1, total_frames, frame_time.as_secs_f64(), eta);
    }
    
    let total_time = total_start.elapsed();
    println!();
    println!("Animation complete! {} frames in {:.1}s", total_frames, total_time.as_secs_f64());
    println!();
    println!("To create video, run:");
    println!("  ffmpeg -framerate {} -i {}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4", fps, output_dir);
}

fn select_renderer(backend: Option<RenderBackend>) -> Box<dyn Renderer> {
    match backend {
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
    }
}
