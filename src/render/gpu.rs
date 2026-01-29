use image::{ImageBuffer, Rgb};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use crate::types::*;
use super::{Renderer, RenderResult};

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuRenderer {
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }
    
    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await.ok()?;
        
        println!("GPU adapter: {}", adapter.get_info().name);
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Raytracer GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ).await.ok()?;
        
        Some(GpuRenderer { device, queue })
    }
    
    fn render_internal(&self, scene: &Scene, camera: &Camera, samples_limit: Option<u32>, time_limit: Option<Duration>) -> RenderResult {
        let width = scene.width;
        let height = scene.height;
        let total_pixels = (width * height) as usize;
        
        // Compute camera basis vectors
        let forward = camera.look_at.sub(&camera.position).normalize();
        let right = forward.cross(&camera.up).normalize();
        let up = right.cross(&forward).normalize();
        
        // Convert scene to GPU format
        let spheres: Vec<GpuSphere> = scene.spheres.iter().map(|s| GpuSphere::from_sphere(s)).collect();
        let mut triangles: Vec<GpuTriangle> = scene.triangles.iter().map(|t| GpuTriangle::from_triangle(t)).collect();
        
        for poly in &scene.polygons {
            if poly.vertices.len() >= 3 {
                for i in 1..poly.vertices.len() - 1 {
                    triangles.push(GpuTriangle::from_polygon_tri(poly, 0, i, i + 1));
                }
            }
        }
        
        let lights: Vec<GpuLight> = scene.lights.iter().map(|l| GpuLight::from_light(l)).collect();
        
        // Ensure we have at least one element for buffers
        let spheres = if spheres.is_empty() { vec![GpuSphere::dummy()] } else { spheres };
        let triangles = if triangles.is_empty() { vec![GpuTriangle::dummy()] } else { triangles };
        
        let actual_num_spheres = if scene.spheres.is_empty() { 0 } else { spheres.len() as u32 };
        let actual_num_triangles = if scene.triangles.is_empty() && scene.polygons.is_empty() { 0 } else { triangles.len() as u32 };
        
        // Create shader
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raytracer Shader"),
            source: wgpu::ShaderSource::Wgsl(RAYTRACER_SHADER.into()),
        });
        
        // Create buffers
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"),
            contents: &[0u8; std::mem::size_of::<GpuParams>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let spheres_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spheres"),
            contents: bytemuck::cast_slice(&spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let triangles_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Triangles"),
            contents: bytemuck::cast_slice(&triangles),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let lights_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lights"),
            contents: bytemuck::cast_slice(&lights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let accum_size = (total_pixels * 4 * std::mem::size_of::<f32>()) as u64;
        let accum_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accumulator"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let output_size = (total_pixels * 4) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout and pipeline
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Raytracer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raytracer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raytracer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raytracer Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spheres_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: triangles_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: lights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
            ],
        });
        
        let start = Instant::now();
        let mut samples = 0u32;
        
        let workgroup_size = 16u32;
        let workgroups_x = (width + workgroup_size - 1) / workgroup_size;
        let workgroups_y = (height + workgroup_size - 1) / workgroup_size;
        
        let fov_rad = camera.fov.to_radians();
        
        // Render loop
        loop {
            let params = GpuParams {
                width,
                height,
                num_spheres: actual_num_spheres,
                num_triangles: actual_num_triangles,
                num_lights: lights.len() as u32,
                max_bounces: scene.max_bounces,
                sample: samples,
                _pad: 0,
                background: GpuVec3::from_color(&scene.background),
                camera_pos: GpuVec3::from_vec3(&camera.position),
                camera_forward: GpuVec3::from_vec3(&forward),
                camera_right: GpuVec3::from_vec3(&right),
                camera_up: GpuVec3::from_vec3(&up),
                fov: fov_rad as f32,
                _pad2: [0.0; 3],
            };
            
            self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
            
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Raytracer Encoder") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Raytracer Pass"), timestamp_writes: None });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            
            samples += 1;
            
            // Check termination conditions
            if let Some(limit) = samples_limit {
                if samples >= limit {
                    break;
                }
            }
            if let Some(limit) = time_limit {
                if start.elapsed() >= limit {
                    break;
                }
            }
        }
        
        // Read back results
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy Encoder") });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = buffer_slice.get_mapped_range();
        let pixels: &[u8] = &data;
        
        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                img.put_pixel(x, y, Rgb([pixels[idx], pixels[idx + 1], pixels[idx + 2]]));
            }
        }
        
        drop(data);
        staging_buffer.unmap();
        
        RenderResult {
            image: img,
            samples,
            backend: self.name(),
        }
    }
}

impl Renderer for GpuRenderer {
    fn name(&self) -> &'static str {
        "GPU"
    }
    
    fn render(&self, scene: &Scene, camera: &Camera, time_limit: Duration) -> RenderResult {
        self.render_internal(scene, camera, None, Some(time_limit))
    }
    
    fn render_samples(&self, scene: &Scene, camera: &Camera, num_samples: u32) -> RenderResult {
        self.render_internal(scene, camera, Some(num_samples), None)
    }
}

// GPU data structures (must match shader exactly)

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVec3 {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl GpuVec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        GpuVec3 { x, y, z, _pad: 0.0 }
    }
    
    fn from_vec3(v: &Vec3) -> Self {
        GpuVec3::new(v.x as f32, v.y as f32, v.z as f32)
    }
    
    fn from_color(c: &Color) -> Self {
        GpuVec3::new(c.r as f32, c.g as f32, c.b as f32)
    }
    
    fn zero() -> Self {
        GpuVec3::new(0.0, 0.0, 0.0)
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSphere {
    center: GpuVec3,       // 16 bytes, offset 0
    radius: f32,           // 4 bytes, offset 16
    _pad1a: f32,           // 4 bytes, offset 20
    _pad1b: f32,           // 4 bytes, offset 24
    _pad1c: f32,           // 4 bytes, offset 28
    color: GpuVec3,        // 16 bytes, offset 32
    reflectivity: f32,     // 4 bytes, offset 48
    _pad2a: f32,           // 4 bytes, offset 52
    _pad2b: f32,           // 4 bytes, offset 56
    _pad2c: f32,           // 4 bytes, offset 60
    _pad3: GpuVec3,        // 16 bytes, offset 64
    _pad4: GpuVec3,        // 16 bytes, offset 80 = 96 bytes total
}

impl GpuSphere {
    fn from_sphere(s: &Sphere) -> Self {
        GpuSphere {
            center: GpuVec3::from_vec3(&s.center),
            radius: s.radius as f32,
            _pad1a: 0.0, _pad1b: 0.0, _pad1c: 0.0,
            color: GpuVec3::from_color(&s.color),
            reflectivity: s.reflectivity as f32,
            _pad2a: 0.0, _pad2b: 0.0, _pad2c: 0.0,
            _pad3: GpuVec3::zero(),
            _pad4: GpuVec3::zero(),
        }
    }
    
    fn dummy() -> Self {
        GpuSphere {
            center: GpuVec3::new(0.0, 0.0, -1000.0),
            radius: 0.0,
            _pad1a: 0.0, _pad1b: 0.0, _pad1c: 0.0,
            color: GpuVec3::zero(),
            reflectivity: 0.0,
            _pad2a: 0.0, _pad2b: 0.0, _pad2c: 0.0,
            _pad3: GpuVec3::zero(),
            _pad4: GpuVec3::zero(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    v0: GpuVec3,           // 16 bytes, offset 0
    v1: GpuVec3,           // 16 bytes, offset 16
    v2: GpuVec3,           // 16 bytes, offset 32
    color: GpuVec3,        // 16 bytes, offset 48
    reflectivity: f32,     // 4 bytes, offset 64
    _pad1: f32,            // 4 bytes, offset 68
    _pad2: f32,            // 4 bytes, offset 72
    _pad3: f32,            // 4 bytes, offset 76
    _pad4: GpuVec3,        // 16 bytes, offset 80 = 96 bytes total
}

impl GpuTriangle {
    fn from_triangle(t: &Triangle) -> Self {
        GpuTriangle {
            v0: GpuVec3::from_vec3(&t.v0),
            v1: GpuVec3::from_vec3(&t.v1),
            v2: GpuVec3::from_vec3(&t.v2),
            color: GpuVec3::from_color(&t.color),
            reflectivity: t.reflectivity as f32,
            _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
            _pad4: GpuVec3::zero(),
        }
    }
    
    fn from_polygon_tri(poly: &Polygon, i0: usize, i1: usize, i2: usize) -> Self {
        GpuTriangle {
            v0: GpuVec3::from_vec3(&poly.vertices[i0]),
            v1: GpuVec3::from_vec3(&poly.vertices[i1]),
            v2: GpuVec3::from_vec3(&poly.vertices[i2]),
            color: GpuVec3::from_color(&poly.color),
            reflectivity: poly.reflectivity as f32,
            _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
            _pad4: GpuVec3::zero(),
        }
    }
    
    fn dummy() -> Self {
        GpuTriangle {
            v0: GpuVec3::new(0.0, 0.0, -1000.0),
            v1: GpuVec3::new(0.0, 0.0, -1000.0),
            v2: GpuVec3::new(0.0, 0.0, -1000.0),
            color: GpuVec3::zero(),
            reflectivity: 0.0,
            _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
            _pad4: GpuVec3::zero(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position: GpuVec3,     // 16 bytes
    intensity: f32,        // 4 bytes
    _pad: [f32; 3],        // 12 bytes = 32 bytes total
}

impl GpuLight {
    fn from_light(l: &Light) -> Self {
        GpuLight {
            position: GpuVec3::from_vec3(&l.position),
            intensity: l.intensity as f32,
            _pad: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    num_spheres: u32,
    num_triangles: u32,
    num_lights: u32,
    max_bounces: u32,
    sample: u32,
    _pad: u32,
    background: GpuVec3,
    camera_pos: GpuVec3,
    camera_forward: GpuVec3,
    camera_right: GpuVec3,
    camera_up: GpuVec3,
    fov: f32,
    _pad2: [f32; 3],
}

const RAYTRACER_SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    num_spheres: u32,
    num_triangles: u32,
    num_lights: u32,
    max_bounces: u32,
    sample: u32,
    _pad: u32,
    background: vec4<f32>,
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    fov: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
}

struct Sphere {
    center: vec4<f32>,
    radius: f32,
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
    color: vec4<f32>,
    reflectivity: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    _pad3: vec4<f32>,
    _pad4: vec4<f32>,
}

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    color: vec4<f32>,
    reflectivity: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: vec4<f32>,
}

struct Light {
    position: vec4<f32>,
    intensity: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> lights: array<Light>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output: array<u32>;

fn rand_simple(x: u32, y: u32, s: u32) -> f32 {
    var n = x * 374761393u + y * 668265263u + s * 1013904223u;
    n = (n ^ (n >> 13u)) * 1274126177u;
    return f32(n & 0xFFFFu) / 65535.0;
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Hit {
    t: f32,
    normal: vec3<f32>,
    color: vec3<f32>,
    reflectivity: f32,
    hit: bool,
}

fn intersect_sphere(ray: Ray, sphere: Sphere) -> Hit {
    var result: Hit;
    result.hit = false;
    
    let oc = ray.origin - sphere.center.xyz;
    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant >= 0.0 {
        var t = (-b - sqrt(discriminant)) / (2.0 * a);
        if t <= 0.001 {
            t = (-b + sqrt(discriminant)) / (2.0 * a);
        }
        if t > 0.001 {
            result.hit = true;
            result.t = t;
            let hit_point = ray.origin + ray.direction * t;
            result.normal = normalize(hit_point - sphere.center.xyz);
            result.color = sphere.color.xyz;
            result.reflectivity = sphere.reflectivity;
        }
    }
    return result;
}

fn intersect_triangle(ray: Ray, tri: Triangle) -> Hit {
    var result: Hit;
    result.hit = false;
    
    let edge1 = tri.v1.xyz - tri.v0.xyz;
    let edge2 = tri.v2.xyz - tri.v0.xyz;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    
    if abs(a) > 1e-6 {
        let f = 1.0 / a;
        let s = ray.origin - tri.v0.xyz;
        let u = f * dot(s, h);
        
        if u >= 0.0 && u <= 1.0 {
            let q = cross(s, edge1);
            let v = f * dot(ray.direction, q);
            
            if v >= 0.0 && u + v <= 1.0 {
                let t = f * dot(edge2, q);
                if t > 0.001 {
                    result.hit = true;
                    result.t = t;
                    result.normal = normalize(cross(edge1, edge2));
                    if dot(result.normal, ray.direction) > 0.0 {
                        result.normal = -result.normal;
                    }
                    result.color = tri.color.xyz;
                    result.reflectivity = tri.reflectivity;
                }
            }
        }
    }
    return result;
}

fn find_closest_hit(ray: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;
    
    for (var i = 0u; i < params.num_spheres; i++) {
        let hit = intersect_sphere(ray, spheres[i]);
        if hit.hit && hit.t < closest.t {
            closest = hit;
        }
    }
    
    for (var i = 0u; i < params.num_triangles; i++) {
        let hit = intersect_triangle(ray, triangles[i]);
        if hit.hit && hit.t < closest.t {
            closest = hit;
        }
    }
    
    return closest;
}

fn is_in_shadow(ray: Ray, light_dist: f32) -> bool {
    let hit = find_closest_hit(ray);
    return hit.hit && hit.t < light_dist;
}

fn reflect_vec(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn trace_ray(initial_ray: Ray) -> vec3<f32> {
    var ray = initial_ray;
    var color = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    
    for (var depth = 0u; depth <= params.max_bounces; depth++) {
        let hit = find_closest_hit(ray);
        
        if !hit.hit {
            color += throughput * params.background.xyz;
            break;
        }
        
        let hit_point = ray.origin + ray.direction * hit.t;
        
        // Ambient
        var local_color = hit.color * 0.1;
        
        // Lighting
        for (var i = 0u; i < params.num_lights; i++) {
            let light = lights[i];
            let light_dir = normalize(light.position.xyz - hit_point);
            let light_dist = length(light.position.xyz - hit_point);
            
            var shadow_ray: Ray;
            shadow_ray.origin = hit_point + hit.normal * 0.001;
            shadow_ray.direction = light_dir;
            
            if !is_in_shadow(shadow_ray, light_dist) {
                let diffuse = max(dot(hit.normal, light_dir), 0.0);
                local_color += hit.color * diffuse * light.intensity * 0.6;
                
                let reflect_dir = reflect_vec(-light_dir, hit.normal);
                let view_dir = normalize(-ray.direction);
                let specular = pow(max(dot(reflect_dir, view_dir), 0.0), 32.0);
                local_color += vec3<f32>(1.0) * specular * light.intensity * 0.3;
            }
        }
        
        color += throughput * local_color * (1.0 - hit.reflectivity);
        
        if hit.reflectivity <= 0.0 || depth >= params.max_bounces {
            break;
        }
        
        // Continue with reflection
        throughput *= hit.reflectivity;
        ray.origin = hit_point + hit.normal * 0.001;
        ray.direction = reflect_vec(ray.direction, hit.normal);
    }
    
    return color;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if x >= params.width || y >= params.height {
        return;
    }
    
    let idx = y * params.width + x;
    let aspect = f32(params.width) / f32(params.height);
    let scale = tan(params.fov / 2.0);
    
    let jx = select(0.0, (rand_simple(x, y, params.sample) - 0.5) * 0.5, params.sample > 0u);
    let jy = select(0.0, (rand_simple(y, x, params.sample) - 0.5) * 0.5, params.sample > 0u);
    
    let px = (2.0 * (f32(x) + 0.5 + jx) / f32(params.width) - 1.0) * scale * aspect;
    let py = (1.0 - 2.0 * (f32(y) + 0.5 + jy) / f32(params.height)) * scale;
    
    // Compute ray direction using camera basis
    let dir = normalize(params.camera_forward.xyz + px * params.camera_right.xyz + py * params.camera_up.xyz);
    
    var ray: Ray;
    ray.origin = params.camera_pos.xyz;
    ray.direction = dir;
    
    let color = trace_ray(ray);
    
    // Accumulate
    let prev = accum[idx];
    let new_accum = prev + vec4<f32>(color, 1.0);
    accum[idx] = new_accum;
    
    // Output averaged color
    let samples = f32(params.sample + 1u);
    let final_color = clamp(new_accum.xyz / samples, vec3<f32>(0.0), vec3<f32>(1.0));
    
    let r = u32(final_color.x * 255.0);
    let g = u32(final_color.y * 255.0);
    let b = u32(final_color.z * 255.0);
    output[idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
"#;
