use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use std::fs;
use std::env;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn length(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalize(&self) -> Vec3 {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn mul(&self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }

    fn reflect(&self, normal: &Vec3) -> Vec3 {
        self.sub(&normal.mul(2.0 * self.dot(normal)))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn new(r: f64, g: f64, b: f64) -> Self {
        Color { r, g, b }
    }

    fn add(&self, other: &Color) -> Color {
        Color::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }

    fn mul(&self, t: f64) -> Color {
        Color::new(self.r * t, self.g * t, self.b * t)
    }

    fn clamp(&self) -> Color {
        Color::new(
            self.r.min(1.0).max(0.0),
            self.g.min(1.0).max(0.0),
            self.b.min(1.0).max(0.0),
        )
    }

    fn to_rgb(&self) -> Rgb<u8> {
        let c = self.clamp();
        Rgb([
            (c.r * 255.0) as u8,
            (c.g * 255.0) as u8,
            (c.b * 255.0) as u8,
        ])
    }
}

// Material properties shared by all geometry
#[derive(Debug, Clone, Deserialize)]
struct Material {
    color: Color,
    reflectivity: f64,
}

#[derive(Debug, Deserialize)]
struct Sphere {
    center: Vec3,
    radius: f64,
    color: Color,
    reflectivity: f64,
}

#[derive(Debug, Deserialize)]
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    color: Color,
    reflectivity: f64,
}

#[derive(Debug, Deserialize)]
struct Polygon {
    vertices: Vec<Vec3>,
    color: Color,
    reflectivity: f64,
}

#[derive(Debug, Deserialize)]
struct Light {
    position: Vec3,
    intensity: f64,
}

#[derive(Debug, Deserialize)]
struct Scene {
    #[serde(default)]
    spheres: Vec<Sphere>,
    #[serde(default)]
    triangles: Vec<Triangle>,
    #[serde(default)]
    polygons: Vec<Polygon>,
    lights: Vec<Light>,
    background: Color,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_max_bounces")]
    max_bounces: u32,
    #[serde(default = "default_render_time_ms")]
    render_time_ms: u64,
}

fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_max_bounces() -> u32 { 3 }
fn default_render_time_ms() -> u64 { 500 }

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

// Hit information
struct Hit {
    t: f64,
    normal: Vec3,
    color: Color,
    reflectivity: f64,
}

fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<Hit> {
    let oc = ray.origin.sub(&sphere.center);
    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.dot(&oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        None
    } else {
        let t = (-b - discriminant.sqrt()) / (2.0 * a);
        let t = if t > 0.001 {
            t
        } else {
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            if t2 > 0.001 { t2 } else { return None; }
        };
        
        let hit_point = ray.origin.add(&ray.direction.mul(t));
        let normal = hit_point.sub(&sphere.center).normalize();
        
        Some(Hit {
            t,
            normal,
            color: sphere.color.clone(),
            reflectivity: sphere.reflectivity,
        })
    }
}

// Möller–Trumbore intersection algorithm
fn intersect_triangle_points(ray: &Ray, v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Option<f64> {
    let edge1 = v1.sub(v0);
    let edge2 = v2.sub(v0);
    let h = ray.direction.cross(&edge2);
    let a = edge1.dot(&h);
    
    if a.abs() < 1e-10 {
        return None; // Ray parallel to triangle
    }
    
    let f = 1.0 / a;
    let s = ray.origin.sub(v0);
    let u = f * s.dot(&h);
    
    if u < 0.0 || u > 1.0 {
        return None;
    }
    
    let q = s.cross(&edge1);
    let v = f * ray.direction.dot(&q);
    
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    
    let t = f * edge2.dot(&q);
    
    if t > 0.001 {
        Some(t)
    } else {
        None
    }
}

fn intersect_triangle(ray: &Ray, tri: &Triangle) -> Option<Hit> {
    intersect_triangle_points(ray, &tri.v0, &tri.v1, &tri.v2).map(|t| {
        let edge1 = tri.v1.sub(&tri.v0);
        let edge2 = tri.v2.sub(&tri.v0);
        let mut normal = edge1.cross(&edge2).normalize();
        
        // Make sure normal faces the ray
        if normal.dot(&ray.direction) > 0.0 {
            normal = normal.mul(-1.0);
        }
        
        Hit {
            t,
            normal,
            color: tri.color.clone(),
            reflectivity: tri.reflectivity,
        }
    })
}

// Triangulate polygon (fan triangulation - works for convex polygons)
fn intersect_polygon(ray: &Ray, poly: &Polygon) -> Option<Hit> {
    if poly.vertices.len() < 3 {
        return None;
    }
    
    let mut closest: Option<(f64, Vec3)> = None;
    
    // Fan triangulation from first vertex
    for i in 1..poly.vertices.len() - 1 {
        if let Some(t) = intersect_triangle_points(ray, &poly.vertices[0], &poly.vertices[i], &poly.vertices[i + 1]) {
            match closest {
                None => {
                    let edge1 = poly.vertices[i].sub(&poly.vertices[0]);
                    let edge2 = poly.vertices[i + 1].sub(&poly.vertices[0]);
                    closest = Some((t, edge1.cross(&edge2).normalize()));
                }
                Some((ct, _)) if t < ct => {
                    let edge1 = poly.vertices[i].sub(&poly.vertices[0]);
                    let edge2 = poly.vertices[i + 1].sub(&poly.vertices[0]);
                    closest = Some((t, edge1.cross(&edge2).normalize()));
                }
                _ => {}
            }
        }
    }
    
    closest.map(|(t, mut normal)| {
        if normal.dot(&ray.direction) > 0.0 {
            normal = normal.mul(-1.0);
        }
        Hit {
            t,
            normal,
            color: poly.color.clone(),
            reflectivity: poly.reflectivity,
        }
    })
}

fn find_closest_hit(ray: &Ray, scene: &Scene) -> Option<Hit> {
    let mut closest: Option<Hit> = None;

    for sphere in &scene.spheres {
        if let Some(hit) = intersect_sphere(ray, sphere) {
            match &closest {
                None => closest = Some(hit),
                Some(c) if hit.t < c.t => closest = Some(hit),
                _ => {}
            }
        }
    }

    for tri in &scene.triangles {
        if let Some(hit) = intersect_triangle(ray, tri) {
            match &closest {
                None => closest = Some(hit),
                Some(c) if hit.t < c.t => closest = Some(hit),
                _ => {}
            }
        }
    }

    for poly in &scene.polygons {
        if let Some(hit) = intersect_polygon(ray, poly) {
            match &closest {
                None => closest = Some(hit),
                Some(c) if hit.t < c.t => closest = Some(hit),
                _ => {}
            }
        }
    }

    closest
}

fn is_in_shadow(ray: &Ray, light_dist: f64, scene: &Scene) -> bool {
    if let Some(hit) = find_closest_hit(ray, scene) {
        hit.t < light_dist
    } else {
        false
    }
}

fn trace_ray(ray: &Ray, scene: &Scene, depth: u32) -> Color {
    if depth > scene.max_bounces {
        return scene.background.clone();
    }

    match find_closest_hit(ray, scene) {
        None => scene.background.clone(),
        Some(hit) => {
            let hit_point = ray.origin.add(&ray.direction.mul(hit.t));

            // Ambient
            let mut color = hit.color.mul(0.1);

            // Diffuse and specular from lights
            for light in &scene.lights {
                let light_dir = light.position.sub(&hit_point).normalize();
                let light_dist = light.position.sub(&hit_point).length();

                // Shadow check
                let shadow_ray = Ray {
                    origin: hit_point.add(&hit.normal.mul(0.001)),
                    direction: light_dir.clone(),
                };

                if !is_in_shadow(&shadow_ray, light_dist, scene) {
                    // Diffuse
                    let diffuse = hit.normal.dot(&light_dir).max(0.0);
                    color = color.add(&hit.color.mul(diffuse * light.intensity * 0.6));

                    // Specular
                    let reflect_dir = light_dir.mul(-1.0).reflect(&hit.normal);
                    let view_dir = ray.direction.mul(-1.0).normalize();
                    let specular = reflect_dir.dot(&view_dir).max(0.0).powf(32.0);
                    color = color.add(&Color::new(1.0, 1.0, 1.0).mul(specular * light.intensity * 0.3));
                }
            }

            // Reflection
            if hit.reflectivity > 0.0 && depth < scene.max_bounces {
                let reflect_dir = ray.direction.reflect(&hit.normal);
                let reflect_ray = Ray {
                    origin: hit_point.add(&hit.normal.mul(0.001)),
                    direction: reflect_dir,
                };
                let reflected_color = trace_ray(&reflect_ray, scene, depth + 1);
                color = color.mul(1.0 - hit.reflectivity)
                    .add(&reflected_color.mul(hit.reflectivity));
            }

            color
        }
    }
}

fn render(scene: &Scene, time_limit: Duration) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, u32) {
    let width = scene.width;
    let height = scene.height;
    let total_pixels = width * height;
    
    // Accumulation buffer for progressive rendering
    let mut accum: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); total_pixels as usize];
    let mut samples = 0u32;

    let aspect = width as f64 / height as f64;
    let fov = std::f64::consts::PI / 3.0;
    let scale = (fov / 2.0).tan();
    let camera_pos = Vec3::new(0.0, 0.0, 0.0);

    let start = Instant::now();
    
    // Keep rendering passes until time runs out
    loop {
        for y in 0..height {
            for x in 0..width {
                // Add slight jitter for anti-aliasing on multiple samples
                let jx = if samples > 0 { (rand_simple(x, y, samples) - 0.5) * 0.5 } else { 0.0 };
                let jy = if samples > 0 { (rand_simple(y, x, samples) - 0.5) * 0.5 } else { 0.0 };
                
                let px = (2.0 * (x as f64 + 0.5 + jx) / width as f64 - 1.0) * scale * aspect;
                let py = (1.0 - 2.0 * (y as f64 + 0.5 + jy) / height as f64) * scale;

                let ray = Ray {
                    origin: camera_pos.clone(),
                    direction: Vec3::new(px, py, -1.0).normalize(),
                };

                let color = trace_ray(&ray, scene, 0);
                let idx = (y * width + x) as usize;
                accum[idx].0 += color.r;
                accum[idx].1 += color.g;
                accum[idx].2 += color.b;
            }
        }
        samples += 1;
        
        if start.elapsed() >= time_limit {
            break;
        }
    }

    // Convert accumulation buffer to image
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let color = Color::new(
                accum[idx].0 / samples as f64,
                accum[idx].1 / samples as f64,
                accum[idx].2 / samples as f64,
            );
            img.put_pixel(x, y, color.to_rgb());
        }
    }

    (img, samples)
}

// Simple deterministic "random" for jittering
fn rand_simple(x: u32, y: u32, sample: u32) -> f64 {
    let n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(sample.wrapping_mul(1013904223));
    let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0xFFFF) as f64 / 65535.0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: raytracer <scene.json> <output.png> [time_ms]");
        std::process::exit(1);
    }

    let scene_file = &args[1];
    let output_file = &args[2];
    let time_ms: u64 = args.get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 means use scene default

    let scene_json = fs::read_to_string(scene_file)
        .expect("Failed to read scene file");
    let scene: Scene = serde_json::from_str(&scene_json)
        .expect("Failed to parse scene JSON");

    let render_time = if time_ms > 0 { time_ms } else { scene.render_time_ms };
    let time_limit = Duration::from_millis(render_time);

    let geom_count = scene.spheres.len() + scene.triangles.len() + scene.polygons.len();
    println!("Rendering {}x{} with {} objects, {} lights, {} max bounces, {}ms time limit...",
             scene.width, scene.height, geom_count, scene.lights.len(), scene.max_bounces, render_time);

    let start = Instant::now();
    let (img, samples) = render(&scene, time_limit);
    let elapsed = start.elapsed();

    img.save(output_file).expect("Failed to save image");
    println!("Rendered {} samples in {:.2}s -> {}", samples, elapsed.as_secs_f64(), output_file);
}
