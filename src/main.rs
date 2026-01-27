use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use std::fs;
use std::env;
use std::time::Instant;

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

    fn blend(&self, other: &Color) -> Color {
        Color::new(self.r * other.r, self.g * other.g, self.b * other.b)
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

#[derive(Debug, Deserialize)]
struct Sphere {
    center: Vec3,
    radius: f64,
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
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
    background: Color,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_max_bounces")]
    max_bounces: u32,
}

fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_max_bounces() -> u32 { 3 }

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<f64> {
    let oc = ray.origin.sub(&sphere.center);
    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.dot(&oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        None
    } else {
        let t = (-b - discriminant.sqrt()) / (2.0 * a);
        if t > 0.001 {
            Some(t)
        } else {
            let t = (-b + discriminant.sqrt()) / (2.0 * a);
            if t > 0.001 {
                Some(t)
            } else {
                None
            }
        }
    }
}

fn find_closest_hit<'a>(ray: &Ray, spheres: &'a [Sphere]) -> Option<(f64, &'a Sphere)> {
    let mut closest: Option<(f64, &Sphere)> = None;

    for sphere in spheres {
        if let Some(t) = intersect_sphere(ray, sphere) {
            match closest {
                None => closest = Some((t, sphere)),
                Some((closest_t, _)) if t < closest_t => closest = Some((t, sphere)),
                _ => {}
            }
        }
    }

    closest
}

fn trace_ray(ray: &Ray, scene: &Scene, depth: u32) -> Color {
    if depth > scene.max_bounces {
        return scene.background.clone();
    }

    match find_closest_hit(ray, &scene.spheres) {
        None => scene.background.clone(),
        Some((t, sphere)) => {
            let hit_point = ray.origin.add(&ray.direction.mul(t));
            let normal = hit_point.sub(&sphere.center).normalize();

            // Ambient
            let mut color = sphere.color.mul(0.1);

            // Diffuse and specular from lights
            for light in &scene.lights {
                let light_dir = light.position.sub(&hit_point).normalize();
                let light_dist = light.position.sub(&hit_point).length();

                // Shadow check
                let shadow_ray = Ray {
                    origin: hit_point.add(&normal.mul(0.001)),
                    direction: light_dir.clone(),
                };

                let in_shadow = scene.spheres.iter().any(|s| {
                    if let Some(st) = intersect_sphere(&shadow_ray, s) {
                        st < light_dist
                    } else {
                        false
                    }
                });

                if !in_shadow {
                    // Diffuse
                    let diffuse = normal.dot(&light_dir).max(0.0);
                    color = color.add(&sphere.color.mul(diffuse * light.intensity * 0.6));

                    // Specular
                    let reflect_dir = light_dir.mul(-1.0).reflect(&normal);
                    let view_dir = ray.direction.mul(-1.0).normalize();
                    let specular = reflect_dir.dot(&view_dir).max(0.0).powf(32.0);
                    color = color.add(&Color::new(1.0, 1.0, 1.0).mul(specular * light.intensity * 0.3));
                }
            }

            // Reflection
            if sphere.reflectivity > 0.0 && depth < scene.max_bounces {
                let reflect_dir = ray.direction.reflect(&normal);
                let reflect_ray = Ray {
                    origin: hit_point.add(&normal.mul(0.001)),
                    direction: reflect_dir,
                };
                let reflected_color = trace_ray(&reflect_ray, scene, depth + 1);
                color = color.mul(1.0 - sphere.reflectivity)
                    .add(&reflected_color.mul(sphere.reflectivity));
            }

            color
        }
    }
}

fn render(scene: &Scene) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(scene.width, scene.height);

    let aspect = scene.width as f64 / scene.height as f64;
    let fov = std::f64::consts::PI / 3.0;
    let scale = (fov / 2.0).tan();

    let camera_pos = Vec3::new(0.0, 0.0, 0.0);

    for y in 0..scene.height {
        for x in 0..scene.width {
            let px = (2.0 * (x as f64 + 0.5) / scene.width as f64 - 1.0) * scale * aspect;
            let py = (1.0 - 2.0 * (y as f64 + 0.5) / scene.height as f64) * scale;

            let ray = Ray {
                origin: camera_pos.clone(),
                direction: Vec3::new(px, py, -1.0).normalize(),
            };

            let color = trace_ray(&ray, scene, 0);
            img.put_pixel(x, y, color.to_rgb());
        }
    }

    img
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: raytracer <scene.json> <output.png>");
        std::process::exit(1);
    }

    let scene_file = &args[1];
    let output_file = &args[2];

    let scene_json = fs::read_to_string(scene_file)
        .expect("Failed to read scene file");
    let scene: Scene = serde_json::from_str(&scene_json)
        .expect("Failed to parse scene JSON");

    println!("Rendering {}x{} with {} spheres, {} lights, {} max bounces...",
             scene.width, scene.height, scene.spheres.len(), scene.lights.len(), scene.max_bounces);

    let start = Instant::now();
    let img = render(&scene);
    let elapsed = start.elapsed();

    img.save(output_file).expect("Failed to save image");
    println!("Rendered in {:.2}s -> {}", elapsed.as_secs_f64(), output_file);
}
