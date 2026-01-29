use image::{ImageBuffer, Rgb};
use std::time::{Duration, Instant};
use crate::types::*;
use super::{Renderer, RenderResult};

pub struct CpuRenderer;

impl CpuRenderer {
    pub fn new() -> Self {
        CpuRenderer
    }
}

impl Renderer for CpuRenderer {
    fn name(&self) -> &'static str {
        "CPU"
    }
    
    fn render(&self, scene: &Scene, time_limit: Duration) -> RenderResult {
        let width = scene.width;
        let height = scene.height;
        let total_pixels = width * height;
        
        let mut accum: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); total_pixels as usize];
        let mut samples = 0u32;

        let aspect = width as f64 / height as f64;
        let fov = std::f64::consts::PI / 3.0;
        let scale = (fov / 2.0).tan();
        let camera_pos = Vec3::new(0.0, 0.0, 0.0);

        let start = Instant::now();
        
        loop {
            for y in 0..height {
                for x in 0..width {
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

        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let color = Color::new(
                    accum[idx].0 / samples as f64,
                    accum[idx].1 / samples as f64,
                    accum[idx].2 / samples as f64,
                );
                let c = color.clamp();
                img.put_pixel(x, y, Rgb([
                    (c.r * 255.0) as u8,
                    (c.g * 255.0) as u8,
                    (c.b * 255.0) as u8,
                ]));
            }
        }

        RenderResult {
            image: img,
            samples,
            backend: self.name(),
        }
    }
}

// Raytracing helpers

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

fn intersect_triangle_points(ray: &Ray, v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Option<f64> {
    let edge1 = v1.sub(v0);
    let edge2 = v2.sub(v0);
    let h = ray.direction.cross(&edge2);
    let a = edge1.dot(&h);
    
    if a.abs() < 1e-10 {
        return None;
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

fn intersect_polygon(ray: &Ray, poly: &Polygon) -> Option<Hit> {
    if poly.vertices.len() < 3 {
        return None;
    }
    
    let mut closest: Option<(f64, Vec3)> = None;
    
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
            let mut color = hit.color.mul(0.1);

            for light in &scene.lights {
                let light_dir = light.position.sub(&hit_point).normalize();
                let light_dist = light.position.sub(&hit_point).length();

                let shadow_ray = Ray {
                    origin: hit_point.add(&hit.normal.mul(0.001)),
                    direction: light_dir.clone(),
                };

                if !is_in_shadow(&shadow_ray, light_dist, scene) {
                    let diffuse = hit.normal.dot(&light_dir).max(0.0);
                    color = color.add(&hit.color.mul(diffuse * light.intensity * 0.6));

                    let reflect_dir = light_dir.mul(-1.0).reflect(&hit.normal);
                    let view_dir = ray.direction.mul(-1.0).normalize();
                    let specular = reflect_dir.dot(&view_dir).max(0.0).powf(32.0);
                    color = color.add(&Color::new(1.0, 1.0, 1.0).mul(specular * light.intensity * 0.3));
                }
            }

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

fn rand_simple(x: u32, y: u32, sample: u32) -> f64 {
    let n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(sample.wrapping_mul(1013904223));
    let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0xFFFF) as f64 / 65535.0
}
