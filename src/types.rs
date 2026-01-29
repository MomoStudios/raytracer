use serde::Deserialize;

/// 3D Vector - shared between CPU and GPU (will be converted to f32 for GPU)
#[derive(Debug, Clone, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        Vec3::new(self.x / len, self.y / len, self.z / len)
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn mul(&self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }

    pub fn reflect(&self, normal: &Vec3) -> Vec3 {
        self.sub(&normal.mul(2.0 * self.dot(normal)))
    }
    
    pub fn to_f32_array(&self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }
}

/// Color - RGB values 0.0-1.0
#[derive(Debug, Clone, Deserialize)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Color { r, g, b }
    }

    pub fn add(&self, other: &Color) -> Color {
        Color::new(self.r + other.r, self.g + other.g, self.b + other.b)
    }

    pub fn mul(&self, t: f64) -> Color {
        Color::new(self.r * t, self.g * t, self.b * t)
    }

    pub fn clamp(&self) -> Color {
        Color::new(
            self.r.min(1.0).max(0.0),
            self.g.min(1.0).max(0.0),
            self.b.min(1.0).max(0.0),
        )
    }
    
    pub fn to_f32_array(&self) -> [f32; 3] {
        [self.r as f32, self.g as f32, self.b as f32]
    }
}

/// Sphere geometry
#[derive(Debug, Deserialize)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub color: Color,
    pub reflectivity: f64,
}

/// Triangle geometry
#[derive(Debug, Deserialize)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub color: Color,
    pub reflectivity: f64,
}

/// Polygon geometry (convex, will be triangulated)
#[derive(Debug, Deserialize)]
pub struct Polygon {
    pub vertices: Vec<Vec3>,
    pub color: Color,
    pub reflectivity: f64,
}

/// Point light
#[derive(Debug, Deserialize)]
pub struct Light {
    pub position: Vec3,
    pub intensity: f64,
}

/// Scene definition
#[derive(Debug, Deserialize)]
pub struct Scene {
    #[serde(default)]
    pub spheres: Vec<Sphere>,
    #[serde(default)]
    pub triangles: Vec<Triangle>,
    #[serde(default)]
    pub polygons: Vec<Polygon>,
    pub lights: Vec<Light>,
    pub background: Color,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_max_bounces")]
    pub max_bounces: u32,
    #[serde(default = "default_render_time_ms")]
    pub render_time_ms: u64,
}

fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_max_bounces() -> u32 { 3 }
fn default_render_time_ms() -> u64 { 500 }

/// Ray for tracing
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Hit information from intersection
pub struct Hit {
    pub t: f64,
    pub normal: Vec3,
    pub color: Color,
    pub reflectivity: f64,
}

/// Render backend selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderBackend {
    Cpu,
    Gpu,
}

impl std::str::FromStr for RenderBackend {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(RenderBackend::Cpu),
            "gpu" => Ok(RenderBackend::Gpu),
            _ => Err(format!("Unknown backend: {}. Use 'cpu' or 'gpu'", s)),
        }
    }
}
