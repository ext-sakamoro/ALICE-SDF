//! PBR Material System for AAA rendering
//!
//! Provides physically-based rendering material definitions compatible
//! with major game engines (UE5, Unity, Godot).
//!
//! # Features
//! - PBR metallic-roughness workflow
//! - Material library with named materials
//! - Default materials for common use cases
//!
//! Author: Moroya Sakamoto

use serde::{Deserialize, Serialize};

/// Texture map reference for PBR materials
///
/// Points to an external texture file. When exported to glTF/OBJ,
/// these paths are resolved relative to the output file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextureSlot {
    /// File path to the texture image (PNG, JPEG, EXR, etc.)
    pub path: String,
    /// UV channel index (0 = primary UV, 1 = lightmap UV)
    pub uv_channel: u32,
    /// Tiling factor (repeat count)
    pub tiling: [f32; 2],
    /// UV offset
    pub offset: [f32; 2],
}

impl TextureSlot {
    /// Create a texture slot from a file path
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            uv_channel: 0,
            tiling: [1.0, 1.0],
            offset: [0.0, 0.0],
        }
    }

    /// Set UV channel (0 = primary, 1 = lightmap)
    pub const fn with_uv_channel(mut self, channel: u32) -> Self {
        self.uv_channel = channel;
        self
    }

    /// Set tiling factor
    pub const fn with_tiling(mut self, u: f32, v: f32) -> Self {
        self.tiling = [u, v];
        self
    }
}

/// PBR Material definition
///
/// Uses the metallic-roughness workflow (glTF 2.0 / UE5 / Unity HDRP).
/// Supports both flat values and texture maps for all PBR channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Material name
    pub name: String,
    /// Base color (RGBA, linear space)
    pub base_color: [f32; 4],
    /// Metallic factor (0.0 = dielectric, 1.0 = metal)
    pub metallic: f32,
    /// Roughness factor (0.0 = mirror, 1.0 = diffuse)
    pub roughness: f32,
    /// Emissive color (RGB, linear space)
    pub emission: [f32; 3],
    /// Emissive intensity multiplier
    pub emission_strength: f32,
    /// Opacity (0.0 = transparent, 1.0 = opaque)
    pub opacity: f32,
    /// Index of refraction (glass = 1.5, water = 1.33, diamond = 2.42)
    pub ior: f32,
    /// Normal map strength multiplier
    pub normal_scale: f32,

    // --- Texture Maps (AAA pipeline) ---
    /// Albedo / base color texture map
    pub albedo_map: Option<TextureSlot>,
    /// Normal map (tangent-space)
    pub normal_map: Option<TextureSlot>,
    /// Metallic map (grayscale, R channel)
    pub metallic_map: Option<TextureSlot>,
    /// Roughness map (grayscale, R channel)
    pub roughness_map: Option<TextureSlot>,
    /// Ambient occlusion map (grayscale, R channel)
    pub ao_map: Option<TextureSlot>,
    /// Emissive map (RGB)
    pub emissive_map: Option<TextureSlot>,
    /// Metallic-roughness combined map (glTF: B=metallic, G=roughness)
    pub metallic_roughness_map: Option<TextureSlot>,

    // --- Extended PBR (AAA pipeline) ---
    /// Clearcoat factor (0.0 = no clearcoat, 1.0 = full clearcoat)
    pub clearcoat: f32,
    /// Clearcoat roughness
    pub clearcoat_roughness: f32,
    /// Clearcoat normal map
    pub clearcoat_normal_map: Option<TextureSlot>,

    /// Sheen color (RGB, linear space)
    pub sheen_color: [f32; 3],
    /// Sheen roughness factor
    pub sheen_roughness: f32,

    /// Transmission factor (0.0 = opaque, 1.0 = fully transmissive)
    pub transmission: f32,
    /// Volume thickness (for transmission/SSS)
    pub thickness: f32,
    /// Attenuation distance (for volume absorption)
    pub attenuation_distance: f32,
    /// Attenuation color (for volume absorption, RGB linear)
    pub attenuation_color: [f32; 3],

    /// Anisotropy strength (-1.0 to 1.0)
    pub anisotropy: f32,
    /// Anisotropy rotation (0.0 to 1.0, mapped to 0-2π)
    pub anisotropy_rotation: f32,

    /// Subsurface scattering factor (0.0 = no SSS)
    pub subsurface: f32,
    /// Subsurface color (RGB, linear space)
    pub subsurface_color: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emission: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            opacity: 1.0,
            ior: 1.5,
            normal_scale: 1.0,
            albedo_map: None,
            normal_map: None,
            metallic_map: None,
            roughness_map: None,
            ao_map: None,
            emissive_map: None,
            metallic_roughness_map: None,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            clearcoat_normal_map: None,
            sheen_color: [0.0, 0.0, 0.0],
            sheen_roughness: 0.0,
            transmission: 0.0,
            thickness: 0.0,
            attenuation_distance: f32::INFINITY,
            attenuation_color: [1.0, 1.0, 1.0],
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0, 1.0, 1.0],
        }
    }
}

impl Material {
    /// Create a new material with a name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set base color (RGBA)
    #[inline]
    pub const fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.base_color = [r, g, b, a];
        self
    }

    /// Set metallic factor
    #[inline]
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    /// Set roughness factor
    #[inline]
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Set emission color and strength
    #[inline]
    pub const fn with_emission(mut self, r: f32, g: f32, b: f32, strength: f32) -> Self {
        self.emission = [r, g, b];
        self.emission_strength = strength;
        self
    }

    /// Create a metal material
    pub fn metal(name: impl Into<String>, r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Self::new(name)
            .with_color(r, g, b, 1.0)
            .with_metallic(1.0)
            .with_roughness(roughness)
    }

    /// Create a dielectric (non-metal) material
    pub fn dielectric(name: impl Into<String>, r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Self::new(name)
            .with_color(r, g, b, 1.0)
            .with_metallic(0.0)
            .with_roughness(roughness)
    }

    /// Set albedo texture map
    #[inline]
    pub fn with_albedo_map(mut self, path: impl Into<String>) -> Self {
        self.albedo_map = Some(TextureSlot::new(path));
        self
    }

    /// Set normal map
    #[inline]
    pub fn with_normal_map(mut self, path: impl Into<String>) -> Self {
        self.normal_map = Some(TextureSlot::new(path));
        self
    }

    /// Set metallic-roughness combined map (glTF workflow)
    #[inline]
    pub fn with_metallic_roughness_map(mut self, path: impl Into<String>) -> Self {
        self.metallic_roughness_map = Some(TextureSlot::new(path));
        self
    }

    /// Set ambient occlusion map
    #[inline]
    pub fn with_ao_map(mut self, path: impl Into<String>) -> Self {
        self.ao_map = Some(TextureSlot::new(path));
        self
    }

    /// Set emissive map
    #[inline]
    pub fn with_emissive_map(mut self, path: impl Into<String>) -> Self {
        self.emissive_map = Some(TextureSlot::new(path));
        self
    }

    /// Set clearcoat properties (car paint, lacquer, etc.)
    #[inline]
    pub fn with_clearcoat(mut self, factor: f32, roughness: f32) -> Self {
        self.clearcoat = factor.clamp(0.0, 1.0);
        self.clearcoat_roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Set sheen properties (fabric, velvet, etc.)
    #[inline]
    pub fn with_sheen(mut self, r: f32, g: f32, b: f32, roughness: f32) -> Self {
        self.sheen_color = [r, g, b];
        self.sheen_roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Set transmission factor (glass, water, etc.)
    #[inline]
    pub fn with_transmission(mut self, factor: f32) -> Self {
        self.transmission = factor.clamp(0.0, 1.0);
        self
    }

    /// Set volume properties (for transmission materials)
    #[inline]
    pub const fn with_volume(
        mut self,
        thickness: f32,
        attenuation_distance: f32,
        r: f32,
        g: f32,
        b: f32,
    ) -> Self {
        self.thickness = thickness;
        self.attenuation_distance = attenuation_distance;
        self.attenuation_color = [r, g, b];
        self
    }

    /// Set anisotropy (brushed metal, hair, etc.)
    #[inline]
    pub fn with_anisotropy(mut self, strength: f32, rotation: f32) -> Self {
        self.anisotropy = strength.clamp(-1.0, 1.0);
        self.anisotropy_rotation = rotation;
        self
    }

    /// Set subsurface scattering (skin, wax, marble, etc.)
    #[inline]
    pub fn with_subsurface(mut self, factor: f32, r: f32, g: f32, b: f32) -> Self {
        self.subsurface = factor.clamp(0.0, 1.0);
        self.subsurface_color = [r, g, b];
        self
    }

    /// Create a glass material
    pub fn glass(name: impl Into<String>, ior: f32) -> Self {
        Self {
            name: name.into(),
            base_color: [1.0, 1.0, 1.0, 0.1],
            metallic: 0.0,
            roughness: 0.0,
            emission: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            opacity: 0.1,
            ior,
            normal_scale: 1.0,
            transmission: 1.0,
            ..Default::default()
        }
    }

    /// Create an emissive material
    pub fn emissive(name: impl Into<String>, r: f32, g: f32, b: f32, strength: f32) -> Self {
        Self::new(name).with_emission(r, g, b, strength)
    }
}

/// Lightweight material for particle rendering (24 bytes)
///
/// Optimized for GPU transfer of thousands of particles.
/// Contains only the essential properties: color, emission, and opacity.
/// Use this instead of `Material` when rendering SDF text particles in
/// browsers or real-time particle systems.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ParticleMaterial {
    /// RGBA color (linear space)
    pub color: [f32; 4],
    /// Emission intensity (0.0 = no glow)
    pub emission_strength: f32,
    /// Opacity (0.0 = transparent, 1.0 = opaque)
    pub opacity: f32,
}

impl ParticleMaterial {
    /// Create a solid-color particle material
    #[inline]
    pub const fn solid(r: f32, g: f32, b: f32) -> Self {
        Self {
            color: [r, g, b, 1.0],
            emission_strength: 0.0,
            opacity: 1.0,
        }
    }

    /// Create a glowing particle material
    #[inline]
    pub const fn glow(r: f32, g: f32, b: f32, strength: f32) -> Self {
        Self {
            color: [r, g, b, 1.0],
            emission_strength: strength,
            opacity: 1.0,
        }
    }

    /// Create from a full PBR Material (lossy conversion)
    #[inline]
    pub const fn from_material(mat: &Material) -> Self {
        Self {
            color: mat.base_color,
            emission_strength: mat.emission_strength,
            opacity: mat.opacity,
        }
    }
}

impl Material {
    /// Convert to a lightweight ParticleMaterial (lossy)
    #[inline]
    pub const fn to_particle(&self) -> ParticleMaterial {
        ParticleMaterial::from_material(self)
    }
}

/// Material library for managing multiple materials
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterialLibrary {
    /// Materials indexed by ID
    pub materials: Vec<Material>,
}

impl MaterialLibrary {
    /// Create an empty material library with a default material at index 0
    pub fn new() -> Self {
        Self {
            materials: vec![Material::default()],
        }
    }

    /// Add a material and return its ID
    pub fn add(&mut self, material: Material) -> u32 {
        let id = self.materials.len() as u32;
        self.materials.push(material);
        id
    }

    /// Get a material by ID
    #[inline]
    pub fn get(&self, id: u32) -> Option<&Material> {
        self.materials.get(id as usize)
    }

    /// Get the default material (ID 0)
    #[inline]
    pub fn default_material(&self) -> &Material {
        &self.materials[0]
    }

    /// Number of materials in the library
    #[inline]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Check if the library is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Iterate over all materials with their IDs
    pub fn iter(&self) -> impl Iterator<Item = (u32, &Material)> {
        self.materials
            .iter()
            .enumerate()
            .map(|(i, m)| (i as u32, m))
    }

    /// Find a material by name (first match)
    pub fn find_by_name(&self, name: &str) -> Option<(u32, &Material)> {
        self.iter().find(|(_, m)| m.name == name)
    }
}

/// Linearly interpolate between two materials.
///
/// Interpolates scalar PBR properties. Texture slots are taken from `a`
/// when t < 0.5, otherwise from `b`. Name is concatenated.
pub fn material_lerp(a: &Material, b: &Material, t: f32) -> Material {
    let t = t.clamp(0.0, 1.0);
    let it = 1.0 - t;
    Material {
        name: format!("{}_{}_blend", a.name, b.name),
        base_color: [
            a.base_color[0].mul_add(it, b.base_color[0] * t),
            a.base_color[1].mul_add(it, b.base_color[1] * t),
            a.base_color[2].mul_add(it, b.base_color[2] * t),
            a.base_color[3].mul_add(it, b.base_color[3] * t),
        ],
        metallic: a.metallic.mul_add(it, b.metallic * t),
        roughness: a.roughness.mul_add(it, b.roughness * t),
        emission: [
            a.emission[0].mul_add(it, b.emission[0] * t),
            a.emission[1].mul_add(it, b.emission[1] * t),
            a.emission[2].mul_add(it, b.emission[2] * t),
        ],
        emission_strength: a.emission_strength.mul_add(it, b.emission_strength * t),
        opacity: a.opacity.mul_add(it, b.opacity * t),
        ior: a.ior.mul_add(it, b.ior * t),
        normal_scale: a.normal_scale.mul_add(it, b.normal_scale * t),
        clearcoat: a.clearcoat.mul_add(it, b.clearcoat * t),
        clearcoat_roughness: a.clearcoat_roughness.mul_add(it, b.clearcoat_roughness * t),
        transmission: a.transmission.mul_add(it, b.transmission * t),
        thickness: a.thickness.mul_add(it, b.thickness * t),
        attenuation_distance: a.attenuation_distance.min(b.attenuation_distance),
        attenuation_color: [
            a.attenuation_color[0].mul_add(it, b.attenuation_color[0] * t),
            a.attenuation_color[1].mul_add(it, b.attenuation_color[1] * t),
            a.attenuation_color[2].mul_add(it, b.attenuation_color[2] * t),
        ],
        anisotropy: a.anisotropy.mul_add(it, b.anisotropy * t),
        anisotropy_rotation: a.anisotropy_rotation.mul_add(it, b.anisotropy_rotation * t),
        subsurface: a.subsurface.mul_add(it, b.subsurface * t),
        subsurface_color: [
            a.subsurface_color[0].mul_add(it, b.subsurface_color[0] * t),
            a.subsurface_color[1].mul_add(it, b.subsurface_color[1] * t),
            a.subsurface_color[2].mul_add(it, b.subsurface_color[2] * t),
        ],
        sheen_color: [
            a.sheen_color[0].mul_add(it, b.sheen_color[0] * t),
            a.sheen_color[1].mul_add(it, b.sheen_color[1] * t),
            a.sheen_color[2].mul_add(it, b.sheen_color[2] * t),
        ],
        sheen_roughness: a.sheen_roughness.mul_add(it, b.sheen_roughness * t),
        // Textures: take from nearest side
        albedo_map: if t < 0.5 {
            a.albedo_map.clone()
        } else {
            b.albedo_map.clone()
        },
        normal_map: if t < 0.5 {
            a.normal_map.clone()
        } else {
            b.normal_map.clone()
        },
        metallic_map: if t < 0.5 {
            a.metallic_map.clone()
        } else {
            b.metallic_map.clone()
        },
        roughness_map: if t < 0.5 {
            a.roughness_map.clone()
        } else {
            b.roughness_map.clone()
        },
        ao_map: if t < 0.5 {
            a.ao_map.clone()
        } else {
            b.ao_map.clone()
        },
        emissive_map: if t < 0.5 {
            a.emissive_map.clone()
        } else {
            b.emissive_map.clone()
        },
        metallic_roughness_map: if t < 0.5 {
            a.metallic_roughness_map.clone()
        } else {
            b.metallic_roughness_map.clone()
        },
        clearcoat_normal_map: if t < 0.5 {
            a.clearcoat_normal_map.clone()
        } else {
            b.clearcoat_normal_map.clone()
        },
    }
}

/// Standard material presets for common real-world materials.
pub struct StandardMaterials;

impl StandardMaterials {
    /// Polished gold
    pub fn gold() -> Material {
        Material::metal("Gold", 1.0, 0.766, 0.336, 0.1)
    }

    /// Brushed aluminum
    pub fn aluminum() -> Material {
        Material::metal("Aluminum", 0.913, 0.922, 0.924, 0.35).with_anisotropy(0.5, 0.0)
    }

    /// Polished copper
    pub fn copper() -> Material {
        Material::metal("Copper", 0.955, 0.638, 0.538, 0.15)
    }

    /// Matte plastic (white)
    pub fn plastic_white() -> Material {
        Material::dielectric("PlasticWhite", 0.9, 0.9, 0.9, 0.5)
    }

    /// Glossy red plastic
    pub fn plastic_red() -> Material {
        Material::dielectric("PlasticRed", 0.8, 0.05, 0.05, 0.2)
    }

    /// Clear glass
    pub fn glass() -> Material {
        Material::glass("Glass", 1.5)
    }

    /// Diamond
    pub fn diamond() -> Material {
        Material::glass("Diamond", 2.42).with_roughness(0.0)
    }

    /// Water
    pub fn water() -> Material {
        Material::new("Water")
            .with_color(0.3, 0.5, 0.7, 0.6)
            .with_transmission(0.9)
            .with_roughness(0.05)
    }

    /// Human skin
    pub fn skin() -> Material {
        Material::dielectric("Skin", 0.8, 0.6, 0.5, 0.5).with_subsurface(0.5, 0.9, 0.4, 0.3)
    }

    /// Marble
    pub fn marble() -> Material {
        Material::dielectric("Marble", 0.95, 0.93, 0.88, 0.15)
            .with_subsurface(0.2, 0.95, 0.93, 0.88)
    }

    /// Wet asphalt
    pub fn wet_asphalt() -> Material {
        Material::dielectric("WetAsphalt", 0.1, 0.1, 0.1, 0.15).with_clearcoat(0.8, 0.05)
    }

    /// Velvet fabric
    pub fn velvet() -> Material {
        Material::dielectric("Velvet", 0.3, 0.05, 0.1, 0.8).with_sheen(0.6, 0.1, 0.2, 0.5)
    }

    /// Rubber
    pub fn rubber() -> Material {
        Material::dielectric("Rubber", 0.15, 0.15, 0.15, 0.9)
    }

    /// Concrete
    pub fn concrete() -> Material {
        Material::dielectric("Concrete", 0.6, 0.58, 0.55, 0.85)
    }

    /// Polished chrome
    pub fn chrome() -> Material {
        Material::metal("Chrome", 0.95, 0.93, 0.88, 0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_material() {
        let mat = Material::default();
        assert_eq!(mat.metallic, 0.0);
        assert_eq!(mat.roughness, 0.5);
        assert_eq!(mat.opacity, 1.0);
    }

    #[test]
    fn test_metal_material() {
        let gold = Material::metal("Gold", 1.0, 0.766, 0.336, 0.3);
        assert_eq!(gold.metallic, 1.0);
        assert_eq!(gold.roughness, 0.3);
        assert_eq!(gold.base_color[0], 1.0);
    }

    #[test]
    fn test_glass_material() {
        let glass = Material::glass("Glass", 1.5);
        assert_eq!(glass.ior, 1.5);
        assert_eq!(glass.opacity, 0.1);
        assert_eq!(glass.roughness, 0.0);
    }

    #[test]
    fn test_material_library() {
        let mut lib = MaterialLibrary::new();
        assert_eq!(lib.len(), 1); // default material

        let id = lib.add(Material::metal("Steel", 0.7, 0.7, 0.7, 0.4));
        assert_eq!(id, 1);
        assert_eq!(lib.len(), 2);

        let steel = lib.get(id).unwrap();
        assert_eq!(steel.name, "Steel");
    }

    #[test]
    fn test_builder_pattern() {
        let mat = Material::new("Custom")
            .with_color(1.0, 0.0, 0.0, 1.0)
            .with_metallic(0.8)
            .with_roughness(0.2)
            .with_emission(1.0, 0.5, 0.0, 5.0);

        assert_eq!(mat.base_color[0], 1.0);
        assert_eq!(mat.metallic, 0.8);
        assert_eq!(mat.emission_strength, 5.0);
    }

    #[test]
    fn test_texture_slots() {
        let mat = Material::new("Textured")
            .with_albedo_map("textures/albedo.png")
            .with_normal_map("textures/normal.png")
            .with_metallic_roughness_map("textures/mr.png")
            .with_ao_map("textures/ao.png");

        assert!(mat.albedo_map.is_some());
        assert_eq!(mat.albedo_map.as_ref().unwrap().path, "textures/albedo.png");
        assert!(mat.normal_map.is_some());
        assert!(mat.metallic_roughness_map.is_some());
        assert!(mat.ao_map.is_some());
        assert!(mat.emissive_map.is_none());
    }

    #[test]
    fn test_texture_slot_tiling() {
        let slot = TextureSlot::new("tex.png")
            .with_uv_channel(1)
            .with_tiling(2.0, 3.0);

        assert_eq!(slot.uv_channel, 1);
        assert_eq!(slot.tiling, [2.0, 3.0]);
    }

    #[test]
    fn test_extended_pbr_clearcoat() {
        let mat = Material::new("CarPaint")
            .with_color(0.8, 0.0, 0.0, 1.0)
            .with_clearcoat(1.0, 0.1);

        assert_eq!(mat.clearcoat, 1.0);
        assert_eq!(mat.clearcoat_roughness, 0.1);
    }

    #[test]
    fn test_extended_pbr_sheen() {
        let mat = Material::new("Velvet").with_sheen(0.5, 0.3, 0.8, 0.7);

        assert_eq!(mat.sheen_color, [0.5, 0.3, 0.8]);
        assert_eq!(mat.sheen_roughness, 0.7);
    }

    #[test]
    fn test_extended_pbr_transmission() {
        let glass = Material::glass("Glass", 1.5);
        assert_eq!(glass.transmission, 1.0);
        assert_eq!(glass.ior, 1.5);

        let water = Material::new("Water").with_transmission(0.95);
        assert_eq!(water.transmission, 0.95);
    }

    #[test]
    fn test_extended_pbr_volume() {
        let mat = Material::new("ColoredGlass")
            .with_transmission(1.0)
            .with_volume(0.5, 2.0, 0.2, 0.8, 0.2);

        assert_eq!(mat.thickness, 0.5);
        assert_eq!(mat.attenuation_distance, 2.0);
        assert_eq!(mat.attenuation_color, [0.2, 0.8, 0.2]);
    }

    #[test]
    fn test_extended_pbr_anisotropy() {
        let mat = Material::metal("BrushedSteel", 0.7, 0.7, 0.7, 0.3).with_anisotropy(0.8, 0.5);

        assert_eq!(mat.anisotropy, 0.8);
        assert_eq!(mat.anisotropy_rotation, 0.5);
    }

    #[test]
    fn test_particle_material_solid() {
        let p = ParticleMaterial::solid(1.0, 0.0, 0.0);
        assert_eq!(p.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.emission_strength, 0.0);
        assert_eq!(p.opacity, 1.0);
    }

    #[test]
    fn test_particle_material_glow() {
        let p = ParticleMaterial::glow(0.0, 1.0, 0.5, 10.0);
        assert_eq!(p.emission_strength, 10.0);
    }

    #[test]
    fn test_particle_material_from_material() {
        let mat = Material::metal("Gold", 1.0, 0.766, 0.336, 0.3).with_emission(1.0, 0.5, 0.0, 5.0);
        let p = mat.to_particle();
        assert_eq!(p.color[0], 1.0);
        assert_eq!(p.emission_strength, 5.0);
        assert_eq!(p.opacity, 1.0);
    }

    #[test]
    fn test_particle_material_size() {
        assert_eq!(std::mem::size_of::<ParticleMaterial>(), 24);
    }

    #[test]
    fn test_extended_pbr_subsurface() {
        let mat = Material::new("Skin").with_subsurface(0.6, 1.0, 0.5, 0.4);

        assert_eq!(mat.subsurface, 0.6);
        assert_eq!(mat.subsurface_color, [1.0, 0.5, 0.4]);
    }

    #[test]
    fn test_material_lerp_midpoint() {
        let a = Material::metal("A", 1.0, 0.0, 0.0, 0.0);
        let b = Material::metal("B", 0.0, 0.0, 1.0, 1.0);
        let mid = material_lerp(&a, &b, 0.5);
        assert!((mid.base_color[0] - 0.5).abs() < 1e-4);
        assert!((mid.base_color[2] - 0.5).abs() < 1e-4);
        assert!((mid.roughness - 0.5).abs() < 1e-4);
        assert!((mid.metallic - 1.0).abs() < 1e-4); // both 1.0
    }

    #[test]
    fn test_material_lerp_boundaries() {
        let a = Material::metal("Gold", 1.0, 0.766, 0.336, 0.1);
        let b = Material::dielectric("Plastic", 0.8, 0.0, 0.0, 0.5);
        let at_zero = material_lerp(&a, &b, 0.0);
        assert_eq!(at_zero.base_color[0], a.base_color[0]);
        let at_one = material_lerp(&a, &b, 1.0);
        assert_eq!(at_one.base_color[0], b.base_color[0]);
    }

    #[test]
    fn test_standard_materials_gold() {
        let gold = StandardMaterials::gold();
        assert_eq!(gold.metallic, 1.0);
        assert!(gold.roughness < 0.2);
    }

    #[test]
    fn test_standard_materials_glass() {
        let glass = StandardMaterials::glass();
        assert_eq!(glass.ior, 1.5);
        assert_eq!(glass.transmission, 1.0);
    }

    #[test]
    fn test_standard_materials_skin() {
        let skin = StandardMaterials::skin();
        assert!(skin.subsurface > 0.0);
    }

    #[test]
    fn test_standard_materials_variety() {
        // Ensure all presets construct without panic
        let _ = StandardMaterials::aluminum();
        let _ = StandardMaterials::copper();
        let _ = StandardMaterials::plastic_white();
        let _ = StandardMaterials::plastic_red();
        let _ = StandardMaterials::diamond();
        let _ = StandardMaterials::water();
        let _ = StandardMaterials::marble();
        let _ = StandardMaterials::wet_asphalt();
        let _ = StandardMaterials::velvet();
        let _ = StandardMaterials::rubber();
        let _ = StandardMaterials::concrete();
        let _ = StandardMaterials::chrome();
    }

    #[test]
    fn test_find_by_name() {
        let mut lib = MaterialLibrary::new();
        lib.add(Material::metal("Steel", 0.7, 0.7, 0.7, 0.4));
        lib.add(Material::metal("Gold", 1.0, 0.8, 0.3, 0.1));
        let (id, mat) = lib.find_by_name("Gold").unwrap();
        assert_eq!(id, 2);
        assert_eq!(mat.name, "Gold");
        assert!(lib.find_by_name("NotExist").is_none());
    }

    #[test]
    fn test_material_library_iter() {
        let mut lib = MaterialLibrary::new();
        lib.add(Material::new("A"));
        lib.add(Material::new("B"));
        assert_eq!(lib.iter().count(), 3); // default + A + B
    }

    #[test]
    fn test_dielectric_material() {
        let plastic = Material::dielectric("Plastic", 0.5, 0.5, 0.5, 0.6);
        assert_eq!(plastic.metallic, 0.0);
        assert_eq!(plastic.roughness, 0.6);
    }

    #[test]
    fn test_emissive_material() {
        let light = Material::emissive("Light", 1.0, 1.0, 1.0, 100.0);
        assert_eq!(light.emission_strength, 100.0);
        assert_eq!(light.emission, [1.0, 1.0, 1.0]);
    }
}
