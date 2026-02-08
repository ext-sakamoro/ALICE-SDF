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
        TextureSlot {
            path: path.into(),
            uv_channel: 0,
            tiling: [1.0, 1.0],
            offset: [0.0, 0.0],
        }
    }

    /// Set UV channel (0 = primary, 1 = lightmap)
    pub fn with_uv_channel(mut self, channel: u32) -> Self {
        self.uv_channel = channel;
        self
    }

    /// Set tiling factor
    pub fn with_tiling(mut self, u: f32, v: f32) -> Self {
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
        Material {
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
        Material {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set base color (RGBA)
    #[inline]
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
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
    pub fn with_emission(mut self, r: f32, g: f32, b: f32, strength: f32) -> Self {
        self.emission = [r, g, b];
        self.emission_strength = strength;
        self
    }

    /// Create a metal material
    pub fn metal(name: impl Into<String>, r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Material::new(name)
            .with_color(r, g, b, 1.0)
            .with_metallic(1.0)
            .with_roughness(roughness)
    }

    /// Create a dielectric (non-metal) material
    pub fn dielectric(name: impl Into<String>, r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Material::new(name)
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
    pub fn with_volume(
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
        Material {
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
        Material::new(name).with_emission(r, g, b, strength)
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
    pub fn solid(r: f32, g: f32, b: f32) -> Self {
        ParticleMaterial {
            color: [r, g, b, 1.0],
            emission_strength: 0.0,
            opacity: 1.0,
        }
    }

    /// Create a glowing particle material
    #[inline]
    pub fn glow(r: f32, g: f32, b: f32, strength: f32) -> Self {
        ParticleMaterial {
            color: [r, g, b, 1.0],
            emission_strength: strength,
            opacity: 1.0,
        }
    }

    /// Create from a full PBR Material (lossy conversion)
    #[inline]
    pub fn from_material(mat: &Material) -> Self {
        ParticleMaterial {
            color: mat.base_color,
            emission_strength: mat.emission_strength,
            opacity: mat.opacity,
        }
    }
}

impl Material {
    /// Convert to a lightweight ParticleMaterial (lossy)
    #[inline]
    pub fn to_particle(&self) -> ParticleMaterial {
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
        MaterialLibrary {
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
}
