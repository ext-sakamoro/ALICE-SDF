//! Irradiance Probes: 3D grid of L1 spherical harmonics (Deep Fried Edition)
//!
//! Stores indirect lighting as a grid of SH probes that can be
//! interpolated at runtime for fast indirect light lookups.
//!
//! Uses L1 spherical harmonics (4 coefficients per color channel)
//! for a good balance between quality and storage.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// L1 Spherical Harmonics coefficients for one color channel
///
/// 4 coefficients: DC (constant), X, Y, Z directional components.
#[derive(Debug, Clone, Copy)]
pub struct SH1 {
    /// Coefficients: [DC, X, Y, Z]
    pub coeffs: [f32; 4],
}

impl Default for SH1 {
    fn default() -> Self {
        SH1 { coeffs: [0.0; 4] }
    }
}

impl SH1 {
    /// Evaluate SH in a direction
    #[inline]
    pub fn evaluate(&self, dir: Vec3) -> f32 {
        // L0: 0.282095
        // L1: 0.488603 * {y, z, x}
        let c = &self.coeffs;
        c[0] * 0.282095
            + c[1] * 0.488603 * dir.x
            + c[2] * 0.488603 * dir.y
            + c[3] * 0.488603 * dir.z
    }

    /// Project a directional sample into SH
    #[inline]
    pub fn project(dir: Vec3, value: f32) -> Self {
        SH1 {
            coeffs: [
                value * 0.282095,
                value * 0.488603 * dir.x,
                value * 0.488603 * dir.y,
                value * 0.488603 * dir.z,
            ],
        }
    }

    /// Add another SH
    #[inline]
    pub fn add(&mut self, other: &SH1) {
        for i in 0..4 {
            self.coeffs[i] += other.coeffs[i];
        }
    }

    /// Scale all coefficients
    #[inline]
    pub fn scale(&mut self, s: f32) {
        for c in &mut self.coeffs {
            *c *= s;
        }
    }
}

/// A single irradiance probe storing L1 SH for RGB
///
/// Aligned to 64 bytes (cache line) for optimal iteration in probe grids.
#[derive(Debug, Clone, Copy)]
#[repr(align(64))]
pub struct IrradianceProbe {
    /// World-space position of the probe
    pub position: Vec3,
    /// Red channel SH
    pub sh_r: SH1,
    /// Green channel SH
    pub sh_g: SH1,
    /// Blue channel SH
    pub sh_b: SH1,
}

impl Default for IrradianceProbe {
    fn default() -> Self {
        IrradianceProbe {
            position: Vec3::ZERO,
            sh_r: SH1::default(),
            sh_g: SH1::default(),
            sh_b: SH1::default(),
        }
    }
}

impl IrradianceProbe {
    /// Evaluate the irradiance in a given direction
    pub fn evaluate(&self, direction: Vec3) -> Vec3 {
        Vec3::new(
            self.sh_r.evaluate(direction).max(0.0),
            self.sh_g.evaluate(direction).max(0.0),
            self.sh_b.evaluate(direction).max(0.0),
        )
    }

    /// Add a directional radiance sample to the probe
    pub fn add_sample(&mut self, direction: Vec3, color: Vec3) {
        self.sh_r.add(&SH1::project(direction, color.x));
        self.sh_g.add(&SH1::project(direction, color.y));
        self.sh_b.add(&SH1::project(direction, color.z));
    }

    /// Normalize the probe by sample count
    pub fn normalize(&mut self, sample_count: u32) {
        if sample_count > 0 {
            let inv = 1.0 / sample_count as f32;
            self.sh_r.scale(inv);
            self.sh_g.scale(inv);
            self.sh_b.scale(inv);
        }
    }
}

/// A 3D grid of irradiance probes
pub struct IrradianceGrid {
    /// Probes in flat array (x + y*sx + z*sx*sy)
    pub probes: Vec<IrradianceProbe>,
    /// Grid dimensions
    pub grid_size: [u32; 3],
    /// World-space min bounds
    pub bounds_min: Vec3,
    /// World-space max bounds
    pub bounds_max: Vec3,
}

impl IrradianceGrid {
    /// Create an empty grid with evenly spaced probes
    pub fn new(grid_size: [u32; 3], bounds_min: Vec3, bounds_max: Vec3) -> Self {
        let total = (grid_size[0] * grid_size[1] * grid_size[2]) as usize;
        let mut probes = Vec::with_capacity(total);

        let size = bounds_max - bounds_min;
        let step = Vec3::new(
            size.x / grid_size[0].max(1) as f32,
            size.y / grid_size[1].max(1) as f32,
            size.z / grid_size[2].max(1) as f32,
        );

        for z in 0..grid_size[2] {
            for y in 0..grid_size[1] {
                for x in 0..grid_size[0] {
                    let pos = bounds_min
                        + Vec3::new(
                            (x as f32 + 0.5) * step.x,
                            (y as f32 + 0.5) * step.y,
                            (z as f32 + 0.5) * step.z,
                        );
                    probes.push(IrradianceProbe {
                        position: pos,
                        ..Default::default()
                    });
                }
            }
        }

        IrradianceGrid {
            probes,
            grid_size,
            bounds_min,
            bounds_max,
        }
    }

    /// Sample irradiance at a world position with trilinear interpolation
    pub fn sample(&self, position: Vec3, normal: Vec3) -> Vec3 {
        let size = self.bounds_max - self.bounds_min;
        let t = (position - self.bounds_min) / size;
        let fx = t.x * self.grid_size[0] as f32 - 0.5;
        let fy = t.y * self.grid_size[1] as f32 - 0.5;
        let fz = t.z * self.grid_size[2] as f32 - 0.5;

        let x0 = (fx.floor() as i32).clamp(0, self.grid_size[0] as i32 - 1) as u32;
        let y0 = (fy.floor() as i32).clamp(0, self.grid_size[1] as i32 - 1) as u32;
        let z0 = (fz.floor() as i32).clamp(0, self.grid_size[2] as i32 - 1) as u32;
        let x1 = (x0 + 1).min(self.grid_size[0] - 1);
        let y1 = (y0 + 1).min(self.grid_size[1] - 1);
        let z1 = (z0 + 1).min(self.grid_size[2] - 1);

        let tx = (fx - fx.floor()).clamp(0.0, 1.0);
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);
        let tz = (fz - fz.floor()).clamp(0.0, 1.0);

        // Trilinear interpolation of 8 surrounding probes
        let eval = |x: u32, y: u32, z: u32| -> Vec3 {
            let idx =
                (x + y * self.grid_size[0] + z * self.grid_size[0] * self.grid_size[1]) as usize;
            if idx < self.probes.len() {
                self.probes[idx].evaluate(normal)
            } else {
                Vec3::ZERO
            }
        };

        let c000 = eval(x0, y0, z0);
        let c100 = eval(x1, y0, z0);
        let c010 = eval(x0, y1, z0);
        let c110 = eval(x1, y1, z0);
        let c001 = eval(x0, y0, z1);
        let c101 = eval(x1, y0, z1);
        let c011 = eval(x0, y1, z1);
        let c111 = eval(x1, y1, z1);

        let c00 = c000 + (c100 - c000) * tx;
        let c10 = c010 + (c110 - c010) * tx;
        let c01 = c001 + (c101 - c001) * tx;
        let c11 = c011 + (c111 - c011) * tx;

        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;

        c0 + (c1 - c0) * tz
    }

    /// Get a probe by grid coordinates
    pub fn get_probe(&self, x: u32, y: u32, z: u32) -> Option<&IrradianceProbe> {
        let idx = (x + y * self.grid_size[0] + z * self.grid_size[0] * self.grid_size[1]) as usize;
        self.probes.get(idx)
    }

    /// Get a mutable probe by grid coordinates
    pub fn get_probe_mut(&mut self, x: u32, y: u32, z: u32) -> Option<&mut IrradianceProbe> {
        let idx = (x + y * self.grid_size[0] + z * self.grid_size[0] * self.grid_size[1]) as usize;
        self.probes.get_mut(idx)
    }

    /// Total number of probes
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Memory size in bytes
    pub fn memory_bytes(&self) -> usize {
        self.probes.len() * std::mem::size_of::<IrradianceProbe>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh1_project_evaluate() {
        let dir = Vec3::Y;
        let sh = SH1::project(dir, 1.0);

        // Evaluating in the same direction should give a positive value
        let val = sh.evaluate(dir);
        assert!(
            val > 0.0,
            "SH eval in projected dir should be positive, got {}",
            val
        );

        // Evaluating in opposite direction should be lower
        let val_neg = sh.evaluate(Vec3::NEG_Y);
        assert!(val > val_neg);
    }

    #[test]
    fn test_probe_add_sample() {
        let mut probe = IrradianceProbe::default();
        probe.add_sample(Vec3::Y, Vec3::new(1.0, 0.5, 0.0));

        let irr = probe.evaluate(Vec3::Y);
        assert!(irr.x > 0.0);
        assert!(irr.y > 0.0);
    }

    #[test]
    fn test_probe_normalize() {
        let mut probe = IrradianceProbe::default();
        probe.add_sample(Vec3::Y, Vec3::splat(2.0));
        probe.add_sample(Vec3::Y, Vec3::splat(4.0));
        probe.normalize(2);

        // After normalizing by 2, effective sample is (2+4)/2 = 3
        let irr = probe.evaluate(Vec3::Y);
        assert!(irr.x > 0.0);
    }

    #[test]
    fn test_grid_new() {
        let grid = IrradianceGrid::new([4, 4, 4], Vec3::splat(-2.0), Vec3::splat(2.0));

        assert_eq!(grid.probe_count(), 64);
        assert!(grid.memory_bytes() > 0);
    }

    #[test]
    fn test_grid_probe_positions() {
        let grid = IrradianceGrid::new([2, 2, 2], Vec3::ZERO, Vec3::splat(4.0));

        // First probe should be near (1, 1, 1) (center of first cell)
        let p = grid.get_probe(0, 0, 0).unwrap();
        assert!((p.position - Vec3::splat(1.0)).length() < 0.01);

        // Last probe should be near (3, 3, 3)
        let p = grid.get_probe(1, 1, 1).unwrap();
        assert!((p.position - Vec3::splat(3.0)).length() < 0.01);
    }

    #[test]
    fn test_grid_sample_uniform() {
        let mut grid = IrradianceGrid::new([2, 2, 2], Vec3::ZERO, Vec3::splat(4.0));

        // Set all probes to constant red light from above
        for probe in &mut grid.probes {
            probe.add_sample(Vec3::Y, Vec3::new(1.0, 0.0, 0.0));
        }

        // Sample anywhere should give red-ish light when looking up
        let irr = grid.sample(Vec3::splat(2.0), Vec3::Y);
        assert!(irr.x > 0.0, "Should have red light");
    }

    #[test]
    fn test_grid_sample_interpolated() {
        let mut grid = IrradianceGrid::new([2, 1, 1], Vec3::ZERO, Vec3::new(4.0, 1.0, 1.0));

        // Left probe: red, right probe: blue
        grid.probes[0].add_sample(Vec3::Y, Vec3::new(2.0, 0.0, 0.0));
        grid.probes[1].add_sample(Vec3::Y, Vec3::new(0.0, 0.0, 2.0));

        // Sample at midpoint should blend
        let mid = grid.sample(Vec3::new(2.0, 0.5, 0.5), Vec3::Y);
        assert!(mid.x > 0.0, "Should have some red");
        assert!(mid.z > 0.0, "Should have some blue");
    }
}
