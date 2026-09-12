//! Palette-based color primitives
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Sample a color from an ordered palette by parameter `t` in `[0, 1]`
///
/// Linearly interpolates between adjacent palette entries. Empty palette
/// returns `Vec3::ZERO`; single-entry palette returns that entry.
#[inline]
#[must_use]
pub fn palette_gradient(t: f32, palette: &[Vec3]) -> Vec3 {
    if palette.is_empty() {
        return Vec3::ZERO;
    }
    if palette.len() == 1 {
        return palette[0];
    }
    let n = palette.len();
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * (n as f32 - 1.0);
    let idx = (scaled.floor() as usize).min(n - 1);
    let next = (idx + 1).min(n - 1);
    let frac = (scaled - idx as f32).clamp(0.0, 1.0);
    palette[idx].lerp(palette[next], frac)
}

/// Three-anchor palette blend driven by sun altitude
///
/// `sun_altitude` in `[-1, 1]`: -1 = deep below horizon (night),
/// 0 = at horizon (dawn/dusk), 1 = zenith (noon). Below horizon blends
/// `night -> dawn_dusk`; above horizon blends `dawn_dusk -> noon`.
#[inline]
#[must_use]
pub fn time_of_day(sun_altitude: f32, night: Vec3, dawn_dusk: Vec3, noon: Vec3) -> Vec3 {
    let alt = sun_altitude.clamp(-1.0, 1.0);
    if alt <= 0.0 {
        night.lerp(dawn_dusk, (alt + 1.0).clamp(0.0, 1.0))
    } else {
        dawn_dusk.lerp(noon, alt.clamp(0.0, 1.0))
    }
}

/// Four-season palette blend on a normalized year cycle
///
/// `t` in `[0, 1]`: 0 = spring, 0.25 = summer, 0.5 = autumn, 0.75 = winter,
/// 1.0 = spring (loops). Between anchors the color is linearly interpolated.
#[inline]
#[must_use]
pub fn season_palette(t: f32, spring: Vec3, summer: Vec3, autumn: Vec3, winter: Vec3) -> Vec3 {
    let palette = [spring, summer, autumn, winter, spring];
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * 4.0;
    let idx = (scaled.floor() as usize).min(4);
    let next = (idx + 1).min(4);
    let frac = (scaled - idx as f32).clamp(0.0, 1.0);
    palette[idx].lerp(palette[next], frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_gradient_start_is_first() {
        let p = [Vec3::X, Vec3::Y, Vec3::Z];
        let c = palette_gradient(0.0, &p);
        assert!((c - Vec3::X).length() < 1e-4);
    }

    #[test]
    fn palette_gradient_end_is_last() {
        let p = [Vec3::X, Vec3::Y, Vec3::Z];
        let c = palette_gradient(1.0, &p);
        assert!((c - Vec3::Z).length() < 1e-4);
    }

    #[test]
    fn palette_gradient_empty_is_zero() {
        assert_eq!(palette_gradient(0.5, &[]), Vec3::ZERO);
    }

    #[test]
    fn palette_gradient_single_returns_it() {
        let p = [Vec3::new(0.3, 0.4, 0.5)];
        assert_eq!(palette_gradient(0.7, &p), p[0]);
    }

    #[test]
    fn time_of_day_night_returns_night() {
        let night = Vec3::new(0.05, 0.05, 0.15);
        let dusk = Vec3::new(0.9, 0.5, 0.3);
        let noon = Vec3::new(0.7, 0.8, 1.0);
        let c = time_of_day(-1.0, night, dusk, noon);
        assert!((c - night).length() < 1e-4);
    }

    #[test]
    fn time_of_day_zenith_returns_noon() {
        let night = Vec3::new(0.05, 0.05, 0.15);
        let dusk = Vec3::new(0.9, 0.5, 0.3);
        let noon = Vec3::new(0.7, 0.8, 1.0);
        let c = time_of_day(1.0, night, dusk, noon);
        assert!((c - noon).length() < 1e-4);
    }

    #[test]
    fn time_of_day_horizon_returns_dusk() {
        let night = Vec3::new(0.05, 0.05, 0.15);
        let dusk = Vec3::new(0.9, 0.5, 0.3);
        let noon = Vec3::new(0.7, 0.8, 1.0);
        let c = time_of_day(0.0, night, dusk, noon);
        assert!((c - dusk).length() < 1e-4);
    }

    #[test]
    fn season_palette_spring_at_zero() {
        let s = Vec3::new(0.9, 0.8, 0.7);
        let u = Vec3::new(0.3, 0.9, 0.2);
        let a = Vec3::new(0.9, 0.5, 0.1);
        let w = Vec3::new(0.8, 0.9, 1.0);
        let c = season_palette(0.0, s, u, a, w);
        assert!((c - s).length() < 1e-4);
    }

    #[test]
    fn season_palette_summer_at_quarter() {
        let s = Vec3::new(0.9, 0.8, 0.7);
        let u = Vec3::new(0.3, 0.9, 0.2);
        let a = Vec3::new(0.9, 0.5, 0.1);
        let w = Vec3::new(0.8, 0.9, 1.0);
        let c = season_palette(0.25, s, u, a, w);
        assert!((c - u).length() < 1e-4);
    }

    #[test]
    fn season_palette_loops_at_one() {
        let s = Vec3::new(0.9, 0.8, 0.7);
        let u = Vec3::new(0.3, 0.9, 0.2);
        let a = Vec3::new(0.9, 0.5, 0.1);
        let w = Vec3::new(0.8, 0.9, 1.0);
        let c = season_palette(1.0, s, u, a, w);
        assert!((c - s).length() < 1e-4);
    }
}
