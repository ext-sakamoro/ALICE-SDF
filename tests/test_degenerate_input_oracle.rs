//! Degenerate-input oracle: no public evaluator may panic on a value it can be handed.
//!
//! A distance field is evaluated at points that come from rays, grids and user data, and its
//! parameters come from files, scripts and sliders. A radius of `0`, a negative size, a `NaN`
//! slider, an `inf` coordinate or a `1e30` far point are all values a host can pass, and none of
//! them may take the process down (the result may be any number, including `NaN`).
//!
//! The sweeps are generated from the public API rather than hand-picked:
//!
//! * every `SdfNode` primitive constructor x the cartesian product of degenerate parameter values,
//! * every modifier / transform builder x the same product x three base shapes,
//! * every node of the grammar corpus x degenerate points,
//!
//! each through the tree evaluator, the gradient / normal path, the Lipschitz bound, ray marching
//! and the compiled bytecode evaluator. File importers get truncated, corrupted and random bytes
//! (the fuzz targets only cover `.asdf` decoding, bincode and evaluation).
//!
//! Documented panics are not in scope: `CompiledSdf::compile` (`# Panics`, with `try_compile` as
//! the fallible form) and `Interval::new` (a `debug_assert!` on bounds the caller computed).
//!
//! Author: Moroya Sakamoto
#![allow(clippy::too_many_lines)]

mod common;

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::mesh::{dual_contouring, sdf_to_mesh, DualContouringConfig, MarchingCubesConfig};
use alice_sdf::prelude::*;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Distinct panic messages -> (count, first offending case)
#[derive(Default)]
struct Panics(BTreeMap<String, (usize, String)>);

impl Panics {
    fn run(&mut self, label: impl FnOnce() -> String, f: impl FnOnce()) {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            let entry = self
                .0
                .entry(msg.chars().take(100).collect())
                .or_insert_with(|| (0, label()));
            entry.0 += 1;
        }
    }

    fn assert_none(&self, what: &str) {
        let report: Vec<String> = self
            .0
            .iter()
            .map(|(msg, (n, first))| format!("  x{n:<5} {msg}\n         first: {first}"))
            .collect();
        assert!(
            self.0.is_empty(),
            "{what}: {} distinct panic(s)\n{}",
            self.0.len(),
            report.join("\n")
        );
    }
}

const NAN: f32 = f32::NAN;
const INF: f32 = f32::INFINITY;

/// Values a host can plausibly hand over: zero, negative, `NaN`, `inf`, tiny, ordinary
const VALUES: [f32; 6] = [0.0, -1.0, NAN, INF, 1e-30, 1.0];
/// A smaller set for builders with five parameters (4^5 instead of 6^5)
const VALUES_MID: [f32; 5] = [0.0, -1.0, NAN, INF, 1.0];
const VALUES_WIDE: [f32; 4] = [0.0, -1.0, NAN, 1.0];

fn degenerate_points() -> Vec<Vec3> {
    let mut pts = vec![Vec3::ZERO, Vec3::new(0.3, -0.2, 0.7)];
    for &v in &[
        NAN,
        INF,
        -INF_F,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN_POSITIVE,
        -0.0,
    ] {
        pts.push(Vec3::splat(v));
        pts.push(Vec3::new(v, 0.0, 0.0));
        pts.push(Vec3::new(0.0, v, 1.0));
    }
    pts
}
const INF_F: f32 = f32::INFINITY;

/// Decode `code` into the digits of a mixed-radix number: one value per parameter
fn params(code: usize, n: usize, values: &[f32]) -> Vec<f32> {
    let mut c = code;
    (0..n)
        .map(|_| {
            let v = values[c % values.len()];
            c /= values.len();
            v
        })
        .collect()
}

const fn values_for(n: usize) -> &'static [f32] {
    match n {
        0..=2 => &VALUES,
        3 => &VALUES_MID,
        _ => &VALUES_WIDE,
    }
}

type PrimCtor = (&'static str, usize, Box<dyn Fn(&[f32]) -> SdfNode>);
type ModCtor = (&'static str, usize, Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>);

fn primitive_table() -> Vec<PrimCtor> {
    vec![
        (
            "sphere",
            1,
            Box::new(|a: &[f32]| SdfNode::sphere(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "box3d",
            3,
            Box::new(|a: &[f32]| SdfNode::box3d(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "box3d_half_extents",
            3,
            Box::new(|a: &[f32]| SdfNode::box3d_half_extents(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "cylinder",
            2,
            Box::new(|a: &[f32]| SdfNode::cylinder(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "torus",
            2,
            Box::new(|a: &[f32]| SdfNode::torus(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "plane",
            2,
            Box::new(|a: &[f32]| SdfNode::plane(Vec3::splat(a[0]), a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "capsule",
            3,
            Box::new(|a: &[f32]| SdfNode::capsule(Vec3::splat(a[0]), Vec3::splat(a[1]), a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "cone",
            2,
            Box::new(|a: &[f32]| SdfNode::cone(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "ellipsoid",
            3,
            Box::new(|a: &[f32]| SdfNode::ellipsoid(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rounded_cone",
            3,
            Box::new(|a: &[f32]| SdfNode::rounded_cone(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "pyramid",
            1,
            Box::new(|a: &[f32]| SdfNode::pyramid(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "octahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::octahedron(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "hex_prism",
            2,
            Box::new(|a: &[f32]| SdfNode::hex_prism(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "link",
            3,
            Box::new(|a: &[f32]| SdfNode::link(a[0], a[1], a[2])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "triangle",
            3,
            Box::new(|a: &[f32]| {
                SdfNode::triangle(Vec3::splat(a[0]), Vec3::splat(a[1]), Vec3::splat(a[2]))
            }) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "bezier",
            4,
            Box::new(|a: &[f32]| {
                SdfNode::bezier(
                    Vec3::splat(a[0]),
                    Vec3::splat(a[1]),
                    Vec3::splat(a[2]),
                    a[3],
                )
            }) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rounded_box",
            4,
            Box::new(|a: &[f32]| SdfNode::rounded_box(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "capped_cone",
            3,
            Box::new(|a: &[f32]| SdfNode::capped_cone(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "capped_torus",
            3,
            Box::new(|a: &[f32]| SdfNode::capped_torus(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rounded_cylinder",
            3,
            Box::new(|a: &[f32]| SdfNode::rounded_cylinder(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "triangular_prism",
            2,
            Box::new(|a: &[f32]| SdfNode::triangular_prism(a[0], a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "cut_sphere",
            2,
            Box::new(|a: &[f32]| SdfNode::cut_sphere(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "cut_hollow_sphere",
            3,
            Box::new(|a: &[f32]| SdfNode::cut_hollow_sphere(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "death_star",
            3,
            Box::new(|a: &[f32]| SdfNode::death_star(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "solid_angle",
            2,
            Box::new(|a: &[f32]| SdfNode::solid_angle(a[0], a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rhombus",
            4,
            Box::new(|a: &[f32]| SdfNode::rhombus(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "horseshoe",
            5,
            Box::new(|a: &[f32]| SdfNode::horseshoe(a[0], a[1], a[2], a[3], a[4]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "vesica",
            2,
            Box::new(|a: &[f32]| SdfNode::vesica(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "infinite_cylinder",
            1,
            Box::new(|a: &[f32]| SdfNode::infinite_cylinder(a[0]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "infinite_cone",
            1,
            Box::new(|a: &[f32]| SdfNode::infinite_cone(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "gyroid",
            2,
            Box::new(|a: &[f32]| SdfNode::gyroid(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "heart",
            1,
            Box::new(|a: &[f32]| SdfNode::heart(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "tube",
            3,
            Box::new(|a: &[f32]| SdfNode::tube(a[0], a[1], a[2])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "barrel",
            3,
            Box::new(|a: &[f32]| SdfNode::barrel(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "diamond",
            2,
            Box::new(|a: &[f32]| SdfNode::diamond(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "chamfered_cube",
            4,
            Box::new(|a: &[f32]| SdfNode::chamfered_cube(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "schwarz_p",
            2,
            Box::new(|a: &[f32]| SdfNode::schwarz_p(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "superellipsoid",
            5,
            Box::new(|a: &[f32]| SdfNode::superellipsoid(a[0], a[1], a[2], a[3], a[4]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rounded_x",
            3,
            Box::new(|a: &[f32]| SdfNode::rounded_x(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "pie",
            3,
            Box::new(|a: &[f32]| SdfNode::pie(a[0], a[1], a[2])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "trapezoid",
            4,
            Box::new(|a: &[f32]| SdfNode::trapezoid(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "parallelogram",
            4,
            Box::new(|a: &[f32]| SdfNode::parallelogram(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "tunnel",
            3,
            Box::new(|a: &[f32]| SdfNode::tunnel(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "uneven_capsule",
            4,
            Box::new(|a: &[f32]| SdfNode::uneven_capsule(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "egg",
            2,
            Box::new(|a: &[f32]| SdfNode::egg(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "arc_shape",
            4,
            Box::new(|a: &[f32]| SdfNode::arc_shape(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "moon",
            4,
            Box::new(|a: &[f32]| SdfNode::moon(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "cross_shape",
            4,
            Box::new(|a: &[f32]| SdfNode::cross_shape(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "blobby_cross",
            2,
            Box::new(|a: &[f32]| SdfNode::blobby_cross(a[0], a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "parabola_segment",
            3,
            Box::new(|a: &[f32]| SdfNode::parabola_segment(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "helix",
            4,
            Box::new(|a: &[f32]| SdfNode::helix(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "tetrahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::tetrahedron(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "dodecahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::dodecahedron(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "icosahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::icosahedron(a[0])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "truncated_octahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::truncated_octahedron(a[0]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "truncated_icosahedron",
            1,
            Box::new(|a: &[f32]| SdfNode::truncated_icosahedron(a[0]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "box_frame",
            2,
            Box::new(|a: &[f32]| SdfNode::box_frame(Vec3::splat(a[0]), a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "diamond_surface",
            2,
            Box::new(|a: &[f32]| SdfNode::diamond_surface(a[0], a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "neovius",
            2,
            Box::new(|a: &[f32]| SdfNode::neovius(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "lidinoid",
            2,
            Box::new(|a: &[f32]| SdfNode::lidinoid(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "iwp",
            2,
            Box::new(|a: &[f32]| SdfNode::iwp(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "frd",
            2,
            Box::new(|a: &[f32]| SdfNode::frd(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "fischer_koch_s",
            2,
            Box::new(|a: &[f32]| SdfNode::fischer_koch_s(a[0], a[1]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "pmy",
            2,
            Box::new(|a: &[f32]| SdfNode::pmy(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "circle_2d",
            2,
            Box::new(|a: &[f32]| SdfNode::circle_2d(a[0], a[1])) as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rect_2d",
            3,
            Box::new(|a: &[f32]| SdfNode::rect_2d(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "rounded_rect_2d",
            4,
            Box::new(|a: &[f32]| SdfNode::rounded_rect_2d(a[0], a[1], a[2], a[3]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
        (
            "annular_2d",
            3,
            Box::new(|a: &[f32]| SdfNode::annular_2d(a[0], a[1], a[2]))
                as Box<dyn Fn(&[f32]) -> SdfNode>,
        ),
    ]
}

fn modifier_table() -> Vec<ModCtor> {
    vec![
        (
            "twist",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.twist(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "bend",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.bend(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "repeat_infinite",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.repeat_infinite(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "repeat_finite",
            2,
            Box::new(|n: SdfNode, a: &[f32]| {
                n.repeat_finite([(a[0].abs().min(4.0)) as u32; 3], Vec3::splat(a[1]))
            }) as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "noise",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.noise(a[0], a[1], (a[2].abs().min(8.0)) as u32))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "round",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.round(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "onion",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.onion(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "elongate",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.elongate(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "mirror",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.mirror(a[0] > 0.0, a[1] > 0.0, a[2] > 0.0))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "revolution",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.revolution(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "extrude",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.extrude(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "taper",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.taper(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "displacement",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.displacement(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "sine_displacement",
            2,
            Box::new(|n: SdfNode, a: &[f32]| n.sine_displacement(a[0], a[1]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "polar_repeat",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.polar_repeat((a[0].abs().min(8.0)) as u32))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "shear",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.shear(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "animated",
            2,
            Box::new(|n: SdfNode, a: &[f32]| n.animated(a[0], a[1]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "with_material",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.with_material((a[0].abs().min(8.0)) as u32))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "surface_roughness",
            3,
            Box::new(|n: SdfNode, a: &[f32]| {
                n.surface_roughness(a[0], a[1], (a[2].abs().min(8.0)) as u32)
            }) as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "translate",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.translate(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "translate_vec",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.translate_vec(Vec3::splat(a[0])))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "rotate_euler",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.rotate_euler(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "scale",
            1,
            Box::new(|n: SdfNode, a: &[f32]| n.scale(a[0]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
        (
            "scale_xyz",
            3,
            Box::new(|n: SdfNode, a: &[f32]| n.scale_xyz(a[0], a[1], a[2]))
                as Box<dyn Fn(SdfNode, &[f32]) -> SdfNode>,
        ),
    ]
}

/// Evaluate `node` through every public evaluation path at `points`
fn exercise(p: &mut Panics, name: &str, node: &SdfNode, points: &[Vec3], heavy: bool) {
    for &pt in points {
        let n = node.clone();
        p.run(
            || format!("eval {name} at {pt:?}"),
            || {
                let _ = eval(&n, pt);
            },
        );
    }
    let probe = Vec3::new(0.3, -0.2, 0.7);
    let n = node.clone();
    p.run(
        || format!("eval_gradient {name}"),
        || {
            let _ = alice_sdf::eval::eval_gradient(&n, probe);
        },
    );
    let n = node.clone();
    p.run(
        || format!("eval_normal {name}"),
        || {
            let _ = alice_sdf::eval::eval_normal(&n, probe);
        },
    );
    if heavy {
        let n = node.clone();
        p.run(
            || format!("eval_lipschitz {name}"),
            || {
                let _ = alice_sdf::interval::eval_lipschitz(&n);
            },
        );
        let n = node.clone();
        p.run(
            || format!("raymarch {name}"),
            || {
                let _ = alice_sdf::raycast::raymarch(&n, Vec3::new(-3.0, 0.1, 0.2), Vec3::X, 10.0);
            },
        );
    }
    let n = node.clone();
    p.run(
        || format!("compiled {name}"),
        || {
            if let Ok(c) = CompiledSdf::try_compile(&n) {
                for &pt in points {
                    let _ = eval_compiled(&c, pt);
                }
            }
        },
    );
}

#[test]
fn no_corpus_node_panics_at_a_degenerate_point() {
    let mut p = Panics::default();
    let pts = degenerate_points();
    for (name, node) in common::corpus::corpus() {
        exercise(&mut p, name, &node, &pts, true);
    }
    p.assert_none("corpus node x degenerate point");
}

#[test]
fn no_primitive_panics_for_any_degenerate_parameter_combination() {
    let mut p = Panics::default();
    let pts = [
        Vec3::ZERO,
        Vec3::new(0.3, -0.2, 0.7),
        Vec3::splat(NAN),
        Vec3::splat(INF),
    ];
    for (name, n, ctor) in primitive_table() {
        let values = values_for(n);
        for code in 0..values.len().pow(u32::try_from(n).expect("small")) {
            let a = params(code, n, values);
            let mut built = None;
            p.run(
                || format!("construct {name}({a:?})"),
                || built = Some(ctor(&a)),
            );
            if let Some(node) = built {
                exercise(
                    &mut p,
                    &format!("{name}({a:?})"),
                    &node,
                    &pts,
                    code % 8 == 0,
                );
            }
        }
    }
    p.assert_none("primitive x degenerate parameters");
}

#[test]
fn no_modifier_or_transform_panics_for_any_degenerate_parameter_combination() {
    let mut p = Panics::default();
    let bases = [SdfNode::sphere(1.0), SdfNode::box3d(1.0, 0.6, 0.4)];
    let pts = [
        Vec3::ZERO,
        Vec3::new(0.3, -0.2, 0.7),
        Vec3::splat(NAN),
        Vec3::splat(INF),
        Vec3::splat(1e30),
    ];
    for (name, n, ctor) in modifier_table() {
        // three floats (translate, scale_xyz, shear, ...) would be 6^3 x bases: use fewer values
        let values: &[f32] = match n {
            1 => &VALUES,
            2 => &VALUES_MID,
            _ => &VALUES_WIDE,
        };
        for code in 0..values.len().pow(u32::try_from(n).expect("small")) {
            let a = params(code, n, values);
            for base in &bases {
                let mut built = None;
                p.run(
                    || format!("construct {name}({a:?})"),
                    || built = Some(ctor(base.clone(), &a)),
                );
                if let Some(node) = built {
                    exercise(
                        &mut p,
                        &format!("{name}({a:?})"),
                        &node,
                        &pts,
                        code % 8 == 0,
                    );
                }
            }
        }
    }
    p.assert_none("modifier / transform x degenerate parameters");
}

#[test]
fn noise_stays_finite_at_huge_finite_coordinates() {
    // `x.floor() as i32` saturates for |x| > 2^31, and the lattice index `+ 1` that follows must
    // wrap (release behaviour) instead of overflowing: a far point is a finite, well-defined value
    // (up to 1e18 so that the sphere's own `x^2` does not overflow f32)
    let node = SdfNode::sphere(1.0).noise(0.2, 3.0, 7);
    for &far in &[3.0e9_f32, -3.0e9, 2.2e9, 1.0e12, 1.0e15, 1.0e18, -1.0e18] {
        for pt in [
            Vec3::new(far, 0.5, 0.5),
            Vec3::new(0.5, far, 0.5),
            Vec3::new(0.5, 0.5, far),
            Vec3::splat(far),
        ] {
            let v = eval(&node, pt);
            assert!(v.is_finite(), "noise at {pt:?} = {v}");
        }
    }
}

#[test]
fn degenerate_heightmaps_never_panic() {
    // width / height come from the file or script that built the node: zero, one, and a data
    // buffer shorter than `width * height` must be harmless
    let mut p = Panics::default();
    let sizes = [0_u32, 1, 2, 4];
    let pts = [
        Vec3::ZERO,
        Vec3::new(0.3, -0.2, 0.7),
        Vec3::new(2.0, 0.1, 0.1),
        Vec3::splat(NAN),
        Vec3::splat(INF),
    ];
    for &w in &sizes {
        for &h in &sizes {
            for len in [
                0_usize,
                1,
                (w * h).saturating_sub(1) as usize,
                (w * h) as usize,
            ] {
                for &(amp, scale) in &[(1.0_f32, 1.0_f32), (NAN, 1.0), (1.0, INF), (0.0, 0.0)] {
                    let node = SdfNode::sphere(1.0).heightmap_displacement(
                        vec![0.5; len],
                        w,
                        h,
                        amp,
                        scale,
                    );
                    for &pt in &pts {
                        let n = node.clone();
                        p.run(
                            || {
                                format!(
                                    "heightmap {w}x{h} len {len} amp {amp} scale {scale} at {pt:?}"
                                )
                            },
                            || {
                                let _ = eval(&n, pt);
                            },
                        );
                    }
                }
            }
        }
    }
    p.assert_none("heightmap displacement");
}

#[test]
fn the_ray_step_budget_stays_bounded_for_a_degenerate_lipschitz_bound() {
    use alice_sdf::raycast::RaymarchConfig;
    // a TPMS (L = 7) keeps its documented budget growth ...
    assert_eq!(RaymarchConfig::default().with_bound(7.0).max_steps, 1792);
    // ... but the bound of a degenerate field (scale 0 -> 1e6) must not ask for 256 * 1e6 steps
    let degenerate = RaymarchConfig::default().with_bound(1.0e6);
    assert!(
        degenerate.max_steps <= 65_536,
        "budget {} steps",
        degenerate.max_steps
    );
    assert!(
        degenerate.max_steps >= 1792,
        "the cap must not shrink the TPMS budget"
    );
    // and a ray through such a node returns (a miss) in bounded time instead of freezing the caller
    for node in [
        SdfNode::sphere(1.0).scale_xyz(0.0, 1.0, 1.0),
        SdfNode::sphere(1.0).scale_xyz(1.0, 0.0, 1.0),
    ] {
        let started = std::time::Instant::now();
        let _ = alice_sdf::raycast::raymarch(&node, Vec3::new(-3.0, 0.1, 0.2), Vec3::X, 10.0);
        assert!(
            started.elapsed().as_secs() < 5,
            "raymarch took {:?}",
            started.elapsed()
        );
    }
}

#[test]
fn public_primitive_functions_survive_degenerate_clamp_bounds() {
    // these are public but not reached by any `SdfNode` variant, so the `SdfNode` sweeps cannot see
    // them: a negative / NaN size is a clamp bound (`f32::clamp` panics for `min > max` or NaN)
    use alice_sdf::primitives::{
        sdf_capsule_horizontal, sdf_capsule_vertical, sdf_regular_polygon,
    };
    let mut p = Panics::default();
    let pts = [
        Vec3::ZERO,
        Vec3::new(0.3, -0.2, 0.7),
        Vec3::new(3.0, -4.0, 5.0),
    ];
    for &size in &[0.0_f32, -1.0, NAN, INF, 1.0] {
        for &r in &[0.0_f32, -1.0, NAN, 1.0] {
            for &pt in &pts {
                p.run(
                    || format!("sdf_capsule_vertical({pt:?}, {size}, {r})"),
                    || {
                        let _ = sdf_capsule_vertical(pt, size, r);
                    },
                );
                p.run(
                    || format!("sdf_capsule_horizontal({pt:?}, {size}, {r})"),
                    || {
                        let _ = sdf_capsule_horizontal(pt, size, r);
                    },
                );
                for &sides in &[0.0_f32, 3.0, NAN, -4.0] {
                    p.run(
                        || format!("sdf_regular_polygon({pt:?}, {size}, {sides}, {r})"),
                        || {
                            let _ = sdf_regular_polygon(pt, size, sides, r);
                        },
                    );
                }
            }
        }
    }
    p.assert_none("public primitive functions");
}

#[test]
fn bilinear_sample_survives_degenerate_maps() {
    use alice_sdf::modifiers::bilinear_sample;
    let mut p = Panics::default();
    for &(w, h) in &[(0_u32, 0_u32), (0, 4), (4, 0), (1, 1), (2, 2), (4, 4)] {
        for len in [0_usize, 1, 3, 16] {
            for &(u, v) in &[
                (0.0_f32, 0.0_f32),
                (1.5, 2.5),
                (NAN, 0.0),
                (INF, -INF),
                (-1.0, 100.0),
            ] {
                let map = vec![0.25_f32; len];
                p.run(
                    || format!("bilinear_sample {w}x{h} len {len} at ({u}, {v})"),
                    || {
                        let _ = bilinear_sample(&map, w, h, u, v);
                    },
                );
            }
        }
    }
    p.assert_none("bilinear_sample");
}

/// Rebuild a GLB after `edit` has rewritten its JSON chunk (the BIN chunk is kept as is)
fn edit_glb_json(glb: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let json_len = u32::from_le_bytes(glb[12..16].try_into().expect("len")) as usize;
    let json_end = 20 + json_len;
    let mut root: serde_json::Value = serde_json::from_slice(&glb[20..json_end]).expect("json");
    edit(&mut root);
    let mut json = serde_json::to_vec(&root).expect("json out");
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let bin = &glb[json_end..];
    let total = 12 + 8 + json.len() + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&glb[0..8]);
    out.extend_from_slice(&u32::try_from(total).expect("total").to_le_bytes());
    out.extend_from_slice(&u32::try_from(json.len()).expect("json").to_le_bytes());
    out.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&json);
    out.extend_from_slice(bin);
    out
}

#[test]
fn a_glb_with_inconsistent_references_is_an_error_not_a_panic() {
    use alice_sdf::io::{export_glb_bytes, import_glb_bytes, GltfConfig};
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 6,
            ..Default::default()
        },
    );
    let glb = export_glb_bytes(&mesh, &GltfConfig::default(), None).expect("export");
    assert!(
        import_glb_bytes(&glb).is_ok(),
        "the unmodified file must import"
    );

    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let accessor = |root: &mut serde_json::Value, i: usize| root["accessors"][i].clone();
    let _ = accessor;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "NORMAL accessor shorter than POSITION",
            Box::new(|r| {
                let n = r["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
                    .as_u64()
                    .expect("NORMAL");
                r["accessors"][n as usize]["count"] = 1.into();
            }),
        ),
        (
            "TEXCOORD_0 accessor shorter than POSITION",
            Box::new(|r| {
                let attrs = r["meshes"][0]["primitives"][0]["attributes"].clone();
                if let Some(uv) = attrs["TEXCOORD_0"].as_u64() {
                    r["accessors"][uv as usize]["count"] = 1.into();
                } else {
                    // no UVs in this export: shorten NORMAL instead so the case still exercises the check
                    let n = attrs["NORMAL"].as_u64().expect("NORMAL");
                    r["accessors"][n as usize]["count"] = 2.into();
                }
            }),
        ),
        (
            "indices past the last vertex (POSITION truncated)",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["count"] = 3.into();
                // keep NORMAL consistent so the index check is what rejects the file
                if let Some(n) = r["meshes"][0]["primitives"][0]["attributes"]["NORMAL"].as_u64() {
                    r["accessors"][n as usize]["count"] = 3.into();
                }
            }),
        ),
        (
            "POSITION accessor index out of range",
            Box::new(|r| {
                r["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = 99.into();
            }),
        ),
        (
            "indices accessor index out of range",
            Box::new(|r| {
                r["meshes"][0]["primitives"][0]["indices"] = 99.into();
            }),
        ),
        (
            "bufferView index out of range",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["bufferView"] = 99.into();
            }),
        ),
        (
            "count so large that count * 12 overflows",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["count"] = u64::MAX.into();
            }),
        ),
        (
            "byteOffset past the end of the BIN chunk",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["byteOffset"] = (u64::MAX - 4).into();
            }),
        ),
        (
            "huge byteStride with a large count",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                let bv = r["accessors"][p as usize]["bufferView"]
                    .as_u64()
                    .unwrap_or(0);
                r["bufferViews"][bv as usize]["byteStride"] = (1u64 << 40).into();
                r["accessors"][p as usize]["count"] = 1_000_000.into();
            }),
        ),
        (
            "byteOffset past the end of the BIN chunk without overflowing",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["byteOffset"] = 1_000_000.into();
            }),
        ),
        (
            "POSITION count past the end of the BIN chunk without overflowing",
            Box::new(|r| {
                let p = r["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
                    .as_u64()
                    .expect("POSITION");
                r["accessors"][p as usize]["count"] = 1_000_000_000_u64.into();
            }),
        ),
        (
            "index count past the end of the BIN chunk",
            Box::new(|r| {
                let i = r["meshes"][0]["primitives"][0]["indices"]
                    .as_u64()
                    .expect("indices");
                r["accessors"][i as usize]["count"] = 1_000_000_000_u64.into();
            }),
        ),
    ];
    let mut p = Panics::default();
    for (label, edit) in cases {
        let bytes = edit_glb_json(&glb, edit);
        let mut result = None;
        p.run(
            || label.to_string(),
            || result = Some(import_glb_bytes(&bytes)),
        );
        if let Some(r) = result {
            assert!(r.is_err(), "{label}: the inconsistent file was accepted");
        }
    }
    p.assert_none("inconsistent GLB references");
}

#[test]
fn interval_evaluation_stays_a_sound_enclosure_for_degenerate_scale_factors() {
    // `eval_interval` must return an ordered interval that contains the field wherever the field is
    // finite. A scale of 0 / NaN / inf makes the field NaN or constant, and "nothing is known"
    // (the entire line) is the only honest enclosure: dividing by zero must not build an inverted
    // interval (`lo > hi` fires the `Interval::new` debug assertion and, in release, is an empty
    // interval that wrongly claims `hi < 0` = "entirely inside").
    use alice_sdf::interval::{eval_interval, Vec3Interval};
    let factors = [0.0_f32, -0.0, NAN, INF, -INF, 1e-30, -1.0, 1e30, 0.5];
    let boxes = [
        (Vec3::splat(-1.0), Vec3::splat(1.0)),
        (Vec3::new(0.2, 0.2, 0.2), Vec3::new(0.4, 0.5, 0.6)),
        (Vec3::new(1.5, -0.5, 0.0), Vec3::new(2.5, 0.5, 1.0)),
    ];
    let mut p = Panics::default();
    for &f in &factors {
        let nodes = [
            ("scale", SdfNode::sphere(1.0).scale(f)),
            (
                "scale_xyz(f,1,1)",
                SdfNode::sphere(1.0).scale_xyz(f, 1.0, 1.0),
            ),
            (
                "scale_xyz(1,f,1)",
                SdfNode::sphere(1.0).scale_xyz(1.0, f, 1.0),
            ),
            (
                "scale_xyz(1,1,f)",
                SdfNode::sphere(1.0).scale_xyz(1.0, 1.0, f),
            ),
            ("scale_xyz(f,f,f)", SdfNode::sphere(1.0).scale_xyz(f, f, f)),
            (
                "scale_xyz(f,-f,1)",
                SdfNode::box3d(1.0, 0.5, 0.5).scale_xyz(f, -f, 1.0),
            ),
        ];
        for (name, node) in &nodes {
            for &(lo, hi) in &boxes {
                let mut got = None;
                p.run(
                    || format!("{name} factor {f} box [{lo:?}, {hi:?}]"),
                    || {
                        got = Some(eval_interval(node, Vec3Interval::from_bounds(lo, hi)));
                    },
                );
                let Some(iv) = got else { continue };
                assert!(
                    !iv.lo.is_nan() && !iv.hi.is_nan() && iv.lo <= iv.hi,
                    "{name} factor {f} box [{lo:?}, {hi:?}]: not an ordered interval [{}, {}]",
                    iv.lo,
                    iv.hi
                );
                // a zero / NaN / infinite factor leaves nothing to bound: the entire line
                if f == 0.0 || !f.is_finite() {
                    assert!(
                        iv.lo == f32::NEG_INFINITY && iv.hi == f32::INFINITY,
                        "{name} factor {f}: want the unbounded interval, got [{}, {}]",
                        iv.lo,
                        iv.hi
                    );
                }
                // soundness: every finite field value inside the box lies in the interval
                for ix in 0..4 {
                    for iy in 0..4 {
                        for iz in 0..4 {
                            let t = |i: i32| i as f32 / 3.0;
                            let pt = lo + (hi - lo) * Vec3::new(t(ix), t(iy), t(iz));
                            let v = eval(node, pt);
                            if v.is_finite() {
                                let tol = 1e-3 * (1.0 + v.abs());
                                assert!(
                                    iv.lo - tol <= v && v <= iv.hi + tol,
                                    "{name} factor {f}: field {v} at {pt:?} escapes [{}, {}]",
                                    iv.lo,
                                    iv.hi
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    p.assert_none("eval_interval on degenerate scales");
}

#[test]
fn compiling_never_panics_through_try_compile() {
    // `compile` documents a panic for unsupported primitives; `try_compile` is the fallible form
    let mut p = Panics::default();
    for (name, node) in common::corpus::corpus() {
        let n = node.clone();
        p.run(
            || format!("try_compile {name}"),
            || {
                let _ = CompiledSdf::try_compile(&n);
            },
        );
    }
    let unsupported = SdfNode::triangle(Vec3::ZERO, Vec3::X, Vec3::Y);
    p.run(
        || "try_compile triangle".to_string(),
        || {
            assert!(
                CompiledSdf::try_compile(&unsupported).is_err(),
                "triangle has no bytecode form"
            );
        },
    );
    p.assert_none("try_compile");
}

#[test]
fn meshing_and_ray_marching_survive_degenerate_arguments() {
    let mut p = Panics::default();
    let sphere = SdfNode::sphere(1.0);
    let boxes = [
        (Vec3::splat(-2.0), Vec3::splat(2.0)),
        (Vec3::splat(2.0), Vec3::splat(-2.0)),
        (Vec3::ZERO, Vec3::ZERO),
        (Vec3::splat(NAN), Vec3::splat(NAN)),
        (Vec3::splat(-INF), Vec3::splat(INF)),
        (Vec3::splat(-1e30), Vec3::splat(1e30)),
    ];
    for res in [0_usize, 1, 2, 3] {
        for (lo, hi) in boxes {
            let n = sphere.clone();
            let cfg = MarchingCubesConfig {
                resolution: res,
                ..Default::default()
            };
            p.run(
                || format!("sdf_to_mesh res={res} [{lo:?}, {hi:?}]"),
                || {
                    let _ = sdf_to_mesh(&n, lo, hi, &cfg);
                },
            );
            let n = sphere.clone();
            let cfg = DualContouringConfig {
                resolution: res,
                ..Default::default()
            };
            p.run(
                || format!("dual_contouring res={res} [{lo:?}, {hi:?}]"),
                || {
                    let _ = dual_contouring(&n, lo, hi, &cfg);
                },
            );
        }
    }
    for dir in [Vec3::ZERO, Vec3::splat(NAN), Vec3::splat(INF), Vec3::X] {
        for max_distance in [0.0_f32, -1.0, NAN, INF, 1e30] {
            for origin in [
                Vec3::new(-3.0, 0.0, 0.0),
                Vec3::splat(NAN),
                Vec3::splat(INF),
            ] {
                let n = sphere.clone();
                p.run(
                    || format!("raymarch dir {dir:?} max {max_distance} origin {origin:?}"),
                    || {
                        let _ = alice_sdf::raycast::raymarch(&n, origin, dir, max_distance);
                    },
                );
            }
        }
    }
    p.assert_none("meshing / ray marching");
}

#[test]
fn file_importers_never_panic_on_truncated_or_corrupted_files() {
    use alice_sdf::io::*;
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 6,
            ..Default::default()
        },
    );
    let dir = std::env::temp_dir().join(format!("sdf_degenerate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    type Importer = Box<dyn Fn(&std::path::Path)>;
    let mut cases: Vec<(&str, std::path::PathBuf, Importer)> = Vec::new();
    let tree = SdfTree::new(SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2));

    let path = dir.join("a.obj");
    export_obj(&mesh, &path, &ObjConfig::default(), None).expect("obj");
    cases.push(("obj", path, Box::new(|f| drop(import_obj(f)))));
    let path = dir.join("a.stl");
    export_stl(&mesh, &path).expect("stl");
    cases.push(("stl (binary)", path, Box::new(|f| drop(import_stl(f)))));
    let path = dir.join("b.stl");
    export_stl_ascii(&mesh, &path).expect("stl ascii");
    cases.push(("stl (ascii)", path, Box::new(|f| drop(import_stl(f)))));
    let path = dir.join("a.ply");
    export_ply(&mesh, &path, &PlyConfig::default()).expect("ply");
    cases.push(("ply", path, Box::new(|f| drop(import_ply(f)))));
    let path = dir.join("a.glb");
    export_glb(&mesh, &path, &GltfConfig::default(), None).expect("glb");
    cases.push(("glb", path, Box::new(|f| drop(import_glb(f)))));
    let path = dir.join("a.fbx");
    export_fbx(&mesh, &path, &FbxConfig::default(), None).expect("fbx");
    cases.push(("fbx", path, Box::new(|f| drop(import_fbx(f)))));
    let path = dir.join("a.abm");
    save_abm(&mesh, &path).expect("abm");
    cases.push(("abm", path, Box::new(|f| drop(load_abm(f)))));
    let path = dir.join("a.json");
    save_asdf_json(&tree, &path).expect("json");
    cases.push(("asdf json", path, Box::new(|f| drop(load_asdf_json(f)))));
    let path = dir.join("a.asdf");
    save_asdf(&tree, &path).expect("asdf");
    cases.push(("asdf", path, Box::new(|f| drop(load_asdf(f)))));

    let mut state: u32 = 0x1357_9BDF;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as usize
    };
    let mut p = Panics::default();
    for (format_name, path, import) in &cases {
        let original = std::fs::read(path).expect("read");
        let work = dir.join("mutated");
        let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
        // every prefix of the header, then a thinned set of prefixes
        let step = (original.len() / 60).max(1);
        for n in (0..original.len().min(64)).chain((64..original.len()).step_by(step)) {
            variants.push((
                format!("truncated to {n}/{}", original.len()),
                original[..n].to_vec(),
            ));
        }
        for t in 0..300 {
            let mut bytes = original.clone();
            for _ in 0..=(t % 4) {
                let k = next() % bytes.len();
                bytes[k] = (next() & 0xff) as u8;
            }
            variants.push((format!("corrupted #{t}"), bytes));
        }
        for t in 0..40 {
            variants.push((
                format!("random #{t}"),
                (0..next() % 400).map(|_| (next() & 0xff) as u8).collect(),
            ));
        }
        for (what, bytes) in variants {
            std::fs::write(&work, &bytes).expect("write");
            p.run(|| format!("{format_name}: {what}"), || import(&work));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    p.assert_none("file importers");
}

#[test]
fn surface_roughness_with_a_huge_octave_count_is_finite_and_prompt() {
    // `1u32 << i` overflows from octave 32, and a count of 4e9 would loop for minutes: octaves
    // beyond the f32 mantissa (amplitude 0.5^i < 2^-24) cannot change the value beyond rounding, so the
    // result for a huge count must match the result at 24
    let start = std::time::Instant::now();
    let pt = Vec3::new(1.2, 0.1, 0.1);
    let reference = eval(&SdfNode::sphere(1.0).surface_roughness(0.5, 2.0, 24), pt);
    assert!(reference.is_finite());
    for &octaves in &[25u32, 31, 32, 33, 64, 1_000_000, u32::MAX] {
        let v = eval(
            &SdfNode::sphere(1.0).surface_roughness(0.5, 2.0, octaves),
            pt,
        );
        // the 25th octave still adds up to 2 * 2^-24 of amplitude, so allow that much
        assert!(
            (v - reference).abs() <= 2.0e-6,
            "octaves={octaves}: {v} vs {reference}"
        );
    }
    assert!(start.elapsed().as_secs() < 10, "took {:?}", start.elapsed());
}
