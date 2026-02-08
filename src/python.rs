//! Python bindings for ALICE-SDF
//!
//! Provides Python API via PyO3.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, IntoPyArray};
use glam::{Vec2, Vec3};

use crate::types::{SdfNode, SdfTree};
use crate::eval::{eval, eval_batch_parallel};
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
use crate::io::{save, load};

/// Python-visible SdfNode wrapper
#[pyclass(name = "SdfNode")]
#[derive(Clone)]
pub struct PySdfNode {
    inner: SdfNode,
}

#[pymethods]
impl PySdfNode {
    /// Create a sphere
    #[staticmethod]
    fn sphere(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::sphere(radius),
        }
    }

    /// Create a box
    #[staticmethod]
    fn box3d(width: f32, height: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::box3d(width, height, depth),
        }
    }

    /// Create a cylinder
    #[staticmethod]
    fn cylinder(radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cylinder(radius, height),
        }
    }

    /// Create a torus
    #[staticmethod]
    fn torus(major_radius: f32, minor_radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::torus(major_radius, minor_radius),
        }
    }

    /// Create a capsule
    #[staticmethod]
    fn capsule(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::capsule(Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), radius),
        }
    }

    /// Create a plane
    #[staticmethod]
    fn plane(nx: f32, ny: f32, nz: f32, distance: f32) -> Self {
        PySdfNode {
            inner: SdfNode::Plane {
                normal: Vec3::new(nx, ny, nz).normalize(),
                distance,
            },
        }
    }

    /// Create a cone
    #[staticmethod]
    fn cone(radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cone(radius, height),
        }
    }

    /// Create an ellipsoid
    #[staticmethod]
    fn ellipsoid(rx: f32, ry: f32, rz: f32) -> Self {
        PySdfNode {
            inner: SdfNode::ellipsoid(rx, ry, rz),
        }
    }

    /// Create a rounded cone
    #[staticmethod]
    fn rounded_cone(r1: f32, r2: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rounded_cone(r1, r2, height),
        }
    }

    /// Create a pyramid
    #[staticmethod]
    fn pyramid(height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::pyramid(height),
        }
    }

    /// Create an octahedron
    #[staticmethod]
    fn octahedron(size: f32) -> Self {
        PySdfNode {
            inner: SdfNode::octahedron(size),
        }
    }

    /// Create a hexagonal prism
    #[staticmethod]
    fn hex_prism(hex_radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::hex_prism(hex_radius, height),
        }
    }

    /// Create a chain link
    #[staticmethod]
    fn link(length: f32, r1: f32, r2: f32) -> Self {
        PySdfNode {
            inner: SdfNode::link(length, r1, r2),
        }
    }

    /// Create a triangle from 3 vertices (9 floats)
    #[staticmethod]
    fn triangle(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, cx: f32, cy: f32, cz: f32) -> Self {
        PySdfNode {
            inner: SdfNode::triangle(Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), Vec3::new(cx, cy, cz)),
        }
    }

    /// Create a quadratic Bezier curve tube (9 floats + radius)
    #[staticmethod]
    fn bezier(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, cx: f32, cy: f32, cz: f32, radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::bezier(Vec3::new(ax, ay, az), Vec3::new(bx, by, bz), Vec3::new(cx, cy, cz), radius),
        }
    }

    /// Create a rounded box
    #[staticmethod]
    fn rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rounded_box(hx, hy, hz, round_radius),
        }
    }

    /// Create a capped cone
    #[staticmethod]
    fn capped_cone(height: f32, r1: f32, r2: f32) -> Self {
        PySdfNode {
            inner: SdfNode::capped_cone(height, r1, r2),
        }
    }

    /// Create a capped torus
    #[staticmethod]
    fn capped_torus(major_radius: f32, minor_radius: f32, cap_angle: f32) -> Self {
        PySdfNode {
            inner: SdfNode::capped_torus(major_radius, minor_radius, cap_angle),
        }
    }

    /// Create a rounded cylinder
    #[staticmethod]
    fn rounded_cylinder(radius: f32, round_radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rounded_cylinder(radius, round_radius, height),
        }
    }

    /// Create a triangular prism
    #[staticmethod]
    fn triangular_prism(width: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::triangular_prism(width, depth),
        }
    }

    /// Create a cut sphere
    #[staticmethod]
    fn cut_sphere(radius: f32, cut_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cut_sphere(radius, cut_height),
        }
    }

    /// Create a cut hollow sphere
    #[staticmethod]
    fn cut_hollow_sphere(radius: f32, cut_height: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cut_hollow_sphere(radius, cut_height, thickness),
        }
    }

    /// Create a death star shape
    #[staticmethod]
    fn death_star(ra: f32, rb: f32, d: f32) -> Self {
        PySdfNode {
            inner: SdfNode::death_star(ra, rb, d),
        }
    }

    /// Create a solid angle
    #[staticmethod]
    fn solid_angle(angle: f32, radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::solid_angle(angle, radius),
        }
    }

    /// Create a rhombus
    #[staticmethod]
    fn rhombus(la: f32, lb: f32, height: f32, round_radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rhombus(la, lb, height, round_radius),
        }
    }

    /// Create a horseshoe shape
    #[staticmethod]
    fn horseshoe(angle: f32, radius: f32, length: f32, width: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::horseshoe(angle, radius, length, width, thickness),
        }
    }

    /// Create a vesica shape
    #[staticmethod]
    fn vesica(radius: f32, dist: f32) -> Self {
        PySdfNode {
            inner: SdfNode::vesica(radius, dist),
        }
    }

    /// Create an infinite cylinder
    #[staticmethod]
    fn infinite_cylinder(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::infinite_cylinder(radius),
        }
    }

    /// Create an infinite cone
    #[staticmethod]
    fn infinite_cone(angle: f32) -> Self {
        PySdfNode {
            inner: SdfNode::infinite_cone(angle),
        }
    }

    /// Create a gyroid surface
    #[staticmethod]
    fn gyroid(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::gyroid(scale, thickness),
        }
    }

    /// Create a 3D heart
    #[staticmethod]
    fn heart(size: f32) -> Self {
        PySdfNode {
            inner: SdfNode::heart(size),
        }
    }

    /// Create a tube (hollow cylinder)
    #[staticmethod]
    fn tube(outer_radius: f32, thickness: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::tube(outer_radius, thickness, height),
        }
    }

    /// Create a barrel
    #[staticmethod]
    fn barrel(radius: f32, height: f32, bulge: f32) -> Self {
        PySdfNode {
            inner: SdfNode::barrel(radius, height, bulge),
        }
    }

    /// Create a diamond shape
    #[staticmethod]
    fn diamond_shape(radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::diamond(radius, height),
        }
    }

    /// Create a chamfered cube
    #[staticmethod]
    fn chamfered_cube(width: f32, height: f32, depth: f32, chamfer: f32) -> Self {
        PySdfNode {
            inner: SdfNode::chamfered_cube(width, height, depth, chamfer),
        }
    }

    /// Create a Schwarz P surface (TPMS)
    #[staticmethod]
    fn schwarz_p(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::schwarz_p(scale, thickness),
        }
    }

    /// Create a superellipsoid
    #[staticmethod]
    fn superellipsoid(width: f32, height: f32, depth: f32, e1: f32, e2: f32) -> Self {
        PySdfNode {
            inner: SdfNode::superellipsoid(width, height, depth, e1, e2),
        }
    }

    /// Create a rounded X shape
    #[staticmethod]
    fn rounded_x(width: f32, round_radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rounded_x(width, round_radius, height),
        }
    }

    /// Create a pie (sector) shape
    #[staticmethod]
    fn pie(angle: f32, radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::pie(angle, radius, height),
        }
    }

    /// Create a trapezoid
    #[staticmethod]
    fn trapezoid(r1: f32, r2: f32, trap_height: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::trapezoid(r1, r2, trap_height, depth),
        }
    }

    /// Create a parallelogram
    #[staticmethod]
    fn parallelogram(width: f32, para_height: f32, skew: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::parallelogram(width, para_height, skew, depth),
        }
    }

    /// Create a tunnel
    #[staticmethod]
    fn tunnel(width: f32, height_2d: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::tunnel(width, height_2d, depth),
        }
    }

    /// Create an uneven capsule
    #[staticmethod]
    fn uneven_capsule(r1: f32, r2: f32, cap_height: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::uneven_capsule(r1, r2, cap_height, depth),
        }
    }

    /// Create an egg shape
    #[staticmethod]
    fn egg(ra: f32, rb: f32) -> Self {
        PySdfNode {
            inner: SdfNode::egg(ra, rb),
        }
    }

    /// Create an arc shape (thick ring sector)
    #[staticmethod]
    fn arc_shape(aperture: f32, radius: f32, thickness: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::arc_shape(aperture, radius, thickness, height),
        }
    }

    /// Create a moon (crescent) shape
    #[staticmethod]
    fn moon(d: f32, ra: f32, rb: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::moon(d, ra, rb, height),
        }
    }

    /// Create a cross (plus) shape
    #[staticmethod]
    fn cross_shape(length: f32, thickness: f32, round_radius: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::cross_shape(length, thickness, round_radius, height),
        }
    }

    /// Create a blobby cross (organic) shape
    #[staticmethod]
    fn blobby_cross(size: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::blobby_cross(size, height),
        }
    }

    /// Create a parabola segment
    #[staticmethod]
    fn parabola_segment(width: f32, para_height: f32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::parabola_segment(width, para_height, depth),
        }
    }

    /// Create a regular N-sided polygon prism
    #[staticmethod]
    fn regular_polygon(radius: f32, n_sides: u32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::regular_polygon(radius, n_sides, height),
        }
    }

    /// Create a star polygon prism
    #[staticmethod]
    fn star_polygon(radius: f32, n_points: u32, m: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::star_polygon(radius, n_points, m, height),
        }
    }

    /// Create a staircase shape
    #[staticmethod]
    fn stairs_shape(step_width: f32, step_height: f32, n_steps: u32, depth: f32) -> Self {
        PySdfNode {
            inner: SdfNode::stairs(step_width, step_height, n_steps, depth),
        }
    }

    /// Create a helix (spiral tube)
    #[staticmethod]
    fn helix(major_r: f32, minor_r: f32, pitch: f32, height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::helix(major_r, minor_r, pitch, height),
        }
    }

    /// Create a regular tetrahedron
    #[staticmethod]
    fn tetrahedron(size: f32) -> Self {
        PySdfNode {
            inner: SdfNode::tetrahedron(size),
        }
    }

    /// Create a regular dodecahedron
    #[staticmethod]
    fn dodecahedron(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::dodecahedron(radius),
        }
    }

    /// Create a regular icosahedron
    #[staticmethod]
    fn icosahedron(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::icosahedron(radius),
        }
    }

    /// Create a truncated octahedron
    #[staticmethod]
    fn truncated_octahedron(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::truncated_octahedron(radius),
        }
    }

    /// Create a truncated icosahedron (soccer ball)
    #[staticmethod]
    fn truncated_icosahedron(radius: f32) -> Self {
        PySdfNode {
            inner: SdfNode::truncated_icosahedron(radius),
        }
    }

    /// Create a box frame (wireframe box)
    #[staticmethod]
    fn box_frame(hx: f32, hy: f32, hz: f32, edge: f32) -> Self {
        PySdfNode {
            inner: SdfNode::box_frame(Vec3::new(hx, hy, hz), edge),
        }
    }

    /// Create a diamond surface (TPMS)
    #[staticmethod]
    fn diamond_surface(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::diamond_surface(scale, thickness),
        }
    }

    /// Create a neovius surface (TPMS)
    #[staticmethod]
    fn neovius(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::neovius(scale, thickness),
        }
    }

    /// Create a lidinoid surface (TPMS)
    #[staticmethod]
    fn lidinoid(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::lidinoid(scale, thickness),
        }
    }

    /// Create an IWP surface (TPMS)
    #[staticmethod]
    fn iwp(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::iwp(scale, thickness),
        }
    }

    /// Create an FRD surface (TPMS)
    #[staticmethod]
    fn frd(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::frd(scale, thickness),
        }
    }

    /// Create a Fischer-Koch S surface (TPMS)
    #[staticmethod]
    fn fischer_koch_s(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::fischer_koch_s(scale, thickness),
        }
    }

    /// Create a PMY surface (TPMS)
    #[staticmethod]
    fn pmy(scale: f32, thickness: f32) -> Self {
        PySdfNode {
            inner: SdfNode::pmy(scale, thickness),
        }
    }

    /// Union with another shape
    fn union(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().union(other.inner.clone()),
        }
    }

    /// Intersection with another shape
    fn intersection(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().intersection(other.inner.clone()),
        }
    }

    /// Subtract another shape
    fn subtract(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().subtract(other.inner.clone()),
        }
    }

    /// Smooth union with another shape
    fn smooth_union(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_union(other.inner.clone(), k),
        }
    }

    /// Smooth intersection with another shape
    fn smooth_intersection(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_intersection(other.inner.clone(), k),
        }
    }

    /// Smooth subtraction
    fn smooth_subtract(&self, other: &PySdfNode, k: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().smooth_subtract(other.inner.clone(), k),
        }
    }

    /// Translate
    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().translate(x, y, z),
        }
    }

    /// Rotate by Euler angles (radians)
    fn rotate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().rotate_euler(x, y, z),
        }
    }

    /// Uniform scale
    fn scale(&self, factor: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().scale(factor),
        }
    }

    /// Twist around Y-axis
    fn twist(&self, strength: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().twist(strength),
        }
    }

    /// Bend
    fn bend(&self, curvature: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().bend(curvature),
        }
    }

    /// Infinite repetition
    fn repeat(&self, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().repeat_infinite(spacing_x, spacing_y, spacing_z),
        }
    }

    /// Perlin noise
    fn noise(&self, amplitude: f32, frequency: f32, seed: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().noise(amplitude, frequency, seed),
        }
    }

    /// Round edges
    fn round(&self, radius: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().round(radius),
        }
    }

    /// Shell (onion)
    fn onion(&self, thickness: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().onion(thickness),
        }
    }

    /// Mirror along specified axes
    fn mirror(&self, x: bool, y: bool, z: bool) -> Self {
        PySdfNode {
            inner: self.inner.clone().mirror(x, y, z),
        }
    }

    /// Octant mirror (48-fold symmetry: abs + sort x >= y >= z)
    fn octant_mirror(&self) -> Self {
        PySdfNode {
            inner: self.inner.clone().octant_mirror(),
        }
    }

    /// Elongate along axes
    fn elongate(&self, x: f32, y: f32, z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().elongate(x, y, z),
        }
    }

    /// Finite repetition
    fn repeat_finite(&self, count_x: u32, count_y: u32, count_z: u32, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().repeat_finite(
                [count_x, count_y, count_z],
                Vec3::new(spacing_x, spacing_y, spacing_z),
            ),
        }
    }

    /// Revolution around Y-axis
    fn revolution(&self, offset: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().revolution(offset),
        }
    }

    /// Extrude along Z-axis
    fn extrude(&self, height: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().extrude(height),
        }
    }

    /// Taper along Y-axis
    fn taper(&self, factor: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().taper(factor),
        }
    }

    /// Sin-based displacement
    fn displacement(&self, strength: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().displacement(strength),
        }
    }

    /// Polar repetition around Y-axis
    fn polar_repeat(&self, count: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().polar_repeat(count),
        }
    }

    /// Assign material ID
    fn with_material(&self, material_id: u32) -> Self {
        PySdfNode {
            inner: self.inner.clone().with_material(material_id),
        }
    }

    /// Sweep along a quadratic Bezier curve in XZ plane
    fn sweep_bezier(&self, p0x: f32, p0y: f32, p1x: f32, p1y: f32, p2x: f32, p2y: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().sweep_bezier(
                Vec2::new(p0x, p0y),
                Vec2::new(p1x, p1y),
                Vec2::new(p2x, p2y),
            ),
        }
    }

    /// XOR (symmetric difference) with another shape
    fn xor(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().xor(other.inner.clone()),
        }
    }

    /// Morph between two shapes (t=0: self, t=1: other)
    fn morph(&self, other: &PySdfNode, t: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().morph(other.inner.clone(), t),
        }
    }

    /// Columns union with another shape
    fn columns_union(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().columns_union(other.inner.clone(), r, n),
        }
    }

    /// Columns intersection with another shape
    fn columns_intersection(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().columns_intersection(other.inner.clone(), r, n),
        }
    }

    /// Columns subtraction of another shape
    fn columns_subtract(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().columns_subtract(other.inner.clone(), r, n),
        }
    }

    /// Pipe operation with another shape
    fn pipe(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().pipe(other.inner.clone(), r),
        }
    }

    /// Engrave another shape into this one
    fn engrave(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().engrave(other.inner.clone(), r),
        }
    }

    /// Cut a groove of another shape into this one
    fn groove(&self, other: &PySdfNode, ra: f32, rb: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().groove(other.inner.clone(), ra, rb),
        }
    }

    /// Add a tongue protrusion of another shape
    fn tongue(&self, other: &PySdfNode, ra: f32, rb: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().tongue(other.inner.clone(), ra, rb),
        }
    }

    /// Chamfer union with another shape
    fn chamfer_union(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().chamfer_union(other.inner.clone(), r),
        }
    }

    /// Chamfer intersection with another shape
    fn chamfer_intersection(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().chamfer_intersection(other.inner.clone(), r),
        }
    }

    /// Chamfer subtraction of another shape
    fn chamfer_subtract(&self, other: &PySdfNode, r: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().chamfer_subtract(other.inner.clone(), r),
        }
    }

    /// Stairs union with another shape
    fn stairs_union(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().stairs_union(other.inner.clone(), r, n),
        }
    }

    /// Stairs intersection with another shape
    fn stairs_intersection(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().stairs_intersection(other.inner.clone(), r, n),
        }
    }

    /// Stairs subtraction of another shape
    fn stairs_subtract(&self, other: &PySdfNode, r: f32, n: f32) -> Self {
        PySdfNode {
            inner: self.inner.clone().stairs_subtract(other.inner.clone(), r, n),
        }
    }

    /// Evaluate at a single point (GIL released)
    fn eval(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| eval(inner, Vec3::new(x, y, z)))
    }

    /// Compute analytic gradient at a point (GIL released)
    ///
    /// Returns (gx, gy, gz) gradient vector. For exact SDFs, |g| ≈ 1.
    fn gradient(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let inner = &self.inner;
        let g = py.allow_threads(|| crate::eval::eval_gradient(inner, Vec3::new(x, y, z)));
        (g.x, g.y, g.z)
    }

    /// Compute tight AABB for the SDF surface (GIL released, Rayon parallel)
    ///
    /// Returns ((min_x, min_y, min_z), (max_x, max_y, max_z)).
    #[pyo3(signature = (initial_half_size=10.0, bisection_iterations=20, coarse_subdivisions=8))]
    fn tight_aabb(
        &self,
        py: Python<'_>,
        initial_half_size: f32,
        bisection_iterations: u32,
        coarse_subdivisions: u32,
    ) -> ((f32, f32, f32), (f32, f32, f32)) {
        use crate::tight_aabb::{compute_tight_aabb_with_config, TightAabbConfig};
        let config = TightAabbConfig {
            initial_half_size,
            bisection_iterations,
            coarse_subdivisions,
        };
        let inner = &self.inner;
        let aabb = py.allow_threads(|| compute_tight_aabb_with_config(inner, &config));
        (
            (aabb.min.x, aabb.min.y, aabb.min.z),
            (aabb.max.x, aabb.max.y, aabb.max.z),
        )
    }

    /// Optimize CSG tree by removing identity nodes and merging transforms (GIL released)
    ///
    /// Returns a new optimized SdfNode.
    fn optimize(&self, py: Python<'_>) -> Self {
        let inner = &self.inner;
        let optimized = py.allow_threads(|| crate::optimize::optimize(inner));
        PySdfNode { inner: optimized }
    }

    /// Get node count
    fn node_count(&self) -> u32 {
        self.inner.node_count()
    }

    /// Generate GLSL shader code
    #[cfg(feature = "glsl")]
    fn to_glsl(&self) -> String {
        let shader = crate::compiled::GlslShader::transpile(&self.inner, crate::compiled::GlslTranspileMode::Hardcoded);
        shader.source
    }

    /// Generate HLSL shader code
    #[cfg(feature = "hlsl")]
    fn to_hlsl(&self) -> String {
        let shader = crate::compiled::HlslShader::transpile(&self.inner, crate::compiled::HlslTranspileMode::Hardcoded);
        shader.source
    }

    /// Generate WGSL shader code
    #[cfg(feature = "gpu")]
    fn to_wgsl(&self) -> String {
        let shader = crate::compiled::WgslShader::transpile(&self.inner, crate::compiled::TranspileMode::Hardcoded);
        shader.source
    }

    // --- Operator overloads for Pythonic DSL ---

    /// `a | b` → union
    fn __or__(&self, other: &PySdfNode) -> Self {
        self.union(other)
    }

    /// `a & b` → intersection
    fn __and__(&self, other: &PySdfNode) -> Self {
        self.intersection(other)
    }

    /// `a - b` → subtraction
    fn __sub__(&self, other: &PySdfNode) -> Self {
        self.subtract(other)
    }

    fn __repr__(&self) -> String {
        format!("SdfNode(nodes={})", self.inner.node_count())
    }
}

/// Compiled SDF for fast evaluation
#[pyclass(name = "CompiledSdf")]
pub struct PyCompiledSdf {
    compiled: crate::compiled::CompiledSdf,
    /// Original node retained for mesh generation (sdf_to_mesh needs SdfNode)
    source_node: SdfNode,
}

#[pymethods]
impl PyCompiledSdf {
    /// Evaluate at a single point (GIL released)
    fn eval(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let compiled = &self.compiled;
        py.allow_threads(|| crate::compiled::eval_compiled(compiled, Vec3::new(x, y, z)))
    }

    /// Evaluate compiled SDF at multiple points (GIL released, SIMD + Rayon)
    ///
    /// This is the preferred high-performance evaluation path.
    /// Internally: GIL release → Zero-Copy NumPy → SIMD 8-wide × Rayon parallel.
    fn eval_batch<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let vec_points = numpy_to_vec3_fast(points)?;

        let compiled_ref = &self.compiled;
        let distances = py.allow_threads(|| {
            crate::compiled::eval_compiled_batch_parallel(compiled_ref, &vec_points)
        });
        Ok(distances.into_pyarray(py))
    }

    /// Generate mesh from compiled SDF (GIL released, Marching Cubes parallel)
    ///
    /// Returns (vertices: ndarray[N,3], indices: ndarray[M]) as NumPy arrays.
    #[pyo3(signature = (bounds_min, bounds_max, resolution=64))]
    fn to_mesh<'py>(
        &self,
        py: Python<'py>,
        bounds_min: (f32, f32, f32),
        bounds_max: (f32, f32, f32),
        resolution: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
        let config = MarchingCubesConfig {
            resolution,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };

        let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
        let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

        let compiled_ref = &self.compiled;
        let mesh = py.allow_threads(|| {
            crate::mesh::sdf_to_mesh_compiled(compiled_ref, min, max, &config)
        });
        mesh_to_numpy(py, &mesh)
    }

    /// Get instruction count
    fn instruction_count(&self) -> usize {
        self.compiled.instruction_count()
    }

    fn __repr__(&self) -> String {
        format!("CompiledSdf(instructions={})", self.compiled.instruction_count())
    }
}

/// Compile an SDF for fast evaluation (GIL released during bytecode generation)
#[pyfunction]
fn compile_sdf(py: Python<'_>, node: &PySdfNode) -> PyCompiledSdf {
    let source_node = node.inner.clone();
    let compiled = py.allow_threads(|| {
        crate::compiled::CompiledSdf::compile(&source_node)
    });
    PyCompiledSdf {
        compiled,
        source_node,
    }
}

/// Evaluate compiled SDF at multiple points (NumPy array, GIL released)
#[pyfunction]
fn eval_compiled_batch<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let vec_points = numpy_to_vec3_fast(points)?;

    let compiled_ref = &compiled.compiled;
    let distances = py.allow_threads(|| {
        crate::compiled::eval_compiled_batch_parallel(compiled_ref, &vec_points)
    });
    Ok(distances.into_pyarray(py))
}

/// Evaluate compiled SDF using SoA layout for maximum SIMD throughput (GIL released)
///
/// 20-30% faster than `eval_compiled_batch` on large point clouds (100k+)
/// due to direct SIMD loading from contiguous memory.
#[pyfunction]
fn eval_compiled_batch_soa<'py>(
    py: Python<'py>,
    compiled: &PyCompiledSdf,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let vec_points = numpy_to_vec3_fast(points)?;
    let soa = crate::soa::SoAPoints::from_vec3_slice(&vec_points);

    let compiled_ref = &compiled.compiled;
    let soa_distances = py.allow_threads(|| {
        crate::compiled::eval_compiled_batch_soa_parallel(compiled_ref, &soa)
    });

    let distances = soa_distances.to_vec();
    Ok(distances.into_pyarray(py))
}

/// Evaluate SDF at multiple points (NumPy array, GIL released)
#[pyfunction]
fn eval_batch<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let vec_points = numpy_to_vec3_fast(points)?;

    let node_ref = &node.inner;
    let distances = py.allow_threads(|| {
        eval_batch_parallel(node_ref, &vec_points)
    });
    Ok(distances.into_pyarray(py))
}

/// Convert SDF to mesh (standard marching cubes)
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
fn to_mesh<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    let config = MarchingCubesConfig {
        resolution,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    // Release GIL during mesh generation
    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| {
        sdf_to_mesh(node_ref, min, max, &config)
    });
    mesh_to_numpy(py, &mesh)
}

/// Convert SDF to mesh using adaptive marching cubes
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=6, min_depth=2, surface_threshold=1.0))]
fn to_mesh_adaptive<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    max_depth: u32,
    min_depth: u32,
    surface_threshold: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::AdaptiveConfig;

    let config = AdaptiveConfig {
        max_depth,
        min_depth,
        surface_threshold,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    // Release GIL during mesh generation
    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| {
        crate::mesh::adaptive_marching_cubes(node_ref, min, max, &config)
    });
    mesh_to_numpy(py, &mesh)
}

/// Convert SDF to mesh using Dual Contouring (sharp edges, QEF vertex placement)
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
fn to_mesh_dual_contouring<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{dual_contouring, DualContouringConfig};

    let config = DualContouringConfig {
        resolution,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| {
        dual_contouring(node_ref, min, max, &config)
    });
    mesh_to_numpy(py, &mesh)
}

/// Decimate a mesh to reduce triangle count
#[pyfunction]
#[pyo3(signature = (vertices, indices, target_ratio=0.5, preserve_boundary=true))]
fn decimate_mesh<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
    target_ratio: f32,
    preserve_boundary: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{DecimateConfig, decimate};

    let mut mesh = numpy_to_mesh(vertices, indices)?;
    let config = DecimateConfig {
        target_ratio,
        preserve_boundary,
        ..Default::default()
    };

    py.allow_threads(|| {
        decimate(&mut mesh, &config);
    });
    mesh_to_numpy(py, &mesh)
}

/// Export mesh to OBJ file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_obj(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_obj as io_export_obj, ObjConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_obj(&mesh, path, &ObjConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to GLB file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_glb(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_glb as io_export_glb, GltfConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_glb(&mesh, path, &GltfConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to GLB bytes in memory (no temp file I/O)
#[pyfunction]
#[pyo3(signature = (vertices, indices))]
fn export_glb_bytes<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
) -> PyResult<Vec<u8>> {
    use crate::io::{export_glb_bytes as io_export_glb_bytes, GltfConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_glb_bytes(&mesh, &GltfConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Save SDF to file (GIL released during I/O)
#[pyfunction]
fn save_sdf(py: Python<'_>, node: &PySdfNode, path: &str) -> PyResult<()> {
    let tree = SdfTree::new(node.inner.clone());
    let path = path.to_string();
    py.allow_threads(|| {
        save(&tree, &path)
    }).map_err(|e| PyValueError::new_err(format!("Save error: {}", e)))
}

/// Load SDF from file (GIL released during I/O)
#[pyfunction]
fn load_sdf(py: Python<'_>, path: &str) -> PyResult<PySdfNode> {
    let path = path.to_string();
    let tree = py.allow_threads(|| {
        load(&path)
    }).map_err(|e| PyValueError::new_err(format!("Load error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Parse SDF tree from JSON string (for LLM-generated SDF)
#[pyfunction]
fn from_json(json_str: &str) -> PyResult<PySdfNode> {
    use crate::io::from_json_string;
    let tree = from_json_string(json_str)
        .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    Ok(PySdfNode { inner: tree.root })
}

/// Serialize SDF node to JSON string
#[pyfunction]
fn to_json(node: &PySdfNode) -> PyResult<String> {
    use crate::io::to_json_string;
    let tree = SdfTree::new(node.inner.clone());
    to_json_string(&tree)
        .map_err(|e| PyValueError::new_err(format!("JSON serialize error: {}", e)))
}

/// Bake SDF to 3D volume texture (CPU, returns NumPy 3D array)
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, generate_mips=false))]
fn bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
    generate_mips: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{BakeConfig, bake_volume as cpu_bake};

    let config = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        generate_mips,
        ..Default::default()
    };

    let node_ref = &node.inner;
    let volume = py.allow_threads(|| cpu_bake(node_ref, &config));
    Ok(volume.data.into_pyarray(py))
}

/// Bake SDF to 3D volume texture on GPU (returns NumPy 3D array)
#[cfg(feature = "volume")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64))]
fn gpu_bake_volume<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use crate::volume::{BakeConfig, gpu_bake_volume as gpu_bake};

    let config = BakeConfig {
        resolution: [resolution, resolution, resolution],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        ..Default::default()
    };

    let node_ref = &node.inner;
    let volume = py.allow_threads(|| gpu_bake(node_ref, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU bake error: {}", e)))?;
    Ok(volume.data.into_pyarray(py))
}

/// GPU Marching Cubes mesh generation (returns vertices + indices as NumPy arrays)
#[cfg(feature = "gpu")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, resolution=64, iso_level=0.0))]
fn gpu_marching_cubes<'py>(
    py: Python<'py>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: u32,
    iso_level: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{gpu_marching_cubes as gpu_mc, GpuMarchingCubesConfig};

    let config = GpuMarchingCubesConfig {
        resolution,
        iso_level,
        compute_normals: true,
        ..Default::default()
    };

    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);

    let node_ref = &node.inner;
    let mesh = py.allow_threads(|| gpu_mc(node_ref, min, max, &config))
        .map_err(|e| PyValueError::new_err(format!("GPU MC error: {}", e)))?;
    mesh_to_numpy(py, &mesh)
}

/// Build Sparse Voxel Octree from SDF
#[cfg(feature = "svo")]
#[pyfunction]
#[pyo3(signature = (node, bounds_min, bounds_max, max_depth=8, distance_threshold=1.5))]
fn build_svo(
    py: Python<'_>,
    node: &PySdfNode,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    max_depth: u32,
    distance_threshold: f32,
) -> PyResult<PySvo> {
    use crate::svo::{SvoBuildConfig, SparseVoxelOctree};

    let config = SvoBuildConfig {
        max_depth,
        distance_threshold,
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        use_compiled: true,
        ..Default::default()
    };

    let node_ref = &node.inner;
    let compiled = crate::compiled::CompiledSdf::compile(node_ref);
    let svo = py.allow_threads(|| SparseVoxelOctree::build_compiled(&compiled, &config));

    Ok(PySvo { inner: svo })
}

/// Python-visible SVO wrapper
#[cfg(feature = "svo")]
#[pyclass(name = "SparseVoxelOctree")]
pub struct PySvo {
    inner: crate::svo::SparseVoxelOctree,
}

#[cfg(feature = "svo")]
#[pymethods]
impl PySvo {
    /// Query distance at a point (GIL released)
    fn query_point(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| inner.query_point(Vec3::new(x, y, z)))
    }

    /// Get node count
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get leaf count
    fn leaf_count(&self) -> u32 {
        self.inner.leaf_count
    }

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get max depth
    fn max_depth(&self) -> u32 {
        self.inner.max_depth
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseVoxelOctree(nodes={}, leaves={}, depth={}, memory={}KB)",
            self.inner.node_count(),
            self.inner.leaf_count,
            self.inner.max_depth,
            self.inner.memory_bytes() / 1024,
        )
    }
}

// --- Terrain bindings ---

#[cfg(feature = "terrain")]
#[pyclass(name = "Heightmap")]
struct PyHeightmap {
    inner: crate::terrain::Heightmap,
}

#[cfg(feature = "terrain")]
#[pymethods]
impl PyHeightmap {
    /// Get height at world coordinates (GIL released)
    fn sample(&self, py: Python<'_>, x: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| inner.sample(x, z))
    }

    /// Get the height range (min, max)
    fn height_range(&self) -> (f32, f32) {
        self.inner.height_range()
    }

    /// Get sample count
    fn sample_count(&self) -> usize {
        self.inner.sample_count()
    }

    /// Get heights as a numpy array (avoids intermediate Vec clone)
    fn heights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        numpy::PyArray1::from_slice(py, &self.inner.heights)
    }

    fn __repr__(&self) -> String {
        let (min, max) = self.inner.height_range();
        format!(
            "Heightmap({}x{}, world={}x{}, range=[{:.2}, {:.2}])",
            self.inner.width, self.inner.depth,
            self.inner.world_width, self.inner.world_depth,
            min, max,
        )
    }
}

/// Generate terrain heightmap with fBm noise and optional erosion
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (width=256, depth=256, world_width=100.0, world_depth=100.0, octaves=6, height_scale=10.0, seed=42))]
fn generate_terrain(
    py: Python<'_>,
    width: u32,
    depth: u32,
    world_width: f32,
    world_depth: f32,
    octaves: u32,
    height_scale: f32,
    seed: u64,
) -> PyHeightmap {
    let mut hm = py.allow_threads(|| {
        let mut hm = crate::terrain::Heightmap::new(width, depth, world_width, world_depth);
        hm.generate_fbm(octaves, 0.5, 2.0, seed);
        hm.normalize();
        hm.scale_heights(height_scale);
        hm
    });

    PyHeightmap { inner: hm }
}

/// Apply erosion to a heightmap
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (heightmap, iterations=10000, erosion_rate=0.3, deposition_rate=0.3))]
fn erode_terrain(
    py: Python<'_>,
    heightmap: &mut PyHeightmap,
    iterations: u32,
    erosion_rate: f32,
    deposition_rate: f32,
) {
    let config = crate::terrain::ErosionConfig {
        iterations,
        erosion_rate,
        deposition_rate,
        ..Default::default()
    };

    py.allow_threads(|| {
        crate::terrain::erode(&mut heightmap.inner, &config);
    });
}

// --- Destruction bindings ---

#[cfg(feature = "destruction")]
#[pyclass(name = "VoxelGrid")]
struct PyVoxelGrid {
    inner: crate::destruction::MutableVoxelGrid,
}

#[cfg(feature = "destruction")]
#[pymethods]
impl PyVoxelGrid {
    /// Get the number of voxels
    fn voxel_count(&self) -> usize {
        self.inner.voxel_count()
    }

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get the grid resolution
    fn resolution(&self) -> (u32, u32, u32) {
        let r = self.inner.resolution;
        (r[0], r[1], r[2])
    }

    /// Get distance at a world position (GIL released)
    fn get_distance(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| {
            if let Some([gx, gy, gz]) = inner.world_to_grid(Vec3::new(x, y, z)) {
                inner.get_distance(gx, gy, gz)
            } else {
                f32::MAX
            }
        })
    }

    fn __repr__(&self) -> String {
        let r = self.inner.resolution;
        format!(
            "VoxelGrid({}x{}x{}, voxels={}, memory={}KB)",
            r[0], r[1], r[2],
            self.inner.voxel_count(),
            self.inner.memory_bytes() / 1024,
        )
    }
}

/// Create a voxel grid from an SDF node
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (node, resolution=32, bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0)))]
fn create_voxel_grid(
    py: Python<'_>,
    node: &PySdfNode,
    resolution: u32,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
) -> PyVoxelGrid {
    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);
    let node_ref = &node.inner;

    let grid = py.allow_threads(|| {
        crate::destruction::MutableVoxelGrid::from_sdf(
            node_ref,
            [resolution, resolution, resolution],
            min, max,
        )
    });

    PyVoxelGrid { inner: grid }
}

/// Carve a sphere from a voxel grid
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius))]
fn carve_sphere(
    py: Python<'_>,
    grid: &mut PyVoxelGrid,
    center: (f32, f32, f32),
    radius: f32,
) -> (u32, f32) {
    let shape = crate::destruction::CarveShape::Sphere {
        center: Vec3::new(center.0, center.1, center.2),
        radius,
    };

    let result = py.allow_threads(|| {
        crate::destruction::carve(&mut grid.inner, &shape)
    });

    (result.modified_voxels, result.removed_volume)
}

/// Fracture a voxel grid using Voronoi tessellation
#[cfg(feature = "destruction")]
#[pyfunction]
#[pyo3(signature = (grid, center, radius, piece_count=8, seed=42))]
fn voxel_fracture<'py>(
    py: Python<'py>,
    grid: &PyVoxelGrid,
    center: (f32, f32, f32),
    radius: f32,
    piece_count: u32,
    seed: u64,
) -> PyResult<Vec<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)>> {
    let config = crate::destruction::FractureConfig {
        piece_count,
        seed,
        min_piece_size: 0.01,
        ..Default::default()
    };

    let c = Vec3::new(center.0, center.1, center.2);
    let grid_ref = &grid.inner;

    let pieces = py.allow_threads(|| {
        crate::destruction::voronoi_fracture(grid_ref, c, radius, &config)
    });

    let mut results = Vec::new();
    for piece in &pieces {
        let (verts, indices) = mesh_to_numpy(py, &piece.mesh)?;
        results.push((verts, indices));
    }

    Ok(results)
}

// --- GI bindings ---

/// Bake irradiance probe grid from SVO (returns probe positions + SH coefficients)
#[cfg(feature = "gi")]
#[pyfunction]
#[pyo3(signature = (svo, grid_size=8, bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0), samples_per_probe=32))]
fn bake_gi<'py>(
    py: Python<'py>,
    svo: &PySvo,
    grid_size: u32,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    samples_per_probe: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    use crate::gi::{BakeGiConfig, bake_irradiance_grid, ConeTraceConfig, DirectionalLight};

    let config = BakeGiConfig {
        grid_size: [grid_size, grid_size, grid_size],
        bounds_min: Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
        bounds_max: Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        samples_per_probe,
        cone_config: ConeTraceConfig::default(),
        sun: Some(DirectionalLight::default()),
    };

    let svo_ref = &svo.inner;
    let grid = py.allow_threads(|| bake_irradiance_grid(svo_ref, &config));

    // Return positions (N,3) and SH coefficients (N,12) for RGB L1 SH
    let n = grid.probes.len();
    let positions: Vec<f32> = grid.probes.iter()
        .flat_map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let sh_coeffs: Vec<f32> = grid.probes.iter()
        .flat_map(|p| {
            let mut c = [0.0f32; 12];
            c[0..4].copy_from_slice(&p.sh_r.coeffs);
            c[4..8].copy_from_slice(&p.sh_g.coeffs);
            c[8..12].copy_from_slice(&p.sh_b.coeffs);
            c
        })
        .collect();

    let pos_array = numpy::PyArray1::from_vec(py, positions)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;
    let sh_array = numpy::PyArray1::from_vec(py, sh_coeffs)
        .reshape([n, 12])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    Ok((pos_array, sh_array))
}

/// Export mesh to FBX file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_fbx(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_fbx as io_export_fbx, FbxConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_fbx(&mesh, path, &FbxConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to USDA file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_usda(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_usda as io_export_usda, UsdConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_usda(&mesh, path, &UsdConfig::default(), None))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// Export mesh to Alembic file
#[pyfunction]
#[pyo3(signature = (vertices, indices, path))]
fn export_alembic(
    py: Python<'_>,
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
    path: &str,
) -> PyResult<()> {
    use crate::io::{export_alembic as io_export_alembic, AlembicConfig};
    let mesh = numpy_to_mesh(vertices, indices)?;
    py.allow_threads(|| io_export_alembic(&mesh, path, &AlembicConfig::default()))
        .map_err(|e| PyValueError::new_err(format!("Export error: {}", e)))
}

/// UV unwrap a mesh using LSCM (Least Squares Conformal Map)
///
/// Returns (positions: ndarray[N,3], uvs: ndarray[N,2], indices: ndarray[M]).
#[pyfunction]
#[pyo3(signature = (vertices, indices))]
fn uv_unwrap<'py>(
    py: Python<'py>,
    vertices: &Bound<'py, PyArray2<f32>>,
    indices: &Bound<'py, PyArray1<u32>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    use crate::mesh::{uv_unwrap as mesh_uv_unwrap, apply_uvs, UvUnwrapConfig};

    let mut mesh = numpy_to_mesh(vertices, indices)?;

    py.allow_threads(|| {
        let result = mesh_uv_unwrap(&mesh, &UvUnwrapConfig::default());
        apply_uvs(&mut mesh, &result);
    });

    let n = mesh.vertices.len();

    // Positions [N, 3]
    let positions: Vec<f32> = mesh.vertices.iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();
    let pos_array = numpy::PyArray1::from_vec(py, positions)
        .reshape([n, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    // UVs [N, 2]
    let uvs: Vec<f32> = mesh.vertices.iter()
        .flat_map(|v| [v.uv.x, v.uv.y])
        .collect();
    let uv_array = numpy::PyArray1::from_vec(py, uvs)
        .reshape([n, 2])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices_array = numpy::PyArray1::from_slice(py, &mesh.indices);

    Ok((pos_array, uv_array, indices_array))
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Helper: convert NumPy arrays to Mesh (shared by all export functions)
fn numpy_to_mesh(
    vertices: &Bound<'_, PyArray2<f32>>,
    indices: &Bound<'_, PyArray1<u32>>,
) -> PyResult<crate::mesh::Mesh> {
    use crate::mesh::{Mesh, Vertex};

    let verts = unsafe { vertices.as_array() };
    let idx = unsafe { indices.as_array() };

    if verts.shape()[1] != 3 {
        return Err(PyValueError::new_err("Vertices must have shape (N, 3)"));
    }

    let mesh_verts: Vec<Vertex> = verts
        .rows()
        .into_iter()
        .map(|row| Vertex {
            position: Vec3::new(row[0], row[1], row[2]),
            ..Default::default()
        })
        .collect();

    Ok(Mesh {
        vertices: mesh_verts,
        indices: idx.to_vec(),
    })
}

/// Helper: convert Mesh to NumPy arrays
fn mesh_to_numpy<'py>(
    py: Python<'py>,
    mesh: &crate::mesh::Mesh,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    let vertices: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();

    let vertex_array = numpy::PyArray1::from_vec(py, vertices)
        .reshape([mesh.vertices.len(), 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?;

    let indices = numpy::PyArray1::from_slice(py, &mesh.indices);

    Ok((vertex_array, indices))
}

/// Query SVO distance at multiple points (GIL released, Rayon parallel)
#[cfg(feature = "svo")]
#[pyfunction]
#[pyo3(signature = (svo, points))]
fn svo_query_batch<'py>(
    py: Python<'py>,
    svo: &PySvo,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let pts = unsafe { points.as_array() };
    if pts.shape()[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (N, 3)"));
    }
    let vec_points: Vec<Vec3> = pts
        .rows()
        .into_iter()
        .map(|row| Vec3::new(row[0], row[1], row[2]))
        .collect();

    let svo_ref = &svo.inner;
    let results = py.allow_threads(|| {
        use rayon::prelude::*;
        vec_points
            .par_iter()
            .map(|p| svo_ref.query_point(*p))
            .collect::<Vec<f32>>()
    });
    Ok(results.into_pyarray(py))
}

/// Sample heightmap at multiple world positions (GIL released, Rayon parallel)
#[cfg(feature = "terrain")]
#[pyfunction]
#[pyo3(signature = (heightmap, points))]
fn heightmap_sample_batch<'py>(
    py: Python<'py>,
    heightmap: &PyHeightmap,
    points: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let pts = unsafe { points.as_array() };
    if pts.shape()[1] != 2 {
        return Err(PyValueError::new_err("points must have shape (N, 2) for (x, z) pairs"));
    }
    let vec_points: Vec<(f32, f32)> = pts
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    let hm_ref = &heightmap.inner;
    let results = py.allow_threads(|| {
        use rayon::prelude::*;
        vec_points
            .par_iter()
            .map(|&(x, z)| hm_ref.sample(x, z))
            .collect::<Vec<f32>>()
    });
    Ok(results.into_pyarray(py))
}

/// Helper: fast conversion from C-contiguous NumPy (N,3) f32 to Vec<Vec3>.
///
/// Uses raw pointer reinterpret when contiguous (zero-copy slice),
/// falls back to row iteration otherwise.
fn numpy_to_vec3_fast(points: &Bound<'_, PyArray2<f32>>) -> PyResult<Vec<Vec3>> {
    let arr = unsafe { points.as_array() };
    if arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("Points must have shape (N, 3)"));
    }
    let n = arr.shape()[0];
    if let Some(slice) = arr.as_slice() {
        // C-contiguous: reinterpret [f32; N*3] as [Vec3; N]
        let vec3_slice = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const Vec3, n)
        };
        Ok(vec3_slice.to_vec())
    } else {
        // Non-contiguous fallback: row iteration
        Ok(arr.rows().into_iter()
            .map(|row| Vec3::new(row[0], row[1], row[2]))
            .collect())
    }
}

/// Python module
#[pymodule]
pub fn alice_sdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySdfNode>()?;
    m.add_class::<PyCompiledSdf>()?;
    m.add_function(wrap_pyfunction!(compile_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(eval_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eval_compiled_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eval_compiled_batch_soa, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(to_mesh_dual_contouring, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(export_obj, m)?)?;
    m.add_function(wrap_pyfunction!(export_glb, m)?)?;
    m.add_function(wrap_pyfunction!(export_glb_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(export_fbx, m)?)?;
    m.add_function(wrap_pyfunction!(export_usda, m)?)?;
    m.add_function(wrap_pyfunction!(export_alembic, m)?)?;
    m.add_function(wrap_pyfunction!(uv_unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(save_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(load_sdf, m)?)?;
    m.add_function(wrap_pyfunction!(from_json, m)?)?;
    m.add_function(wrap_pyfunction!(to_json, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    #[cfg(feature = "volume")]
    {
        m.add_function(wrap_pyfunction!(bake_volume, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_bake_volume, m)?)?;
    }
    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(gpu_marching_cubes, m)?)?;
    }
    #[cfg(feature = "svo")]
    {
        m.add_class::<PySvo>()?;
        m.add_function(wrap_pyfunction!(build_svo, m)?)?;
        m.add_function(wrap_pyfunction!(svo_query_batch, m)?)?;
    }
    #[cfg(feature = "destruction")]
    {
        m.add_class::<PyVoxelGrid>()?;
        m.add_function(wrap_pyfunction!(create_voxel_grid, m)?)?;
        m.add_function(wrap_pyfunction!(carve_sphere, m)?)?;
        m.add_function(wrap_pyfunction!(voxel_fracture, m)?)?;
    }
    #[cfg(feature = "terrain")]
    {
        m.add_class::<PyHeightmap>()?;
        m.add_function(wrap_pyfunction!(generate_terrain, m)?)?;
        m.add_function(wrap_pyfunction!(erode_terrain, m)?)?;
        m.add_function(wrap_pyfunction!(heightmap_sample_batch, m)?)?;
    }
    #[cfg(feature = "gi")]
    {
        m.add_function(wrap_pyfunction!(bake_gi, m)?)?;
    }
    Ok(())
}
