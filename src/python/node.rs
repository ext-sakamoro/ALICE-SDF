//! PySdfNode struct definition and primitive constructors.

use glam::{Vec2, Vec3};
use pyo3::prelude::*;

use crate::types::SdfNode;

/// Python-visible SdfNode wrapper
#[pyclass(name = "SdfNode")]
#[derive(Clone)]
pub struct PySdfNode {
    pub(crate) inner: SdfNode,
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
    fn triangle(
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        cx: f32,
        cy: f32,
        cz: f32,
    ) -> Self {
        PySdfNode {
            inner: SdfNode::triangle(
                Vec3::new(ax, ay, az),
                Vec3::new(bx, by, bz),
                Vec3::new(cx, cy, cz),
            ),
        }
    }

    /// Create a quadratic Bezier curve tube (9 floats + radius)
    #[staticmethod]
    fn bezier(
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        cx: f32,
        cy: f32,
        cz: f32,
        radius: f32,
    ) -> Self {
        PySdfNode {
            inner: SdfNode::bezier(
                Vec3::new(ax, ay, az),
                Vec3::new(bx, by, bz),
                Vec3::new(cx, cy, cz),
                radius,
            ),
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

    /// Create a 2D circle extruded along Z
    #[staticmethod]
    fn circle_2d(radius: f32, half_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::circle_2d(radius, half_height),
        }
    }

    /// Create a 2D rectangle extruded along Z
    #[staticmethod]
    fn rect_2d(half_w: f32, half_h: f32, half_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rect_2d(half_w, half_h, half_height),
        }
    }

    /// Create a 2D line segment extruded along Z
    #[staticmethod]
    fn segment_2d(ax: f32, ay: f32, bx: f32, by: f32, thickness: f32, half_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::segment_2d(ax, ay, bx, by, thickness, half_height),
        }
    }

    /// Create a 2D polygon extruded along Z
    #[staticmethod]
    fn polygon_2d(vertices: Vec<(f32, f32)>, half_height: f32) -> Self {
        let verts: Vec<Vec2> = vertices.iter().map(|(x, y)| Vec2::new(*x, *y)).collect();
        PySdfNode {
            inner: SdfNode::polygon_2d(verts, half_height),
        }
    }

    /// Create a 2D rounded rectangle extruded along Z
    #[staticmethod]
    fn rounded_rect_2d(half_w: f32, half_h: f32, round_radius: f32, half_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::rounded_rect_2d(half_w, half_h, round_radius, half_height),
        }
    }

    /// Create a 2D annular (ring) shape extruded along Z
    #[staticmethod]
    fn annular_2d(outer_radius: f32, thickness: f32, half_height: f32) -> Self {
        PySdfNode {
            inner: SdfNode::annular_2d(outer_radius, thickness, half_height),
        }
    }

    /// Evaluate at a single point (GIL released)
    fn eval(&self, py: Python<'_>, x: f32, y: f32, z: f32) -> f32 {
        let inner = &self.inner;
        py.allow_threads(|| crate::eval::eval(inner, Vec3::new(x, y, z)))
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
        let shader = crate::compiled::GlslShader::transpile(
            &self.inner,
            crate::compiled::GlslTranspileMode::Hardcoded,
        );
        shader.source
    }

    /// Generate HLSL shader code
    #[cfg(feature = "hlsl")]
    fn to_hlsl(&self) -> String {
        let shader = crate::compiled::HlslShader::transpile(
            &self.inner,
            crate::compiled::HlslTranspileMode::Hardcoded,
        );
        shader.source
    }

    /// Generate WGSL shader code
    #[cfg(feature = "gpu")]
    fn to_wgsl(&self) -> String {
        let shader = crate::compiled::WgslShader::transpile(
            &self.inner,
            crate::compiled::TranspileMode::Hardcoded,
        );
        shader.source
    }

    // --- Operator overloads for Pythonic DSL ---

    /// `a | b` → union
    fn __or__(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().union(other.inner.clone()),
        }
    }

    /// `a & b` → intersection
    fn __and__(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().intersection(other.inner.clone()),
        }
    }

    /// `a - b` → subtraction
    fn __sub__(&self, other: &PySdfNode) -> Self {
        PySdfNode {
            inner: self.inner.clone().subtract(other.inner.clone()),
        }
    }

    fn __repr__(&self) -> String {
        format!("SdfNode(nodes={})", self.inner.node_count())
    }
}
