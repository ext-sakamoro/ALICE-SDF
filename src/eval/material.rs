//! The material id at a point: one walk that returns the distance and the material together
//!
//! The rules are written once, in [`material_rule`]: which operand a two-operand node takes
//! its material from, and that `WithMaterial` tags its subtree. The point each child is read at
//! is the point [`super::descend`] gives it, the same walk [`super::eval`] takes, so the
//! material follows every transform, repetition and fold the distance follows. The shader
//! transpilers emit the material from the same table (`MaterialRule`), so the GPU and the CPU
//! cannot hold two lists of rules.
//!
//! The walk is a loop with an explicit work stack: one evaluation of each node, so the cost is
//! that of [`super::eval`] and the depth of the tree does not reach the native stack.
//!
//! Author: Moroya Sakamoto

use super::{combine, descend, finish, Step};
use crate::operations::BlendMode;
use crate::types::SdfNode;
use crate::work_stack::WorkStack;
use glam::Vec3;

/// Which operand of a two-operand node gives its material, from the operands' values `da`, `db`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPick {
    /// the nearer operand: `a` when `da <= db` (union-like, interpolating nodes)
    Nearer,
    /// the farther operand: `a` when `da >= db` (intersection-like nodes)
    Farther,
    /// `a` when `da >= -db`, else `b` (subtraction-like nodes: `b` where it cuts `a`)
    Cut,
}

impl MaterialPick {
    /// Whether the material is the first operand's
    #[must_use]
    pub const fn takes_first(self, da: f32, db: f32) -> bool {
        match self {
            Self::Nearer => da <= db,
            Self::Farther => da >= db,
            Self::Cut => da >= -db,
        }
    }
}

/// What a node does to the material of the walk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialRule {
    /// no children: material 0
    Leaf,
    /// one child: its material
    Pass,
    /// one child: its material where it has one, else this id
    Tag(u32),
    /// two operands: the material of the one the pick names
    Pick(MaterialPick),
}

/// The material rule of `node` (the one list of rules; every variant is named)
#[must_use]
pub const fn material_rule(node: &SdfNode) -> MaterialRule {
    use MaterialPick::{Cut, Farther, Nearer};
    use MaterialRule::{Leaf, Pass, Pick, Tag};
    match node {
        SdfNode::WithMaterial { material_id, .. } => Tag(*material_id),

        SdfNode::Union { .. }
        | SdfNode::SmoothUnion { .. }
        | SdfNode::ChamferUnion { .. }
        | SdfNode::StairsUnion { .. }
        | SdfNode::ColumnsUnion { .. }
        | SdfNode::ExpSmoothUnion { .. }
        | SdfNode::XOR { .. }
        | SdfNode::Morph { .. }
        | SdfNode::Pipe { .. }
        | SdfNode::MetricBlend { .. } => Pick(Nearer),
        SdfNode::Intersection { .. }
        | SdfNode::SmoothIntersection { .. }
        | SdfNode::ChamferIntersection { .. }
        | SdfNode::StairsIntersection { .. }
        | SdfNode::ColumnsIntersection { .. }
        | SdfNode::ExpSmoothIntersection { .. } => Pick(Farther),
        SdfNode::Subtraction { .. }
        | SdfNode::SmoothSubtraction { .. }
        | SdfNode::ChamferSubtraction { .. }
        | SdfNode::StairsSubtraction { .. }
        | SdfNode::ColumnsSubtraction { .. }
        | SdfNode::ExpSmoothSubtraction { .. }
        | SdfNode::Engrave { .. }
        | SdfNode::Groove { .. }
        | SdfNode::Tongue { .. } => Pick(Cut),
        SdfNode::SmoothBlend { mode, .. } => Pick(match mode {
            BlendMode::Union => Nearer,
            BlendMode::Intersection => Farther,
            BlendMode::Subtraction => Cut,
        }),

        SdfNode::Translate { .. }
        | SdfNode::Rotate { .. }
        | SdfNode::Scale { .. }
        | SdfNode::ScaleNonUniform { .. }
        | SdfNode::ProjectiveTransform { .. }
        | SdfNode::LatticeDeform { .. }
        | SdfNode::SdfSkinning { .. }
        | SdfNode::GridWarp { .. }
        | SdfNode::BellPull { .. }
        | SdfNode::Twist { .. }
        | SdfNode::Bend { .. }
        | SdfNode::RepeatInfinite { .. }
        | SdfNode::RepeatFinite { .. }
        | SdfNode::RepeatCells { .. }
        | SdfNode::Elongate { .. }
        | SdfNode::Mirror { .. }
        | SdfNode::OctantMirror { .. }
        | SdfNode::Revolution { .. }
        | SdfNode::Extrude { .. }
        | SdfNode::Taper { .. }
        | SdfNode::PolarRepeat { .. }
        | SdfNode::SweepBezier { .. }
        | SdfNode::Shear { .. }
        | SdfNode::IcosahedralSymmetry { .. }
        | SdfNode::IFS { .. }
        | SdfNode::Noise { .. }
        | SdfNode::Round { .. }
        | SdfNode::Onion { .. }
        | SdfNode::Displacement { .. }
        | SdfNode::SineDisplacement { .. }
        | SdfNode::Animated { .. }
        | SdfNode::Pose { .. }
        | SdfNode::HeightmapDisplacement { .. }
        | SdfNode::SurfaceRoughness { .. } => Pass,

        SdfNode::Sphere { .. }
        | SdfNode::Box3d { .. }
        | SdfNode::Cylinder { .. }
        | SdfNode::Torus { .. }
        | SdfNode::Plane { .. }
        | SdfNode::Capsule { .. }
        | SdfNode::Cone { .. }
        | SdfNode::Ellipsoid { .. }
        | SdfNode::RoundedCone { .. }
        | SdfNode::Pyramid { .. }
        | SdfNode::Octahedron { .. }
        | SdfNode::HexPrism { .. }
        | SdfNode::Link { .. }
        | SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        | SdfNode::Circle2D { .. }
        | SdfNode::Rect2D { .. }
        | SdfNode::Segment2D { .. }
        | SdfNode::Polygon2D { .. }
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. }
        | SdfNode::Terrain { .. }
        | SdfNode::Loft { .. }
        | SdfNode::LoftCurves { .. }
        | SdfNode::SectionStack { .. }
        | SdfNode::SmoothSectionStack { .. }
        | SdfNode::MetricBall { .. } => Leaf,
    }
}

/// What the walk still owes a node while one of its children is evaluated
enum Owed<'a> {
    /// evaluate the second operand at the node's point, then pick
    Second(&'a SdfNode, &'a SdfNode, Vec3),
    /// combine and pick with the first operand's value and material
    Pick(&'a SdfNode, Vec3, f32, u32),
    /// finish the one-child node and apply its rule to the child's material
    Finish(&'a SdfNode, Vec3, [f32; 3]),
}

/// The distance and the material id of `node` at `point`
///
/// The distance is bit for bit [`super::eval`]'s (the same `descend`, `combine` and `finish`).
#[must_use]
pub fn eval_with_material(node: &SdfNode, point: Vec3) -> (f32, u32) {
    let mut owed: WorkStack<Owed<'_>, 128> = WorkStack::new();
    let (mut node, mut point) = (node, point);
    loop {
        let (mut d, mut m) = loop {
            match descend(node, point) {
                Step::Leaf(d) => break (d, 0),
                Step::Two(a, b) => {
                    owed.push(Owed::Second(node, b, point));
                    node = a;
                }
                Step::One(child, q, extra) => {
                    owed.push(Owed::Finish(node, point, extra));
                    node = child;
                    point = q;
                }
            }
        };
        loop {
            match owed.pop() {
                None => return (d, m),
                Some(Owed::Second(n, b, q)) => {
                    owed.push(Owed::Pick(n, q, d, m));
                    node = b;
                    point = q;
                    break;
                }
                Some(Owed::Pick(n, q, da, ma)) => {
                    // `descend` gives two operands only to nodes with a pick rule
                    // (`tests::rules_match_the_shape_descend_gives`)
                    let pick = match material_rule(n) {
                        MaterialRule::Pick(pick) => pick,
                        other => {
                            debug_assert!(false, "two operands under a {other:?} rule");
                            MaterialPick::Nearer
                        }
                    };
                    if pick.takes_first(da, d) {
                        m = ma;
                    }
                    d = combine(n, da, d, q);
                }
                Some(Owed::Finish(n, q, extra)) => {
                    if let MaterialRule::Tag(id) = material_rule(n) {
                        if m == 0 {
                            m = id;
                        }
                    }
                    d = finish(n, d, q, extra);
                }
            }
        }
    }
}

// The shared test corpus file, of which the test below calls only corpus()
#[cfg(test)]
#[allow(dead_code, unused_imports, clippy::all, clippy::pedantic)]
#[path = "../../tests/common/corpus.rs"]
mod corpus;

#[cfg(test)]
mod tests {
    use super::corpus;
    use super::*;
    use std::collections::BTreeSet;

    const VARIANTS: &str = include_str!("../../tests/fixtures/sdf_node_variants.tsv");

    fn variant_name(node: &SdfNode) -> String {
        let s = format!("{node:?}");
        s.split(|c: char| !c.is_alphanumeric())
            .next()
            .unwrap_or("")
            .to_owned()
    }

    /// Every node of `node`'s tree, by `descend`'s children at `point`
    fn visit<'a>(node: &'a SdfNode, point: Vec3, out: &mut Vec<(&'a SdfNode, Step<'a>)>) {
        let step = descend(node, point);
        let kids: Vec<(&SdfNode, Vec3)> = match &step {
            Step::Leaf(_) => vec![],
            Step::Two(a, b) => vec![(*a, point), (*b, point)],
            Step::One(c, q, _) => vec![(*c, *q)],
        };
        out.push((node, step));
        for (k, q) in kids {
            visit(k, q, out);
        }
    }

    /// The rule table and `descend` agree on every variant: two operands ⇔ a pick, one child ⇔
    /// pass or tag, no children ⇔ leaf; every variant of the stored-form table is seen
    #[test]
    fn rules_match_the_shape_descend_gives() {
        let mut seen = BTreeSet::new();
        let mut checked = 0;
        let mut trees: Vec<SdfNode> = corpus::corpus().into_iter().map(|(_, t)| t).collect();
        // the two variants the corpus leaves out (both leaves)
        trees.push(SdfNode::triangle(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.2),
        ));
        trees.push(SdfNode::bezier(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            0.1,
        ));
        for tree in &trees {
            let mut nodes = Vec::new();
            visit(tree, Vec3::new(0.3, -0.2, 0.1), &mut nodes);
            for (node, step) in nodes {
                let rule = material_rule(node);
                let ok = matches!(
                    (&step, rule),
                    (Step::Leaf(_), MaterialRule::Leaf)
                        | (Step::One(..), MaterialRule::Pass | MaterialRule::Tag(_))
                        | (Step::Two(..), MaterialRule::Pick(_))
                );
                assert!(
                    ok,
                    "{}: rule {rule:?} for a node descend reads otherwise",
                    variant_name(node)
                );
                seen.insert(variant_name(node));
                checked += 1;
            }
        }
        let all: BTreeSet<String> = VARIANTS
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| l.split('\t').nth(1).map(str::to_owned))
            .collect();
        assert!(all.len() > 100, "{} variant names read", all.len());
        let missing: Vec<_> = all.difference(&seen).collect();
        assert!(
            missing.is_empty(),
            "{checked} nodes checked; variants not seen: {missing:?}"
        );
    }
}
