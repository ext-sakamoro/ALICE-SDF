//! LLM integration utilities for ALICE-SDF
//!
//! Provides JSON schema generation and validation for LLM-based
//! SDF generation workflows.
//!
//! Author: Moroya Sakamoto

use crate::types::{SdfNode, SdfTree};

/// Validate an SDF JSON string and return the parsed tree.
///
/// Returns a human-readable error message if parsing fails,
/// suitable for feeding back to an LLM for correction.
pub fn validate_sdf_json(json: &str) -> Result<SdfTree, String> {
    let tree: SdfTree = serde_json::from_str(json).map_err(|e| {
        format!(
            "JSON parse error at line {} column {}: {}",
            e.line(),
            e.column(),
            e
        )
    })?;

    // Validate tree is non-empty
    if tree.node_count() == 0 {
        return Err("SDF tree is empty (no nodes)".to_string());
    }

    // Validate no negative dimensions
    validate_node_dimensions(&tree.root)?;

    Ok(tree)
}

/// Check that primitive dimensions are non-negative
fn validate_node_dimensions(node: &SdfNode) -> Result<(), String> {
    match node {
        SdfNode::Sphere { radius } => {
            if *radius <= 0.0 {
                return Err(format!("Sphere radius must be positive, got {}", radius));
            }
        }
        SdfNode::Box3d { half_extents } => {
            if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
                return Err(format!(
                    "Box3d half_extents must be positive, got {:?}",
                    half_extents
                ));
            }
        }
        SdfNode::Cylinder {
            radius,
            half_height,
        } => {
            if *radius <= 0.0 || *half_height <= 0.0 {
                return Err(format!(
                    "Cylinder dimensions must be positive, got radius={}, half_height={}",
                    radius, half_height
                ));
            }
        }
        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => {
            if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
                return Err(format!(
                    "RoundedBox half_extents must be positive, got {:?}",
                    half_extents
                ));
            }
            if *round_radius < 0.0 {
                return Err(format!(
                    "RoundedBox round_radius must be non-negative, got {}",
                    round_radius
                ));
            }
        }
        SdfNode::Onion { child, thickness } => {
            if *thickness <= 0.0 {
                return Err(format!(
                    "Onion thickness must be positive, got {}",
                    thickness
                ));
            }
            validate_node_dimensions(child)?;
        }
        // Recurse into children for operations
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. } => {
            validate_node_dimensions(a)?;
            validate_node_dimensions(b)?;
        }
        // Recurse into single-child nodes
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Scale { child, .. }
        | SdfNode::Twist { child, .. }
        | SdfNode::Bend { child, .. }
        | SdfNode::Round { child, .. }
        | SdfNode::Elongate { child, .. }
        | SdfNode::Mirror { child, .. }
        | SdfNode::RepeatInfinite { child, .. }
        | SdfNode::RepeatFinite { child, .. }
        | SdfNode::Revolution { child, .. }
        | SdfNode::Extrude { child, .. } => {
            validate_node_dimensions(child)?;
        }
        // Other primitives and nodes — skip detailed validation
        _ => {}
    }
    Ok(())
}

/// Generate a compact JSON schema summary for embedding in LLM prompts.
///
/// Returns a string listing all available SdfNode variants
/// grouped by category.
pub fn schema_summary() -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("ALICE-SDF Node Types (126 total):\n\n");

    s.push_str("Primitives (72): Sphere, Box3d, RoundedBox, Cylinder, Torus, ");
    s.push_str("Capsule, Cone, Ellipsoid, RoundedCone, Plane, HexagonalPrism, ");
    s.push_str("TriangularPrism, RoundedCylinder, CappedTorus, Link, ...\n\n");

    s.push_str("Operations (24): Union, Intersection, Subtraction, ");
    s.push_str("SmoothUnion(k), SmoothIntersection(k), SmoothSubtraction(k), ...\n\n");

    s.push_str("Transforms (7): Translate(offset), Rotate(quat), ");
    s.push_str("Scale(factor), ...\n\n");

    s.push_str("Modifiers (23): Onion(thickness), Round(radius), Twist(strength), ");
    s.push_str("Bend(strength), Elongate(vec3), RepeatInfinite(spacing), ");
    s.push_str("RepeatFinite(spacing,count), Mirror(axis), Revolution(offset), ");
    s.push_str("Extrude(height), ...\n");

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_json() {
        let json = r#"{
            "version": "0.1.0",
            "root": {"Sphere": {"radius": 10.0}}
        }"#;
        assert!(validate_sdf_json(json).is_ok());
    }

    #[test]
    fn test_validate_negative_radius() {
        let json = r#"{
            "version": "0.1.0",
            "root": {"Sphere": {"radius": -1.0}}
        }"#;
        let err = validate_sdf_json(json).unwrap_err();
        assert!(err.contains("positive"));
    }

    #[test]
    fn test_validate_invalid_json() {
        let json = r#"{ invalid }"#;
        assert!(validate_sdf_json(json).is_err());
    }

    #[test]
    fn test_validate_nested() {
        let json = r#"{
            "version": "0.1.0",
            "root": {
                "Subtraction": {
                    "a": {"Sphere": {"radius": 10.0}},
                    "b": {"Box3d": {"half_extents": [5.0, 5.0, 5.0]}}
                }
            }
        }"#;
        assert!(validate_sdf_json(json).is_ok());
    }

    #[test]
    fn test_schema_summary() {
        let s = schema_summary();
        assert!(s.contains("126 total"));
        assert!(s.contains("Primitives"));
        assert!(s.contains("Operations"));
    }
}
