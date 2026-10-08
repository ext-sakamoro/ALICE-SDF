//! `compute_tight_aabb` returns a box that contains the surface.
//!
//! oracle: the closed-form extents of a sphere (`[-r, r]³`) and a torus
//! (`R + r` in the ring plane, `r` along its axis), and, for every corpus
//! node, the ordering `min <= max` on each axis. Until the bisection returned
//! the outer end of its bracket, a sphere of radius 0.6 came back as
//! `±0.5999994` and the plane node as a box with `min.y > max.y`; both pass
//! only when the outer end is returned.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{compute_tight_aabb, compute_tight_aabb_with_config, TightAabbConfig};

fn assert_covers(name: &str, bb: Aabb, half: Vec3) {
    assert!(
        bb.min.cmple(-half).all() && bb.max.cmpge(half).all(),
        "{name}: {bb:?} does not contain ±{half:?}"
    );
}

#[test]
fn sphere_and_torus_boxes_contain_the_closed_form_extents() {
    let configs = [
        ("small", TightAabbConfig::preset_small()),
        ("medium", TightAabbConfig::preset_medium()),
    ];
    let mut compared = 0;
    for (cname, cfg) in &configs {
        for r in [0.6_f32, 1.0, 2.5, 7.0] {
            let bb = compute_tight_aabb_with_config(&SdfNode::sphere(r), cfg);
            assert_covers(&format!("{cname} sphere {r}"), bb, Vec3::splat(r));
            compared += 1;
        }
        for (major, minor) in [(0.7_f32, 0.2_f32), (1.5, 0.25), (3.0, 1.0)] {
            let bb = compute_tight_aabb_with_config(&SdfNode::torus(major, minor), cfg);
            let half = Vec3::new(major + minor, minor, major + minor);
            assert_covers(&format!("{cname} torus {major} {minor}"), bb, half);
            compared += 1;
        }
    }
    assert_eq!(compared, 14);
}

#[test]
fn every_corpus_box_is_ordered() {
    let mut compared = 0;
    for (name, node) in common::corpus::corpus() {
        let bb = compute_tight_aabb(&node);
        assert!(
            bb.min.cmple(bb.max).all(),
            "{name}: min > max on some axis: {bb:?}"
        );
        compared += 1;
    }
    assert!(compared > 100, "corpus has only {compared} nodes");
}
