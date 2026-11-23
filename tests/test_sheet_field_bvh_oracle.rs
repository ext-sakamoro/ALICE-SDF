//! The sheet-box BVH changes no answer: on a field of 300 small sheets, the BVH-backed queries equal the
//! linear scans they replace, bit for bit (oracle: the linear scans, which `test_sheet_field_oracle` pins to
//! closed forms on one sheet).
//!
//! - `sheets_containing(p)` equals the brute-force list of sheets whose box holds `p` (ascending).
//! - `eval(p)` equals `eval_linear(p)`: the same `f`, `g` (bits) and owning sheet (lowest index on ties).
//! - `march` equals `march_linear` for rays through the cloud (same kind, same `t` bits, same sheet).
#![allow(clippy::unwrap_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32) / ((1u64 << 24) as f32)
}

/// 300 disks of radius 1.5 mm in random planes, centres in a 40 mm cube, each one brick of h = 0.5 around it
fn cloud() -> SheetField {
    let mut s = 3u64;
    let mut sheets: Vec<Sheet> = Vec::new();
    for _ in 0..300 {
        let c = Vec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 40.0;
        let n = Vec3::new(rng(&mut s) - 0.5, rng(&mut s) - 0.5, rng(&mut s) - 0.5).normalize();
        let lo = c - Vec3::splat(2.0 + 0.13);
        let lv = level_from_fn(
            lo,
            0.5,
            [1, 1, 1],
            |_c, _e| true,
            move |p| {
                let r = p - c;
                let along = r.dot(n);
                let inplane = (r - n * along).length();
                vec![along, inplane - 1.5]
            },
        )
        .unwrap();
        sheets.push(sheet(vec![lv]).unwrap());
    }
    SheetField::new(sheets, ValidRule::CellV1)
}

#[test]
fn sheets_containing_equals_brute_force() {
    let f = cloud();
    let mut s = 17u64;
    let mut nonempty = 0;
    for _ in 0..5000 {
        let p = Vec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 44.0 - Vec3::splat(2.0);
        let want: Vec<usize> = (0..f.sheets.len())
            .filter(|&k| p.cmpge(f.sheets[k].lo).all() && p.cmple(f.sheets[k].hi).all())
            .collect();
        let got = f.sheets_containing(p);
        assert_eq!(got, want, "{p:?}");
        nonempty += usize::from(!want.is_empty());
    }
    assert!(nonempty > 500, "points inside some box: {nonempty}");
}

#[test]
fn eval_equals_the_linear_scan() {
    let f = cloud();
    let mut s = 23u64;
    for _ in 0..5000 {
        let p = Vec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 50.0 - Vec3::splat(5.0);
        let (a, b) = (f.eval(p), f.eval_linear(p));
        assert_eq!(
            (a.0.to_bits(), a.1.to_bits(), a.2),
            (b.0.to_bits(), b.1.to_bits(), b.2),
            "{p:?}"
        );
    }
}

#[test]
fn march_equals_the_linear_march() {
    let f = cloud();
    let mut s = 29u64;
    let mut hits = 0;
    for _ in 0..400 {
        let o = Vec3::new(rng(&mut s) * 40.0, rng(&mut s) * 40.0, 60.0);
        let d = Vec3::new(rng(&mut s) * 0.4 - 0.2, rng(&mut s) * 0.4 - 0.2, -1.0).normalize();
        let (a, b) = (
            f.march(o, d, 120.0, 4000),
            f.march_linear(o, d, 120.0, 4000),
        );
        match (a, b) {
            (MarchResult::Hit(x), MarchResult::Hit(y)) => {
                hits += 1;
                assert_eq!(
                    (x.t.to_bits(), x.sheet),
                    (y.t.to_bits(), y.sheet),
                    "{o:?} {d:?}"
                );
            }
            (x, y) => assert_eq!(x, y, "{o:?} {d:?}"),
        }
    }
    assert!(hits > 20, "hits {hits}");
}
