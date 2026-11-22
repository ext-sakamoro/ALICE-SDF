//! Building sheets from channel functions (analytic shapes in tests and examples; exporters write the same layout).

use glam::Vec3;

use super::{Channel, Level, Sheet, SheetFieldError, BRICK, BRICK_NODES, NODES};

/// Quantize one channel's node values for the active bricks (`values[brick][node]`) to bytes with a per-brick
/// scale and offset. With `drop_nonpositive`, a brick whose nodes are all <= 0 is dropped (row -1): its
/// trilinear value is then <= 0 everywhere in the brick, so `max(value, 0) = 0` exactly (the `d_b` channel).
pub fn quantize(values: &[Vec<f32>], drop_nonpositive: bool) -> Channel {
    let mut ch = Channel {
        rows: Vec::with_capacity(values.len()),
        q: Vec::new(),
        scale: Vec::new(),
        offset: Vec::new(),
    };
    for v in values {
        assert_eq!(v.len(), BRICK_NODES, "a brick has (BRICK+1)^3 nodes");
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if drop_nonpositive && hi <= 0.0 {
            ch.rows.push(-1);
            continue;
        }
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let sc = (hi - lo).max(1e-6) / 255.0;
        ch.rows.push(ch.scale.len() as i32);
        ch.scale.push(sc);
        ch.offset.push(lo);
        ch.q.extend(
            v.iter()
                .map(|x| ((x - lo) / sc).round().clamp(0.0, 255.0) as u8),
        );
    }
    ch
}

/// One level from a channel function: bricks of `BRICK^3` cells of size `h` from `lo`, `nb` per axis; a brick
/// is active when `active(brick_center, brick_edge)` says so. `chan(p)` returns `[s, d_b]` or `[s, d_b, u, v]`.
pub fn level_from_fn(
    lo: Vec3,
    h: f32,
    nb: [u32; 3],
    active: impl Fn(Vec3, f32) -> bool,
    chan: impl Fn(Vec3) -> Vec<f32>,
) -> Result<Level, SheetFieldError> {
    if !(h > 0.0) || nb.iter().any(|&n| n == 0) {
        return Err(SheetFieldError::Shape(format!("h {h}, nb {nb:?}")));
    }
    let edge = BRICK as f32 * h;
    let mut index = vec![-1i32; (nb[0] * nb[1] * nb[2]) as usize];
    let mut per_ch: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut nch = None;
    let mut count = 0i32;
    for bx in 0..nb[0] {
        for by in 0..nb[1] {
            for bz in 0..nb[2] {
                let center = lo + (Vec3::new(bx as f32, by as f32, bz as f32) + 0.5) * edge;
                if !active(center, edge) {
                    continue;
                }
                index[((bx * nb[1] + by) * nb[2] + bz) as usize] = count;
                count += 1;
                let mut nodes: Vec<Vec<f32>> = Vec::new();
                for x in 0..NODES {
                    for y in 0..NODES {
                        for z in 0..NODES {
                            let p = lo
                                + (Vec3::new(
                                    (bx as usize * BRICK + x) as f32,
                                    (by as usize * BRICK + y) as f32,
                                    (bz as usize * BRICK + z) as f32,
                                )) * h;
                            let c = chan(p);
                            if c.len() != 2 && c.len() != 4 {
                                return Err(SheetFieldError::Shape(format!(
                                    "{} channels (2 or 4)",
                                    c.len()
                                )));
                            }
                            if *nch.get_or_insert(c.len()) != c.len() {
                                return Err(SheetFieldError::Shape("channel count changes".into()));
                            }
                            if nodes.is_empty() {
                                nodes = vec![Vec::with_capacity(BRICK_NODES); c.len()];
                            }
                            for (i, v) in c.into_iter().enumerate() {
                                nodes[i].push(v);
                            }
                        }
                    }
                }
                for (i, v) in nodes.into_iter().enumerate() {
                    if per_ch.len() <= i {
                        per_ch.push(Vec::new());
                    }
                    per_ch[i].push(v);
                }
            }
        }
    }
    if count == 0 {
        return Err(SheetFieldError::Shape("no active brick".into()));
    }
    let channels = per_ch
        .iter()
        .enumerate()
        .map(|(i, v)| quantize(v, i == 1))
        .collect();
    Ok(Level {
        h,
        lo,
        nb,
        index,
        channels,
    })
}

/// A sheet from its levels (lookup order); its box and coarsest cell are derived.
pub fn sheet(levels: Vec<Level>) -> Result<Sheet, SheetFieldError> {
    if levels.is_empty() {
        return Err(SheetFieldError::Shape("a sheet has no level".into()));
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut hc = 0.0f32;
    for lv in &levels {
        let n = Vec3::new(lv.nb[0] as f32, lv.nb[1] as f32, lv.nb[2] as f32);
        lo = lo.min(lv.lo);
        hi = hi.max(lv.lo + n * (BRICK as f32 * lv.h));
        hc = hc.max(lv.h);
    }
    Ok(Sheet { levels, lo, hi, hc })
}
