//! Building sheets from channel functions (analytic shapes in tests and examples; exporters write the same layout).

use glam::Vec3;

use super::{
    Channel, Level, Sheet, SheetFieldError, BAND, BAND_WORDS, BRICK, BRICK_NODES, COARSE_NODES,
    NODES,
};

/// Quantize one channel's node values for the active bricks (`values[brick][node]`) to bytes.
///
/// Each brick has its own scale and offset. With `drop_nonpositive`, a brick whose nodes are all <= 0 is dropped (row -1): its
/// trilinear value is then <= 0 everywhere in the brick, so `max(value, 0) = 0` exactly (the `d_b` channel).
pub fn quantize(values: &[Vec<f32>], drop_nonpositive: bool) -> Channel {
    let mut ch = Channel {
        rows: Vec::with_capacity(values.len()),
        q: Vec::new(),
        scale: Vec::new(),
        offset: Vec::new(),
        nodes: NODES as u32,
        starts: Vec::new(),
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

/// The same channel on a coarser grid of `nodes` per brick axis (one of [`COARSE_NODES`]): every
/// `BRICK / (nodes - 1)`-th node of each kept row, with that row's scale and offset (no re-quantization).
pub fn coarsen(ch: &Channel, nodes: u32) -> Result<Channel, SheetFieldError> {
    if ch.nodes != NODES as u32 || !COARSE_NODES.contains(&nodes) {
        return Err(SheetFieldError::Shape(format!(
            "coarsen a {}-node channel to {nodes} nodes",
            ch.nodes
        )));
    }
    let n = nodes as usize;
    let stride = BRICK / (n - 1);
    let mut q = Vec::with_capacity(ch.scale.len() * n * n * n);
    for r in 0..ch.scale.len() {
        let fine = &ch.q[r * BRICK_NODES..(r + 1) * BRICK_NODES];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    q.push(fine[((x * stride) * NODES + y * stride) * NODES + z * stride]);
                }
            }
        }
    }
    Ok(Channel {
        rows: ch.rows.clone(),
        q,
        scale: ch.scale.clone(),
        offset: ch.offset.clone(),
        nodes,
        starts: Vec::new(),
    })
}

/// One level from a channel function: bricks of `BRICK^3` cells of size `h` from `lo`, `nb` per axis.
///
/// A brick is active when `active(brick_center, brick_edge)` says so. `chan(p)` returns `[s, d_b]` or `[s, d_b, u, v]`.
pub fn level_from_fn(
    lo: Vec3,
    h: f32,
    nb: [u32; 3],
    active: impl Fn(Vec3, f32) -> bool,
    chan: impl Fn(Vec3) -> Vec<f32>,
) -> Result<Level, SheetFieldError> {
    if !h.is_finite() || h <= 0.0 || nb.contains(&0) {
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
                            if !matches!(c.len(), 2 | 4 | 7) {
                                return Err(SheetFieldError::Shape(format!(
                                    "{} channels (2, 4 or 7)",
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
        band: Vec::new(),
    })
}

/// The level with its attribute channels (2..) stored as a [`BAND`].
///
/// Per active brick, only the nodes that are corners of a cell the surface may cross are kept: those whose 8 corner `s` values (from the brick's own `s` row) satisfy
/// `min <= eps` and `max >= -eps` with `eps = eps_steps * scale` of that row. A brick that stores no `s` row keeps
/// every node. A hit interpolates attributes inside the cell it lies in, so a band that holds every such cell
/// gives the same attributes as the full channel; `eps_steps` covers hits taken a little off `s = 0`.
pub fn band(lv: &Level, eps_steps: f32) -> Result<Level, SheetFieldError> {
    if lv.channels.len() < 3 || !lv.band.is_empty() {
        return Err(SheetFieldError::Shape(
            "band: a level with attributes, not yet a band".into(),
        ));
    }
    if lv.channels.iter().any(|c| c.nodes != NODES as u32) {
        return Err(SheetFieldError::Shape(
            "band: every channel on 9 nodes".into(),
        ));
    }
    let s = &lv.channels[0];
    let n_active = s.rows.len();
    let mut mask = vec![0u32; n_active * BAND_WORDS];
    for (k, &row) in s.rows.iter().enumerate() {
        let m = &mut mask[k * BAND_WORDS..(k + 1) * BAND_WORDS];
        if row < 0 {
            for i in 0..BRICK_NODES {
                m[i / 32] |= 1 << (i % 32);
            }
            continue;
        }
        let r = row as usize;
        let val = |i: usize| f32::from(s.q[r * BRICK_NODES + i]) * s.scale[r] + s.offset[r];
        let eps = eps_steps * s.scale[r].abs();
        for cx in 0..BRICK {
            for cy in 0..BRICK {
                for cz in 0..BRICK {
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    for d in 0..8 {
                        let i = ((cx + (d >> 2 & 1)) * NODES + cy + (d >> 1 & 1)) * NODES
                            + cz
                            + (d & 1);
                        let v = val(i);
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                    if lo <= eps && hi >= -eps {
                        for d in 0..8 {
                            let i = ((cx + (d >> 2 & 1)) * NODES + cy + (d >> 1 & 1)) * NODES
                                + cz
                                + (d & 1);
                            m[i / 32] |= 1 << (i % 32);
                        }
                    }
                }
            }
        }
    }
    let mut out = lv.clone();
    for ch in out.channels.iter_mut().skip(2) {
        let mut q = Vec::new();
        let mut starts = Vec::with_capacity(ch.scale.len());
        // rows are referenced by bricks; a row's start follows from the brick that owns it
        let mut owner = vec![usize::MAX; ch.scale.len()];
        for (k, &row) in ch.rows.iter().enumerate() {
            if row >= 0 {
                owner[row as usize] = k;
            }
        }
        for (r, &k) in owner.iter().enumerate() {
            if k == usize::MAX {
                return Err(SheetFieldError::Shape(format!(
                    "band: row {r} has no brick"
                )));
            }
            starts.push(q.len() as u32);
            let m = &mask[k * BAND_WORDS..(k + 1) * BAND_WORDS];
            for i in 0..BRICK_NODES {
                if m[i / 32] >> (i % 32) & 1 == 1 {
                    q.push(ch.q[r * BRICK_NODES + i]);
                }
            }
        }
        ch.q = q;
        ch.starts = starts;
        ch.nodes = BAND;
    }
    out.band = mask;
    Ok(out)
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
        for (c, ch) in lv.channels.iter().enumerate() {
            let fine = ch.nodes == NODES as u32;
            if c >= 2 && ch.nodes == BAND {
                let active = lv.channels[0].rows.len();
                if lv.band.len() != active * BAND_WORDS || ch.starts.len() != ch.scale.len() {
                    return Err(SheetFieldError::Shape(format!(
                        "band channel {c}: mask of {} words for {active} bricks, {} starts for {} rows",
                        lv.band.len(),
                        ch.starts.len(),
                        ch.scale.len()
                    )));
                }
                let mut want = 0u32;
                for (k, &row) in ch.rows.iter().enumerate() {
                    if row >= 0 {
                        let r = row as usize;
                        let n: u32 = lv.band[k * BAND_WORDS..(k + 1) * BAND_WORDS]
                            .iter()
                            .map(|x| x.count_ones())
                            .sum();
                        if ch.starts[r] as usize + n as usize > ch.q.len() {
                            return Err(SheetFieldError::Shape(format!(
                                "band channel {c}: row {r} runs past its bytes"
                            )));
                        }
                        want += n;
                    }
                }
                if want as usize != ch.q.len() || ch.offset.len() != ch.scale.len() {
                    return Err(SheetFieldError::Shape(format!(
                        "band channel {c}: {} bytes for {want} band nodes",
                        ch.q.len()
                    )));
                }
                continue;
            }
            if !(fine || (c >= 2 && COARSE_NODES.contains(&ch.nodes))) {
                return Err(SheetFieldError::Shape(format!(
                    "channel {c} on {} nodes per axis (s and d_b need {NODES}; attributes {NODES} or {COARSE_NODES:?})",
                    ch.nodes
                )));
            }
            let per = (ch.nodes as usize).pow(3);
            if ch.q.len() != ch.scale.len() * per || ch.offset.len() != ch.scale.len() {
                return Err(SheetFieldError::Shape(format!(
                    "channel {c}: {} bytes for {} rows of {per} nodes",
                    ch.q.len(),
                    ch.scale.len()
                )));
            }
        }
        let n = Vec3::new(lv.nb[0] as f32, lv.nb[1] as f32, lv.nb[2] as f32);
        lo = lo.min(lv.lo);
        hi = hi.max(lv.lo + n * (BRICK as f32 * lv.h));
        hc = hc.max(lv.h);
    }
    Ok(Sheet { levels, lo, hi, hc })
}
