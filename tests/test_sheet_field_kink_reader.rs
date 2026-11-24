//! The ASKC reader on corrupt input: a count, a cell range, a brick id or a value the file cannot back is an `Err`,
//! never an abort, a panic or a large allocation.
//!
//! Oracles (none is the reader's own output):
//! - memory: a counting global allocator records the peak bytes requested while the reader runs; a file of a few
//!   dozen bytes that claims 2^28 - 1 records of any kind must stay under a fixed bound (the bytes the input can back,
//!   with slack), where sizing from the count would request gigabytes;
//! - a brick record whose first cell is near `u32::MAX` is refused (a u32 add would wrap past the range check);
//! - a brick id near 2^28 is read without a table of that size and refused at `attach` against the level's active
//!   bricks;
//! - a non-finite slab, vertex, normal or attribute, a slab <= 0, and an origin that is not a whole number of mm
//!   under 2^24 are refused at read (the GPU reads the origin as f32).
//!
//! Every test takes one lock, so the allocator's peak is this test's own.

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::kink::{KinkError, KinkFile};
use alice_sdf::sheet_field::kink_build::write_askc;
use glam::Vec3;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to the system allocator with the same layout; the counters only observe
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let now = CURRENT.fetch_add(l.size(), Ordering::SeqCst) + l.size();
        PEAK.fetch_max(now, Ordering::SeqCst);
        // SAFETY: same layout as asked
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let now = CURRENT.fetch_add(l.size(), Ordering::SeqCst) + l.size();
        PEAK.fetch_max(now, Ordering::SeqCst);
        // SAFETY: same layout as asked
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CURRENT.fetch_sub(l.size(), Ordering::SeqCst);
        // SAFETY: p came from this allocator with this layout
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

static LOCK: Mutex<()> = Mutex::new(());

/// The peak bytes requested above the start while `f` runs.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// What a reader may request for a file of a few hundred bytes: far under the gigabytes a count of 2^28 would size.
const SMALL: usize = 1 << 20;

const HUGE: u32 = (1 << 28) - 1;

/// magic, version and identifier of a sidecar this crate writes, then `ns` sheets
fn header(ns: u32) -> Vec<u8> {
    let mut own = Vec::new();
    write_askc(&mut own, &[(None, 0.01, 0)]).unwrap();
    let mut b = own[..40].to_vec();
    b.extend(ns.to_le_bytes());
    b
}

fn put_u32(b: &mut Vec<u8>, x: u32) {
    b.extend(x.to_le_bytes());
}

fn put_f32(b: &mut Vec<u8>, x: f32) {
    b.extend(x.to_le_bytes());
}

/// slab, origin and attribute flags of one sheet record
fn sheet_head(slab: f32, origin: [f64; 3], attr: u32) -> Vec<u8> {
    let mut b = header(1);
    put_f32(&mut b, slab);
    for x in origin {
        b.extend(x.to_le_bytes());
    }
    put_u32(&mut b, attr);
    b
}

/// a v3 vertex record: position, normal, rim flag
fn put_vertex(b: &mut Vec<u8>, p: [f32; 3], n: [f32; 3]) {
    for x in p.iter().chain(&n) {
        put_f32(b, *x);
    }
    put_u32(b, 0);
}

/// One sheet with `nv` vertices, no faces or segments, the given kink brick records on one level, `ncell` cell
/// records of one triangle-free kind and no refs.
fn one_level(bricks: &[(u32, [u32; 16], u32)], ncell: u32) -> Vec<u8> {
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    put_u32(&mut b, 0); // vertices
    put_u32(&mut b, 0); // faces
    put_u32(&mut b, 0); // segments
    put_u32(&mut b, 1); // levels
    put_u32(&mut b, bricks.len() as u32);
    for (brick, m, first) in bricks {
        put_u32(&mut b, *brick);
        for w in m {
            put_u32(&mut b, *w);
        }
        put_u32(&mut b, *first);
    }
    put_u32(&mut b, ncell);
    for _ in 0..ncell {
        put_u32(&mut b, 0);
        b.extend(0u16.to_le_bytes());
        b.extend(0u16.to_le_bytes());
        put_u32(&mut b, 1);
    }
    put_u32(&mut b, 0); // refs
    b
}

fn read(bytes: &[u8]) -> Result<KinkFile, KinkError> {
    KinkFile::read(&mut &bytes[..])
}

fn format_error(r: Result<KinkFile, KinkError>, want: &str) {
    match r {
        Err(KinkError::Format(m)) => assert!(m.contains(want), "{m} (wanted {want})"),
        Err(e) => panic!("another error: {e}"),
        Ok(_) => panic!("read ({want})"),
    }
}

/// Each count of the format set to 2^28 - 1 in a file that ends right after it: the read ends at the input's end
/// with an error, and asks for no more memory than a small file can back.
#[test]
fn a_count_the_file_cannot_back_allocates_nothing_large() {
    let _g = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    // sheets
    cases.push(("sheets", header(HUGE)));
    // vertices
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    put_u32(&mut b, HUGE);
    cases.push(("vertices", b));
    // faces after one vertex
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    put_u32(&mut b, 1);
    put_vertex(&mut b, [0.0; 3], [0.0, 0.0, 1.0]);
    put_u32(&mut b, HUGE);
    cases.push(("faces", b));
    // segments
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    put_u32(&mut b, 0);
    put_u32(&mut b, 0);
    put_u32(&mut b, HUGE);
    cases.push(("segments", b));
    // kink bricks of one level
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    for n in [0, 0, 0, 1, HUGE] {
        put_u32(&mut b, n);
    }
    cases.push(("kink bricks", b));
    // cell records, then refs
    let full = one_level(&[], 0);
    let at_cells = full.len() - 8;
    let mut b = full[..at_cells].to_vec();
    put_u32(&mut b, HUGE);
    cases.push(("cell records", b));
    let mut b = full[..at_cells + 4].to_vec();
    put_u32(&mut b, HUGE);
    cases.push(("refs", b));
    // counts after records already read: the growth path that holds some records is covered too
    let mut b = sheet_head(0.5, [0.0; 3], 0);
    put_u32(&mut b, 3);
    for p in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        put_vertex(&mut b, p, [0.0, 0.0, 1.0]);
    }
    put_u32(&mut b, HUGE);
    cases.push(("faces after 3 vertices", b));
    let two = one_level(&[], 2);
    let mut b = two[..two.len() - 4].to_vec();
    put_u32(&mut b, HUGE);
    cases.push(("refs after 2 cell records", b));
    let mut compared = 0;
    for (what, bytes) in &cases {
        assert!(bytes.len() < 256, "{what}: {} bytes", bytes.len());
        let (r, peak) = peak_during(|| read(bytes));
        match r {
            Err(KinkError::Io(_) | KinkError::Format(_)) => {}
            Err(e) => panic!("{what}: another error: {e}"),
            Ok(_) => panic!("{what}: a truncated file was read"),
        }
        assert!(
            peak < SMALL,
            "{what}: the reader asked for {peak} bytes on a {}-byte file",
            bytes.len()
        );
        compared += 1;
    }
    assert_eq!(compared, 9);
}

/// A brick record whose first cell is near u32::MAX: the range check sums in u64 and refuses it (in u32 the sum
/// wraps past the check in release and panics in debug).
#[test]
fn a_cell_range_past_u32_is_refused_not_wrapped() {
    let _g = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut m = [0u32; 16];
    m[0] = 0xFFFF; // 16 cells
    for first in [u32::MAX, 0xFFFF_FFF0, 0xFFFF_FFF1] {
        format_error(read(&one_level(&[(0, m, first)], 4)), "cells");
    }
    // the same record inside the cells is read
    assert!(read(&one_level(&[(0, m, 0)], 16)).is_ok());
}

/// A brick id near 2^28 costs no table of that size at read, and `attach` refuses it against the level's active
/// bricks; the same file with an id inside them attaches. Each file holds one cell record: a sheet record with none
/// reads as a sheet without kink cells, which `attach` skips.
#[test]
fn a_large_brick_id_is_read_sparse_and_refused_at_attach() {
    let _g = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let lv = level_from_fn(
        Vec3::new(-1.013, -1.021, -1.007),
        0.25,
        [2, 1, 1],
        |_c, _e| true,
        |p| vec![p.z, p.x.abs().max(p.y.abs()) - 0.5],
    )
    .unwrap();
    let active = lv.index.iter().filter(|&&x| x >= 0).count();
    assert!(active >= 1);
    let m = [0u32; 16];
    let (r, peak) = peak_during(|| read(&one_level(&[(HUGE, m, 0)], 1)));
    let kf = r.expect("a large id is read");
    assert!(peak < SMALL, "the reader asked for {peak} bytes");
    let mut sh = vec![sheet(vec![lv.clone()]).unwrap()];
    match kf.attach(&mut sh, 0) {
        Err(KinkError::Format(msg)) => assert!(msg.contains(&HUGE.to_string()), "{msg}"),
        other => panic!("attached: {other:?}"),
    }
    let kf = read(&one_level(&[(active as u32 - 1, m, 0)], 1)).unwrap();
    let mut sh = vec![sheet(vec![lv]).unwrap()];
    kf.attach(&mut sh, 0)
        .expect("an id inside the active bricks");
    // two records of one brick, in either order, are refused
    format_error(read(&one_level(&[(3, m, 0), (3, m, 0)], 0)), "twice");
    format_error(
        read(&one_level(&[(5, m, 0), (2, m, 0), (5, m, 0)], 0)),
        "twice",
    );
}

/// Non-finite values and a slab <= 0 are refused at read.
#[test]
fn non_finite_values_and_a_bad_slab_are_refused() {
    let _g = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for slab in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        format_error(read(&sheet_head(slab, [0.0; 3], 0)), "slab");
    }
    let vertex = |p: [f32; 3], n: [f32; 3]| {
        let mut b = sheet_head(0.5, [0.0; 3], 0);
        put_u32(&mut b, 1);
        put_vertex(&mut b, p, n);
        b
    };
    format_error(
        read(&vertex([f32::NAN, 0.0, 0.0], [0.0, 0.0, 1.0])),
        "vertex",
    );
    format_error(
        read(&vertex([0.0, f32::INFINITY, 0.0], [0.0, 0.0, 1.0])),
        "vertex",
    );
    format_error(
        read(&vertex([0.0; 3], [0.0, f32::NAN, 1.0])),
        "vertex normal",
    );
    // a face with a NaN corner UV (attribute bit 0)
    let mut b = sheet_head(0.5, [0.0; 3], 1);
    put_u32(&mut b, 3);
    for p in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        put_vertex(&mut b, p, [0.0, 0.0, 1.0]);
    }
    put_u32(&mut b, 1);
    for i in 0..3 {
        put_u32(&mut b, i);
    }
    for x in [0.0, 0.0, 1.0] {
        put_f32(&mut b, x);
    }
    put_u32(&mut b, 0);
    for _ in 0..3 {
        for x in [0.0, 0.0, 1.0] {
            put_f32(&mut b, x);
        }
    }
    for x in [0.0, 0.0, 1.0, f32::NAN, 0.0, 1.0] {
        put_f32(&mut b, x);
    }
    format_error(read(&b), "corner uv");
}

/// The origin must be finite whole numbers of mm under 2^24 (exact in f32, the GPU's local point is the CPU's).
#[test]
fn an_origin_the_gpu_cannot_read_exactly_is_refused() {
    let _g = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for o in [
        [0.5, 0.0, 0.0],
        [0.0, -1e-9, 0.0],
        [16_777_216.0, 0.0, 0.0],
        [0.0, 0.0, -16_777_216.0],
        [f64::NAN, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
    ] {
        format_error(read(&sheet_head(0.5, o, 0)), "origin");
    }
    // whole numbers just inside the range read on (to the input's end)
    for o in [[16_777_215.0, -16_777_215.0, 1360.0], [0.0; 3]] {
        match read(&sheet_head(0.5, o, 0)) {
            Err(KinkError::Io(_)) => {}
            other => panic!("{o:?}: {other:?}"),
        }
    }
}
