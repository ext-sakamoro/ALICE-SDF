//! A count read from a buffer or a stream must not decide how much memory the decoder reserves: the
//! mesh codec's triangle / position count and the NSDF stream's layer count and layer dimensions are
//! checked against what the input can still hold before anything is allocated for them, and a
//! dimension product that does not fit is an error, not a panic.
//! Measured with a capped global allocator: the peak of live bytes while the decoder runs, and the
//! largest single request, which may not exceed 16 MiB (a request above it is refused, not made).

use alice_sdf::mesh::mesh_codec::{
    decode_indices, decode_positions, encode_indices, encode_positions,
};
use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Largest single request the decoders may make in this binary. A request above it is refused
/// (null), so a decoder that sizes a buffer by a claimed count stops the process with
/// "memory allocation of N bytes failed" before anything that large is reserved, the same on every OS.
const CAP: usize = 16 * 1024 * 1024;

struct Capped;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

fn admit(size: usize) -> bool {
    LARGEST.fetch_max(size, Ordering::SeqCst);
    size <= CAP
}

fn note(size: usize) {
    let now = LIVE.fetch_add(size, Ordering::SeqCst) + size;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !admit(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            note(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

/// The tests of this binary measure one process-wide counter, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

/// Peak bytes allocated above the starting level while `f` runs; also checks that no single request
/// went over the cap (one that did was refused, so reaching this check means it was not made).
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    LARGEST.store(0, Ordering::SeqCst);
    let out = f();
    let largest = LARGEST.load(Ordering::SeqCst);
    assert!(largest <= CAP, "a single request of {largest} bytes");
    (out, PEAK.load(Ordering::SeqCst) - base)
}

/// Fixed cost of a decoder apart from what it decodes.
const SLACK: usize = 64 * 1024;

fn varint(mut v: u32) -> Vec<u8> {
    let mut out = Vec::new();
    while v >= 0x80 {
        out.push(((v & 0x7f) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
    out
}

/// A valid empty buffer of the codec (header + count 0) with the count replaced by `count`.
fn with_count(empty: Vec<u8>, count: u32) -> Vec<u8> {
    assert_eq!(empty.len(), 41, "40-byte header and the 1-byte count 0");
    assert_eq!(empty[40], 0);
    let mut b = empty[..40].to_vec();
    b.extend_from_slice(&varint(count));
    b.extend_from_slice(&[0x02, 0x02, 0x02]);
    b
}

#[test]
fn a_claimed_mesh_count_is_not_allocated() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut compared = 0;
    // each claim asks for gigabytes of decoded values from a buffer of under 50 bytes
    let indices = with_count(encode_indices(&[]), 0x1555_5555);
    let (res, peak) = peak_during(|| decode_indices(&indices));
    assert!(
        res.is_err(),
        "indices: a count the buffer cannot hold is an error"
    );
    assert!(
        peak <= 64 * indices.len() + SLACK,
        "indices: decoding reserved {peak} bytes for a {}-byte buffer",
        indices.len()
    );
    compared += 1;

    let positions = with_count(encode_positions(&[]), 0x0555_5555);
    let (res, peak) = peak_during(|| decode_positions(&positions));
    assert!(
        res.is_err(),
        "positions: a count the buffer cannot hold is an error"
    );
    assert!(
        peak <= 64 * positions.len() + SLACK,
        "positions: decoding reserved {peak} bytes for a {}-byte buffer",
        positions.len()
    );
    compared += 1;
    assert_eq!(compared, 2);
}

/// A saved network with one layer, and the offset of its layer count (after the magic, the version,
/// the 32-byte identifier and the frequency count).
fn saved_network() -> (Vec<u8>, usize) {
    let net = NeuralSdf::new(&NeuralSdfConfig {
        hidden_layers: 0,
        hidden_width: 1,
        pos_encoding_freqs: 0,
        ..Default::default()
    });
    let mut bytes = Vec::new();
    net.save(&mut bytes).unwrap();
    let at = 4 + 4 + 32 + 4;
    assert_eq!(&bytes[at - 4..at], &0_u32.to_le_bytes(), "frequency count");
    let layers = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert!(layers >= 1, "the saved network has a layer");
    (bytes, at)
}

#[test]
fn a_claimed_network_shape_is_not_allocated() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (saved, at) = saved_network();
    // a layer count, then one layer's (in, out): a weight matrix of 2^28 floats, a bias of 2^28 floats
    let cases: [(&str, u32, Option<(u32, u32)>); 3] = [
        ("layer count", 0x0100_0000, None),
        ("weight matrix", 1, Some((0x4000, 0x4000))),
        ("bias", 1, Some((0, 0x1000_0000))),
    ];
    let mut compared = 0;
    for (what, layers, dims) in cases {
        let mut b = saved[..at].to_vec();
        b.extend_from_slice(&layers.to_le_bytes());
        if let Some((i, o)) = dims {
            b.extend_from_slice(&i.to_le_bytes());
            b.extend_from_slice(&o.to_le_bytes());
        }
        b.extend_from_slice(&[0; 8]);
        let (res, peak) = peak_during(|| NeuralSdf::load(&mut b.as_slice()));
        assert!(res.is_err(), "{what}: a stream cut short is an error");
        assert!(
            peak <= 64 * b.len() + SLACK,
            "{what}: loading reserved {peak} bytes for a {}-byte stream",
            b.len()
        );
        compared += 1;
    }
    assert_eq!(compared, 3);
}

#[test]
fn a_layer_whose_dimensions_overflow_is_an_error_not_a_panic() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (saved, at) = saved_network();
    let mut b = saved[..at].to_vec();
    b.extend_from_slice(&1_u32.to_le_bytes());
    // in · out = (2^32 - 1)^2 does not fit a 64-bit usize
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    b.extend_from_slice(&u32::MAX.to_le_bytes());
    let out = catch_unwind(AssertUnwindSafe(|| {
        NeuralSdf::load(&mut b.as_slice()).is_err()
    }));
    assert_eq!(
        out.ok(),
        Some(true),
        "the overflowing shape is an Err, not a panic"
    );
}
