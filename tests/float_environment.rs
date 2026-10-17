//! The floating-point environment the Law arithmetic assumes (ALICE-Law G14)
//!
//! Every value this crate fixes bit for bit assumes IEEE 754 binary32 /
//! binary64 with round-to-nearest-even and gradual underflow. A different
//! rounding mode, or a flush-to-zero / denormals-are-zero mode (x86 MXCSR
//! FTZ/DAZ, Arm FPCR.FZ), would change results without any source change, so
//! the two assumptions are asserted here, at run time, on every target the
//! tests run on, rather than relied on. `scripts/build_config_guard.py` checks
//! the build side (no fast-math-style flag, no target-cpu=native).
//!
//! `black_box` keeps each operation at run time: a constant-folded result
//! would come from the compiler, not from the processor's mode. The file is
//! generic and is kept byte-identical across the repositories that carry it.

use std::hint::black_box;

/// 2^k as f64, built from its bits (normal range only)
fn p2(k: i32) -> f64 {
    f64::from_bits(u64::try_from(1023 + k).unwrap() << 52)
}

/// 2^k as f32, built from its bits (normal range only)
fn p2f(k: i32) -> f32 {
    f32::from_bits(u32::try_from(127 + k).unwrap() << 23)
}

/// Ties go to the even neighbour, in both directions, and a non-tie goes to the nearer one
#[test]
fn round_to_nearest_even() {
    // f64: the spacing just above 1.0 is 2^-52, so 2^-53 is exactly half of it
    let one = black_box(1.0_f64);
    let half_ulp = black_box(p2(-53));
    assert_eq!(
        (one + half_ulp).to_bits(),
        1.0_f64.to_bits(),
        "tie at even 1.0 stays"
    );
    let odd = black_box(1.0_f64 + p2(-52));
    assert_eq!(
        (odd + half_ulp).to_bits(),
        (1.0_f64 + p2(-51)).to_bits(),
        "tie at odd rounds up to even"
    );
    assert_eq!(
        (one + black_box(p2(-53) + p2(-60))).to_bits(),
        (1.0_f64 + p2(-52)).to_bits(),
        "above the tie rounds up"
    );
    assert_eq!(
        (one - black_box(p2(-54))).to_bits(),
        1.0_f64.to_bits(),
        "tie just below 1.0 (spacing 2^-53) goes to the even 1.0"
    );
    assert_eq!(
        (black_box(-1.0_f64) - half_ulp).to_bits(),
        (-1.0_f64).to_bits(),
        "negative tie at even stays (not toward -infinity)"
    );
    assert_eq!(
        (black_box(1.0_f64) / black_box(3.0_f64)).to_bits(),
        0x3FD5_5555_5555_5555,
        "1/3 rounds to the nearer (down)"
    );
    assert_eq!(
        (black_box(2.0_f64) / black_box(3.0_f64)).to_bits(),
        0x3FE5_5555_5555_5555,
        "2/3 rounds to the nearer (down)"
    );

    // f32: the spacing just above 1.0 is 2^-23, half of it 2^-24
    let one = black_box(1.0_f32);
    let half_ulp = black_box(p2f(-24));
    assert_eq!(
        (one + half_ulp).to_bits(),
        1.0_f32.to_bits(),
        "f32 tie at even stays"
    );
    let odd = black_box(1.0_f32 + p2f(-23));
    assert_eq!(
        (odd + half_ulp).to_bits(),
        (1.0_f32 + p2f(-22)).to_bits(),
        "f32 tie at odd rounds up to even"
    );
    assert_eq!(
        (black_box(1.0_f32) / black_box(3.0_f32)).to_bits(),
        0x3EAA_AAAB,
        "f32 1/3 rounds to the nearer (up)"
    );
}

/// Subnormal results are produced (no flush-to-zero) and subnormal inputs are used
/// (no denormals-are-zero)
#[test]
fn subnormals_are_not_flushed() {
    // f64: an operation whose exact result is subnormal
    let tiny = black_box(f64::MIN_POSITIVE) / black_box(2.0_f64);
    assert!(
        tiny.is_subnormal(),
        "MIN_POSITIVE / 2 is subnormal, got {tiny:e}"
    );
    assert_eq!(tiny.to_bits(), 0x0008_0000_0000_0000);
    let smallest = f64::from_bits(1);
    // subnormal inputs take part in arithmetic
    assert_eq!(
        (black_box(smallest) * black_box(1.0_f64)).to_bits(),
        1,
        "x * 1 keeps a subnormal"
    );
    assert_eq!(
        (black_box(smallest) + black_box(smallest)).to_bits(),
        2,
        "subnormal + subnormal"
    );
    assert_eq!(
        (black_box(smallest) * black_box(p2(52))).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        "smallest subnormal * 2^52 = 2^-1022"
    );
    assert!(
        black_box(smallest) > black_box(0.0_f64),
        "a subnormal compares above zero"
    );
    // gradual underflow: x - y == 0 only when x == y
    let x = black_box(f64::MIN_POSITIVE * 1.5);
    let y = black_box(f64::MIN_POSITIVE);
    assert_ne!(
        (x - y).to_bits(),
        0.0_f64.to_bits(),
        "difference of distinct normals near the bottom is not 0"
    );

    // f32
    let tiny = black_box(f32::MIN_POSITIVE) / black_box(2.0_f32);
    assert!(
        tiny.is_subnormal(),
        "f32 MIN_POSITIVE / 2 is subnormal, got {tiny:e}"
    );
    assert_eq!(tiny.to_bits(), 0x0040_0000);
    let smallest = f32::from_bits(1);
    assert_eq!(
        (black_box(smallest) * black_box(1.0_f32)).to_bits(),
        1,
        "f32 x * 1 keeps a subnormal"
    );
    assert_eq!(
        (black_box(smallest) + black_box(smallest)).to_bits(),
        2,
        "f32 subnormal + subnormal"
    );
    assert!(
        black_box(smallest) > black_box(0.0_f32),
        "f32 subnormal compares above zero"
    );
}
