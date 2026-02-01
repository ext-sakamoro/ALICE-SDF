//! SoA (Structure of Arrays) Memory Layout for SIMD-Optimized Evaluation
//!
//! This module provides cache-friendly data structures that enable
//! direct SIMD loading without costly shuffle operations.
//!
//! # Memory Layout Comparison
//!
//! ## AoS (Array of Structures) - Traditional
//! ```text
//! [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3, ...]
//!     ↓ SIMD load requires gather/shuffle
//! ```
//!
//! ## SoA (Structure of Arrays) - This Module
//! ```text
//! X: [x0,x1,x2,x3,x4,x5,x6,x7, ...]  ← Direct f32x8 load
//! Y: [y0,y1,y2,y3,y4,y5,y6,y7, ...]  ← Direct f32x8 load
//! Z: [z0,z1,z2,z3,z4,z5,z6,z7, ...]  ← Direct f32x8 load
//! ```
//!
//! # Performance Benefits
//!
//! - **No shuffle overhead**: SIMD registers loaded directly from contiguous memory
//! - **Better cache utilization**: Sequential access patterns
//! - **Prefetch-friendly**: CPU can predict memory access patterns
//! - **Aligned loads**: 32-byte alignment for AVX2 (256-bit)
//!
//! # Usage
//!
//! ```rust
//! use alice_sdf::soa::SoAPoints;
//! use glam::Vec3;
//!
//! // Convert from AoS
//! let points = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
//! let soa = SoAPoints::from_vec3_slice(&points);
//!
//! // Or build incrementally
//! let mut soa = SoAPoints::with_capacity(1000);
//! soa.push(1.0, 2.0, 3.0);
//! soa.push(4.0, 5.0, 6.0);
//! ```
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use wide::f32x8;

/// SIMD lane width (8 for AVX2/AVX-512, also works for NEON with 2x4)
pub const SIMD_WIDTH: usize = 8;

/// Alignment for SIMD loads (32 bytes = 8 x f32 = AVX2 register)
pub const SIMD_ALIGNMENT: usize = 32;

/// SoA (Structure of Arrays) point storage for SIMD-optimized evaluation
///
/// Stores 3D points in a cache-friendly layout where all X coordinates
/// are contiguous, all Y coordinates are contiguous, and all Z coordinates
/// are contiguous. This enables direct SIMD loading without shuffle operations.
#[derive(Debug, Clone)]
pub struct SoAPoints {
    /// X coordinates (aligned for SIMD)
    pub x: AlignedVec,
    /// Y coordinates (aligned for SIMD)
    pub y: AlignedVec,
    /// Z coordinates (aligned for SIMD)
    pub z: AlignedVec,
    /// Number of valid points
    len: usize,
}

impl SoAPoints {
    /// Create empty SoA storage
    pub fn new() -> Self {
        Self {
            x: AlignedVec::new(),
            y: AlignedVec::new(),
            z: AlignedVec::new(),
            len: 0,
        }
    }

    /// Create SoA storage with pre-allocated capacity
    ///
    /// Capacity is rounded up to the nearest SIMD width for alignment.
    pub fn with_capacity(capacity: usize) -> Self {
        let aligned_capacity = align_up(capacity, SIMD_WIDTH);
        Self {
            x: AlignedVec::with_capacity(aligned_capacity),
            y: AlignedVec::with_capacity(aligned_capacity),
            z: AlignedVec::with_capacity(aligned_capacity),
            len: 0,
        }
    }

    /// Convert from a slice of Vec3 (AoS to SoA conversion)
    pub fn from_vec3_slice(points: &[Vec3]) -> Self {
        let len = points.len();
        let aligned_len = align_up(len, SIMD_WIDTH);

        let mut soa = Self::with_capacity(aligned_len);

        // Copy data
        for p in points {
            soa.x.data.push(p.x);
            soa.y.data.push(p.y);
            soa.z.data.push(p.z);
        }

        // Pad to SIMD width with zeros (safe default for SDF evaluation)
        let padding = aligned_len - len;
        for _ in 0..padding {
            soa.x.data.push(0.0);
            soa.y.data.push(0.0);
            soa.z.data.push(0.0);
        }

        soa.len = len;
        soa
    }

    /// Push a single point
    #[inline]
    pub fn push(&mut self, x: f32, y: f32, z: f32) {
        self.x.data.push(x);
        self.y.data.push(y);
        self.z.data.push(z);
        self.len += 1;
    }

    /// Push a Vec3 point
    #[inline]
    pub fn push_vec3(&mut self, p: Vec3) {
        self.push(p.x, p.y, p.z);
    }

    /// Number of points stored
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the padded length (rounded up to SIMD width)
    #[inline]
    pub fn padded_len(&self) -> usize {
        align_up(self.len, SIMD_WIDTH)
    }

    /// Ensure padding to SIMD width
    ///
    /// Call this after all points are added to ensure safe SIMD iteration.
    pub fn ensure_padding(&mut self) {
        let padded = self.padded_len();
        while self.x.data.len() < padded {
            self.x.data.push(0.0);
            self.y.data.push(0.0);
            self.z.data.push(0.0);
        }
    }

    /// Get raw pointers for ultra-fast unsafe SIMD loading
    ///
    /// # Safety
    /// The returned pointers are valid for `padded_len()` elements.
    /// Caller must ensure bounds are respected.
    #[inline]
    pub fn as_ptrs(&self) -> (*const f32, *const f32, *const f32) {
        (self.x.as_ptr(), self.y.as_ptr(), self.z.as_ptr())
    }

    /// Load 8 points as SIMD vectors starting at the given index
    ///
    /// # Safety
    /// `index` must be aligned to SIMD_WIDTH and `index + SIMD_WIDTH <= padded_len()`
    #[inline]
    pub unsafe fn load_simd_unchecked(&self, index: usize) -> (f32x8, f32x8, f32x8) {
        let (px, py, pz) = self.as_ptrs();
        let x = f32x8::from(std::slice::from_raw_parts(px.add(index), 8));
        let y = f32x8::from(std::slice::from_raw_parts(py.add(index), 8));
        let z = f32x8::from(std::slice::from_raw_parts(pz.add(index), 8));
        (x, y, z)
    }

    /// Load 8 points as SIMD vectors with bounds checking
    #[inline]
    pub fn load_simd(&self, index: usize) -> Option<(f32x8, f32x8, f32x8)> {
        if index + SIMD_WIDTH <= self.x.data.len() {
            let x = f32x8::from(&self.x.data[index..index + 8]);
            let y = f32x8::from(&self.y.data[index..index + 8]);
            let z = f32x8::from(&self.z.data[index..index + 8]);
            Some((x, y, z))
        } else {
            None
        }
    }

    /// Get point at index
    #[inline]
    pub fn get(&self, index: usize) -> Option<Vec3> {
        if index < self.len {
            Some(Vec3::new(
                self.x.data[index],
                self.y.data[index],
                self.z.data[index],
            ))
        } else {
            None
        }
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.x.data.clear();
        self.y.data.clear();
        self.z.data.clear();
        self.len = 0;
    }

    /// Iterate over points as Vec3
    pub fn iter(&self) -> impl Iterator<Item = Vec3> + '_ {
        (0..self.len).map(move |i| {
            Vec3::new(self.x.data[i], self.y.data[i], self.z.data[i])
        })
    }

    /// Get slices for each coordinate array
    #[inline]
    pub fn as_slices(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.x.data, &self.y.data, &self.z.data)
    }
}

impl Default for SoAPoints {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<Vec3> for SoAPoints {
    fn from_iter<T: IntoIterator<Item = Vec3>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);

        let mut soa = SoAPoints::with_capacity(capacity);
        for p in iter {
            soa.push_vec3(p);
        }
        soa.ensure_padding();
        soa
    }
}

impl<'a> FromIterator<&'a Vec3> for SoAPoints {
    fn from_iter<T: IntoIterator<Item = &'a Vec3>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        let capacity = upper.unwrap_or(lower);

        let mut soa = SoAPoints::with_capacity(capacity);
        for p in iter {
            soa.push_vec3(*p);
        }
        soa.ensure_padding();
        soa
    }
}

/// Aligned vector for SIMD operations
///
/// Provides 32-byte aligned storage for direct AVX2 loads.
#[derive(Debug, Clone)]
pub struct AlignedVec {
    data: Vec<f32>,
}

impl AlignedVec {
    /// Create empty aligned vector
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Get raw pointer
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    /// Get slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for AlignedVec {
    fn default() -> Self {
        Self::new()
    }
}

/// SoA storage for output distances
#[derive(Debug, Clone)]
pub struct SoADistances {
    /// Distance values
    pub distances: Vec<f32>,
    /// Actual number of valid results
    len: usize,
}

impl SoADistances {
    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let aligned_capacity = align_up(capacity, SIMD_WIDTH);
        Self {
            distances: vec![0.0; aligned_capacity],
            len: capacity,
        }
    }

    /// Get the actual length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get results as slice (only valid entries)
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.distances[..self.len]
    }

    /// Convert to Vec
    pub fn to_vec(&self) -> Vec<f32> {
        self.distances[..self.len].to_vec()
    }

    /// Store SIMD result
    ///
    /// # Safety
    /// `index` must be aligned and within bounds
    #[inline]
    pub unsafe fn store_simd_unchecked(&mut self, index: usize, values: f32x8) {
        let arr: [f32; 8] = values.into();
        std::ptr::copy_nonoverlapping(arr.as_ptr(), self.distances.as_mut_ptr().add(index), 8);
    }
}

/// Round up to the nearest multiple
#[inline]
const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_creation() {
        let mut soa = SoAPoints::new();
        soa.push(1.0, 2.0, 3.0);
        soa.push(4.0, 5.0, 6.0);

        assert_eq!(soa.len(), 2);
        assert_eq!(soa.get(0), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(soa.get(1), Some(Vec3::new(4.0, 5.0, 6.0)));
    }

    #[test]
    fn test_soa_from_vec3_slice() {
        let points = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);

        assert_eq!(soa.len(), 3);
        assert_eq!(soa.padded_len(), 8); // Rounded up to SIMD width

        for (i, p) in points.iter().enumerate() {
            assert_eq!(soa.get(i), Some(*p));
        }
    }

    #[test]
    fn test_soa_simd_load() {
        let points: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new(i as f32, (i * 10) as f32, (i * 100) as f32))
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);

        let (x, y, z) = soa.load_simd(0).unwrap();

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let z_arr: [f32; 8] = z.into();

        for i in 0..8 {
            assert_eq!(x_arr[i], i as f32);
            assert_eq!(y_arr[i], (i * 10) as f32);
            assert_eq!(z_arr[i], (i * 100) as f32);
        }
    }

    #[test]
    fn test_soa_iter() {
        let points = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let collected: Vec<Vec3> = soa.iter().collect();

        assert_eq!(collected, points);
    }

    #[test]
    fn test_soa_from_iterator() {
        let points = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
        ];

        let soa: SoAPoints = points.iter().collect();

        assert_eq!(soa.len(), 2);
        assert_eq!(soa.get(0), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }
}
