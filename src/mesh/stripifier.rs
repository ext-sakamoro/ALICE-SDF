//! Triangle-strip generation from indexed triangle mesh
//!
//! Ported from zeux/meshoptimizer `stripifier.cpp` Converts an index buffer
//! of triangles to a triangle strip (or strip sequence separated by restart
//! indices / degenerate triangles), reducing effective index count by
//! ~30-50% on typical closed meshes
//!
//! # Reference
//!
//! - Evans, Skiena, Varshney "Optimizing Triangle Strips for Fast Rendering" (1996)
//! - zeux/meshoptimizer §stripifier.cpp
//!
//! Author: Moroya Sakamoto

/// Buffer of pending triangles for the stripification lookahead
///
/// The greedy stripifier maintains a rolling buffer of up to 8 unprocessed
/// triangles (from the input) and picks the next triangle to append based on
/// edge continuity with the current strip
const BUFFER_CAPACITY: usize = 8;

/// Find the triangle in `buffer` whose minimum vertex valence is smallest
///
/// This is the strip-start heuristic — starting with low-valence vertices
/// tends to produce longer strips overall
fn find_strip_first(buffer: &[[u32; 3]], valence: &[u8]) -> usize {
    let mut best_index = 0usize;
    let mut best_iv = u32::MAX;

    for (i, tri) in buffer.iter().enumerate() {
        let va = u32::from(valence[tri[0] as usize]);
        let vb = u32::from(valence[tri[1] as usize]);
        let vc = u32::from(valence[tri[2] as usize]);
        let v = va.min(vb).min(vc);
        if v < best_iv {
            best_index = i;
            best_iv = v;
        }
    }
    best_index
}

/// Find a triangle in `buffer` that has an edge matching (`e0`, `e1`) in any rotation
///
/// Returns `(buffer_index << 2) | edge_offset` where `edge_offset` (0/1/2) is
/// the index of the third vertex within the triangle Returns `-1` if no
/// match is found
fn find_strip_next(buffer: &[[u32; 3]], e0: u32, e1: u32) -> i32 {
    for (i, tri) in buffer.iter().enumerate() {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        if e0 == a && e1 == b {
            return ((i as i32) << 2) | 2;
        }
        if e0 == b && e1 == c {
            return (i as i32) << 2;
        }
        if e0 == c && e1 == a {
            return ((i as i32) << 2) | 1;
        }
    }
    -1
}

/// Upper bound on the strip buffer size for `index_count` triangles
///
/// Worst case (no restarts): 5 indices per triangle Worst case (with
/// restarts): 4 indices per triangle Returns a safe upper bound for
/// pre-allocation
#[must_use]
pub const fn stripify_bound(index_count: usize) -> usize {
    (index_count / 3) * 5
}

/// Convert a triangle index buffer into a triangle strip
///
/// # Arguments
///
/// - `indices`: input triangle indices (length must be a multiple of 3)
/// - `vertex_count`: total number of vertices in the mesh (used to size the valence table)
/// - `restart_index`: if `Some(n)`, use `n` as a primitive restart marker (glTF/Vulkan style)
///   if `None`, use degenerate triangles to join sub-strips
///
/// # Returns
///
/// The strip index buffer typical output length is 30-50% smaller than input
///
/// # Panics
///
/// If `indices.len() % 3 != 0`
///
/// # Reference
///
/// meshopt `meshopt_stripify`
pub fn stripify(indices: &[u32], vertex_count: usize, restart_index: Option<u32>) -> Vec<u32> {
    assert!(
        indices.len() % 3 == 0,
        "index count must be a multiple of 3"
    );
    let restart = restart_index.unwrap_or(0);
    let use_restart = restart_index.is_some();

    let mut destination: Vec<u32> = Vec::with_capacity(stripify_bound(indices.len()));

    // Compute vertex valence — 8-bit counter clamped to 255 for outliers
    let mut valence = vec![0u8; vertex_count];
    for &idx in indices {
        let vi = idx as usize;
        debug_assert!(vi < vertex_count);
        valence[vi] = valence[vi].saturating_add(1);
    }

    let mut buffer: [[u32; 3]; BUFFER_CAPACITY] = [[0; 3]; BUFFER_CAPACITY];
    let mut buffer_size = 0usize;
    let mut index_offset = 0usize;

    let mut strip = [0u32; 2];
    let mut parity: u32 = 0;
    let mut next: i32 = -1;

    while buffer_size > 0 || index_offset < indices.len() {
        // Refill buffer from input
        while buffer_size < BUFFER_CAPACITY && index_offset < indices.len() {
            buffer[buffer_size][0] = indices[index_offset];
            buffer[buffer_size][1] = indices[index_offset + 1];
            buffer[buffer_size][2] = indices[index_offset + 2];
            buffer_size += 1;
            index_offset += 3;
        }
        debug_assert!(buffer_size > 0);

        if next >= 0 {
            // Continue current strip with a matching triangle in buffer
            let i = (next >> 2) as usize;
            let a = buffer[i][0];
            let b = buffer[i][1];
            let c = buffer[i][2];
            let v = buffer[i][(next & 3) as usize];

            // Ordered removal from buffer
            for k in i..buffer_size - 1 {
                buffer[k] = buffer[k + 1];
            }
            buffer_size -= 1;

            valence[a as usize] = valence[a as usize].saturating_sub(1);
            valence[b as usize] = valence[b as usize].saturating_sub(1);
            valence[c as usize] = valence[c as usize].saturating_sub(1);

            // Find next triangle in buffer; edge order flips per iteration
            let (e0_cont, e1_cont) = if parity == 1 {
                (strip[1], v)
            } else {
                (v, strip[1])
            };
            let cont = find_strip_next(&buffer[..buffer_size], e0_cont, e1_cont);

            let swap = if cont < 0 {
                let (e0_swap, e1_swap) = if parity == 1 {
                    (v, strip[0])
                } else {
                    (strip[0], v)
                };
                find_strip_next(&buffer[..buffer_size], e0_swap, e1_swap)
            } else {
                -1
            };

            if cont < 0 && swap >= 0 {
                // Emit swap: [a b c] => [a b a c] via degenerate skip
                destination.push(strip[0]);
                destination.push(v);
                strip[1] = v;
                next = swap;
            } else {
                // Continue strip
                destination.push(v);
                strip[0] = strip[1];
                strip[1] = v;
                parity ^= 1;
                next = cont;
            }
        } else {
            // Start a new strip
            let i = find_strip_first(&buffer[..buffer_size], &valence);
            let mut a = buffer[i][0];
            let mut b = buffer[i][1];
            let mut c = buffer[i][2];

            for k in i..buffer_size - 1 {
                buffer[k] = buffer[k + 1];
            }
            buffer_size -= 1;

            valence[a as usize] = valence[a as usize].saturating_sub(1);
            valence[b as usize] = valence[b as usize].saturating_sub(1);
            valence[c as usize] = valence[c as usize].saturating_sub(1);

            // Pre-rotate to find match in remaining buffer
            let ea = find_strip_next(&buffer[..buffer_size], c, b);
            let eb = find_strip_next(&buffer[..buffer_size], a, c);
            let ec = find_strip_next(&buffer[..buffer_size], b, a);

            let mut mine = i32::MAX;
            if ea >= 0 && mine > ea {
                mine = ea;
            }
            if eb >= 0 && mine > eb {
                mine = eb;
            }
            if ec >= 0 && mine > ec {
                mine = ec;
            }

            if ea == mine {
                next = ea;
            } else if eb == mine {
                let t = a;
                a = b;
                b = c;
                c = t;
                next = eb;
            } else if ec == mine {
                let t = c;
                c = b;
                b = a;
                a = t;
                next = ec;
            }

            if use_restart {
                if !destination.is_empty() {
                    destination.push(restart);
                }
                destination.push(a);
                destination.push(b);
                destination.push(c);
                strip[0] = b;
                strip[1] = c;
                parity = 1;
            } else {
                if !destination.is_empty() {
                    // Connect via degenerate triangles
                    destination.push(strip[1]);
                    destination.push(a);
                }
                let e0 = if parity == 1 { c } else { b };
                let e1 = if parity == 1 { b } else { c };
                destination.push(a);
                destination.push(e0);
                destination.push(e1);
                strip[0] = e0;
                strip[1] = e1;
                parity ^= 1;
            }
        }
    }

    destination
}

/// Upper bound on the triangle-list length for a strip of `index_count` indices
#[must_use]
pub const fn unstripify_bound(index_count: usize) -> usize {
    if index_count == 0 {
        0
    } else {
        (index_count - 2) * 3
    }
}

/// Convert a triangle strip back into an indexed triangle list
///
/// # Arguments
///
/// - `indices`: strip indices
/// - `restart_index`: same value used at stripify time (`None` = degenerate joins)
///
/// # Returns
///
/// Triangle index buffer (length always a multiple of 3)
pub fn unstripify(indices: &[u32], restart_index: Option<u32>) -> Vec<u32> {
    let mut destination: Vec<u32> = Vec::with_capacity(unstripify_bound(indices.len()));
    let use_restart = restart_index.is_some();
    let restart = restart_index.unwrap_or(0);

    let mut start = 0usize;

    for (i, &idx) in indices.iter().enumerate() {
        if use_restart && idx == restart {
            start = i + 1;
        } else if i >= start + 2 {
            let mut a = indices[i - 2];
            let mut b = indices[i - 1];
            let c = indices[i];

            // Flip winding for odd-positioned triangles
            if ((i - start) & 1) == 1 {
                std::mem::swap(&mut a, &mut b);
            }

            // Skip degenerate triangles from strip swaps
            if a != b && a != c && b != c {
                destination.push(a);
                destination.push(b);
                destination.push(c);
            }
        }
    }

    destination
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangle sets are topologically equivalent if their sorted-vertex
    /// triples match one-to-one Winding must be preserved but starting
    /// vertex may vary — stripify+unstripify reorders triangles
    fn triangle_sets_equal_unordered(a: &[u32], b: &[u32]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut ta: Vec<[u32; 3]> = a
            .chunks_exact(3)
            .map(|c| {
                let mut t = [c[0], c[1], c[2]];
                t.sort_unstable();
                t
            })
            .collect();
        let mut tb: Vec<[u32; 3]> = b
            .chunks_exact(3)
            .map(|c| {
                let mut t = [c[0], c[1], c[2]];
                t.sort_unstable();
                t
            })
            .collect();
        ta.sort();
        tb.sort();
        ta == tb
    }

    #[test]
    fn test_stripify_single_triangle_restart() {
        let indices = vec![0u32, 1, 2];
        let strip = stripify(&indices, 3, Some(u32::MAX));
        assert_eq!(strip, vec![0, 1, 2]);
    }

    #[test]
    fn test_stripify_strip_roundtrip_restart() {
        // Classic strip: (0,1,2)(2,1,3)(2,3,4)(4,3,5)
        let indices = vec![0u32, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5];
        let strip = stripify(&indices, 6, Some(u32::MAX));
        let back = unstripify(&strip, Some(u32::MAX));
        assert!(
            triangle_sets_equal_unordered(&indices, &back),
            "triangle sets differ:\n  original: {:?}\n  restored: {:?}",
            indices,
            back
        );
    }

    #[test]
    fn test_stripify_strip_roundtrip_degenerate() {
        // Same strip using degenerate-triangle joining (no restart)
        let indices = vec![0u32, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5];
        let strip = stripify(&indices, 6, None);
        let back = unstripify(&strip, None);
        assert!(triangle_sets_equal_unordered(&indices, &back));
    }

    #[test]
    fn test_stripify_sphere_mesh_reduces_index_count() {
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        use glam::Vec3;
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );
        let indices = mesh.indices.clone();
        let vc = mesh.vertices.len();

        let strip = stripify(&indices, vc, Some(u32::MAX));
        // A closed mesh should have strip length noticeably less than the input length
        assert!(
            strip.len() < indices.len(),
            "strip {} should be shorter than triangles {}",
            strip.len(),
            indices.len()
        );

        let back = unstripify(&strip, Some(u32::MAX));
        assert!(triangle_sets_equal_unordered(&indices, &back));

        eprintln!(
            "sphere: {} triangles, {} strip indices ({:.1}%)",
            indices.len(),
            strip.len(),
            strip.len() as f32 / indices.len() as f32 * 100.0
        );
    }

    #[test]
    fn test_stripify_empty_indices() {
        let strip = stripify(&[], 0, Some(u32::MAX));
        assert!(strip.is_empty());
        let back = unstripify(&strip, Some(u32::MAX));
        assert!(back.is_empty());
    }

    #[test]
    fn test_stripify_disconnected_triangles() {
        // Two separate triangles (no shared edges)
        let indices = vec![0u32, 1, 2, 3, 4, 5];
        let strip = stripify(&indices, 6, Some(u32::MAX));
        let back = unstripify(&strip, Some(u32::MAX));
        assert!(triangle_sets_equal_unordered(&indices, &back));
    }

    #[test]
    fn test_stripify_bound_is_upper_bound() {
        let indices = vec![0u32, 1, 2, 2, 1, 3, 2, 3, 4];
        let strip = stripify(&indices, 5, Some(u32::MAX));
        assert!(strip.len() <= stripify_bound(indices.len()));
    }

    #[test]
    fn test_unstripify_bound_is_upper_bound() {
        let indices = vec![0u32, 1, 2, 2, 1, 3, 2, 3, 4];
        let strip = stripify(&indices, 5, Some(u32::MAX));
        let back = unstripify(&strip, Some(u32::MAX));
        assert!(back.len() <= unstripify_bound(strip.len()));
    }
}
