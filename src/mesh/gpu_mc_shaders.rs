//! WGSL Shader Generation for GPU Marching Cubes (Deep Fried Edition)
//!
//! Generates the compute shaders of the GPU MC pipeline:
//!
//! - **Pass 1 (SDF Grid Eval)**: Evaluate SDF at all grid corners
//! - **Pass 2 (Cell Classify + Count)**: Classify cells, count triangle corners
//! - **Pass 2b (Edge Count)**: Count sign-changing lattice edges per grid point
//! - **Pass 3 (Vertex Generation)**: One vertex per sign-changing lattice edge
//! - **Pass 4 (Triangle Indices)**: Cell triangles as indices of those vertices
//!
//! The EDGE_TABLE and TRI_TABLE from Lorensen & Cline are embedded as
//! WGSL constant arrays for zero-latency lookup on the GPU.
//!
//! Author: Moroya Sakamoto

use crate::compiled::WgslShader;

/// Generate Pass 1 shader: SDF Grid Evaluation
///
/// Evaluates the SDF at every grid corner (res+1)^3.
/// Output: flat f32 buffer of SDF distances.
///
/// The lattice coordinates come from `axis_coords` (binding 2: the `res + 1`
/// x, then y, then z coordinates), computed on the host with the CPU
/// marching cubes formula, so both paths sample the field at bit-identical
/// points.
pub fn generate_sdf_grid_shader(sdf_shader: &WgslShader) -> String {
    format!(
        r"// ALICE-SDF GPU Marching Cubes - Pass 1: SDF Grid Eval

struct GridUniforms {{
    resolution: u32,
    iso_level: f32,
    _pad0: u32,
    _pad1: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}}

@group(0) @binding(0) var<storage, read_write> sdf_grid: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GridUniforms;
@group(0) @binding(2) var<storage, read> axis_coords: array<f32>;

{sdf_func}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let grid_res = uniforms.resolution + 1u;

    if (gid.x >= grid_res || gid.y >= grid_res || gid.z >= grid_res) {{
        return;
    }}

    let p = vec3<f32>(
        axis_coords[gid.x],
        axis_coords[grid_res + gid.y],
        axis_coords[2u * grid_res + gid.z],
    );

    let distance = sdf_eval(p);

    let idx = gid.x + gid.y * grid_res + gid.z * grid_res * grid_res;
    sdf_grid[idx] = distance;
}}
",
        sdf_func = sdf_shader.source,
    )
}

/// Generate Pass 2 shader: Cell Classification + Vertex Count
///
/// For each cell, compute cube_index from 8 corners and look up
/// vertex count from TRI_TABLE. Output count per cell for prefix sum.
pub fn generate_classify_shader() -> String {
    format!(
        r"// ALICE-SDF GPU Marching Cubes - Pass 2: Cell Classification

struct GridUniforms {{
    resolution: u32,
    iso_level: f32,
    _pad0: u32,
    _pad1: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}}

@group(0) @binding(0) var<storage, read> sdf_grid: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GridUniforms;
@group(0) @binding(2) var<storage, read_write> cell_vertex_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_cube_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> total_vertex_count: atomic<u32>;

{edge_table}

{vertex_count_table}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution;

    if (gid.x >= res || gid.y >= res || gid.z >= res) {{
        return;
    }}

    let grid_res = res + 1u;
    let iso = uniforms.iso_level;

    // 8 corner indices
    let i000 = gid.x       + gid.y       * grid_res + gid.z       * grid_res * grid_res;
    let i100 = (gid.x + 1u) + gid.y       * grid_res + gid.z       * grid_res * grid_res;
    let i010 = gid.x       + (gid.y + 1u) * grid_res + gid.z       * grid_res * grid_res;
    let i110 = (gid.x + 1u) + (gid.y + 1u) * grid_res + gid.z       * grid_res * grid_res;
    let i001 = gid.x       + gid.y       * grid_res + (gid.z + 1u) * grid_res * grid_res;
    let i101 = (gid.x + 1u) + gid.y       * grid_res + (gid.z + 1u) * grid_res * grid_res;
    let i011 = gid.x       + (gid.y + 1u) * grid_res + (gid.z + 1u) * grid_res * grid_res;
    let i111 = (gid.x + 1u) + (gid.y + 1u) * grid_res + (gid.z + 1u) * grid_res * grid_res;

    // Table corner numbering (Bourke): 0-3 on the y = 0 face (x, then z),
    // 4-7 on the y = 1 face — same as CORNER_OFFSETS on the CPU path.
    let d0 = sdf_grid[i000];
    let d1 = sdf_grid[i100];
    let d2 = sdf_grid[i101];
    let d3 = sdf_grid[i001];
    let d4 = sdf_grid[i010];
    let d5 = sdf_grid[i110];
    let d6 = sdf_grid[i111];
    let d7 = sdf_grid[i011];

    var cube_index = 0u;
    if (d0 < iso) {{ cube_index |= 1u; }}
    if (d1 < iso) {{ cube_index |= 2u; }}
    if (d2 < iso) {{ cube_index |= 4u; }}
    if (d3 < iso) {{ cube_index |= 8u; }}
    if (d4 < iso) {{ cube_index |= 16u; }}
    if (d5 < iso) {{ cube_index |= 32u; }}
    if (d6 < iso) {{ cube_index |= 64u; }}
    if (d7 < iso) {{ cube_index |= 128u; }}

    let cell_idx = gid.x + gid.y * res + gid.z * res * res;
    cell_cube_indices[cell_idx] = cube_index;

    let num_verts = VERTEX_COUNT_TABLE[cube_index];
    cell_vertex_counts[cell_idx] = num_verts;

    if (num_verts > 0u) {{
        atomicAdd(&total_vertex_count, num_verts);
    }}
}}
",
        edge_table = generate_edge_table_wgsl(),
        vertex_count_table = generate_vertex_count_table_wgsl(),
    )
}

/// Uniform block and the "inside" rule shared by the edge passes.
///
/// A grid value equal to the iso-level is outside (`d < iso` is inside), as
/// in the cell classification of Pass 2 and in the CPU marching cubes.
const EDGE_COMMON: &str = r"
struct GridUniforms {
    resolution: u32,
    iso_level: f32,
    _pad0: u32,
    _pad1: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

fn is_inside(d: f32) -> bool {
    return d < uniforms.iso_level;
}

// Number of sign-changing lattice edges at grid point (x, y, z) along the
// axes below `below` (0 = none, 3 = all), in the order x, y, z.
fn crossings_below(x: u32, y: u32, z: u32, below: u32) -> u32 {
    let res = uniforms.resolution;
    let g = res + 1u;
    let i = x + y * g + z * g * g;
    let a = is_inside(sdf_grid[i]);
    var n = 0u;
    if (below > 0u && x < res && a != is_inside(sdf_grid[i + 1u])) { n += 1u; }
    if (below > 1u && y < res && a != is_inside(sdf_grid[i + g])) { n += 1u; }
    if (below > 2u && z < res && a != is_inside(sdf_grid[i + g * g])) { n += 1u; }
    return n;
}
";

/// Generate Pass 2b shader: sign-changing lattice edges per grid point
///
/// Each lattice edge belongs to its lower grid point. Writes the number of
/// sign-changing edges (0–3) of every grid point and adds them to the
/// total vertex count.
pub(crate) fn generate_edge_count_shader() -> String {
    format!(
        r"// ALICE-SDF GPU Marching Cubes - Pass 2b: Edge Count

@group(0) @binding(0) var<storage, read> sdf_grid: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GridUniforms;
@group(0) @binding(2) var<storage, read_write> point_edge_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> total_edge_vertices: atomic<u32>;
{EDGE_COMMON}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let g = uniforms.resolution + 1u;
    if (gid.x >= g || gid.y >= g || gid.z >= g) {{
        return;
    }}
    let n = crossings_below(gid.x, gid.y, gid.z, 3u);
    point_edge_counts[gid.x + gid.y * g + gid.z * g * g] = n;
    if (n > 0u) {{
        atomicAdd(&total_edge_vertices, n);
    }}
}}
"
    )
}

/// Generate Pass 3 shader: Vertex Generation
///
/// One vertex per sign-changing lattice edge, written at the grid point's
/// offset (exclusive prefix sum of Pass 2b) plus the number of crossing
/// edges at that point along lower axes — the CPU marching cubes vertex
/// order. The position is interpolated from the edge's lower endpoint, the
/// normal is the tetrahedral gradient of the SDF at the position, so a
/// vertex has one value whichever cell refers to it.
pub fn generate_vertex_shader(sdf_shader: &WgslShader) -> String {
    format!(
        r"// ALICE-SDF GPU Marching Cubes - Pass 3: Edge Vertex Generation

struct GpuVertex {{
    px: f32, py: f32, pz: f32,
    nx: f32, ny: f32, nz: f32,
    _pad0: f32, _pad1: f32,
}}

@group(0) @binding(0) var<storage, read> sdf_grid: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GridUniforms;
@group(0) @binding(2) var<storage, read> point_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_vertices: array<GpuVertex>;
@group(0) @binding(4) var<storage, read> axis_coords: array<f32>;
{common}
{sdf_func}

fn lattice(x: u32, y: u32, z: u32) -> vec3<f32> {{
    let g = uniforms.resolution + 1u;
    return vec3<f32>(axis_coords[x], axis_coords[g + y], axis_coords[2u * g + z]);
}}

// Iso crossing on the edge p0 -> p1 evaluated from p0, the lower endpoint
// (the CPU marching cubes interpolation).
fn interpolate_edge(p0: vec3<f32>, p1: vec3<f32>, d0: f32, d1: f32, iso: f32) -> vec3<f32> {{
    let denom = d1 - d0;
    let safe = select(-max(abs(denom), 1e-10), max(abs(denom), 1e-10), denom >= 0.0);
    let t = clamp((iso - d0) / safe, 0.0, 1.0);
    return p0 + (p1 - p0) * t;
}}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {{
    let e = 0.001;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);
    return normalize(
        k0 * sdf_eval(p + k0 * e) +
        k1 * sdf_eval(p + k1 * e) +
        k2 * sdf_eval(p + k2 * e) +
        k3 * sdf_eval(p + k3 * e)
    );
}}

fn emit(slot: u32, p: vec3<f32>) {{
    let n = estimate_normal(p);
    output_vertices[slot].px = p.x;
    output_vertices[slot].py = p.y;
    output_vertices[slot].pz = p.z;
    output_vertices[slot].nx = n.x;
    output_vertices[slot].ny = n.y;
    output_vertices[slot].nz = n.z;
    output_vertices[slot]._pad0 = 0.0;
    output_vertices[slot]._pad1 = 0.0;
}}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution;
    let g = res + 1u;
    if (gid.x >= g || gid.y >= g || gid.z >= g) {{
        return;
    }}
    let i = gid.x + gid.y * g + gid.z * g * g;
    let d0 = sdf_grid[i];
    let a = is_inside(d0);
    let iso = uniforms.iso_level;
    let p0 = lattice(gid.x, gid.y, gid.z);
    var slot = point_offsets[i];
    if (gid.x < res) {{
        let d1 = sdf_grid[i + 1u];
        if (a != is_inside(d1)) {{
            emit(slot, interpolate_edge(p0, lattice(gid.x + 1u, gid.y, gid.z), d0, d1, iso));
            slot += 1u;
        }}
    }}
    if (gid.y < res) {{
        let d1 = sdf_grid[i + g];
        if (a != is_inside(d1)) {{
            emit(slot, interpolate_edge(p0, lattice(gid.x, gid.y + 1u, gid.z), d0, d1, iso));
            slot += 1u;
        }}
    }}
    if (gid.z < res) {{
        let d1 = sdf_grid[i + g * g];
        if (a != is_inside(d1)) {{
            emit(slot, interpolate_edge(p0, lattice(gid.x, gid.y, gid.z + 1u), d0, d1, iso));
        }}
    }}
}}
",
        common = EDGE_COMMON,
        sdf_func = sdf_shader.source,
    )
}

/// Generate Pass 4 shader: triangle indices
///
/// For each non-empty cell, writes the triangle table's corners as indices
/// of the Pass 3 edge vertices (lower grid point offset + crossings at that
/// point along lower axes), at the cell's offset (exclusive prefix sum of
/// the Pass 2 counts).
pub(crate) fn generate_cell_index_shader() -> String {
    format!(
        r"// ALICE-SDF GPU Marching Cubes - Pass 4: Triangle Indices

@group(0) @binding(0) var<storage, read> sdf_grid: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GridUniforms;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> cell_cube_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_indices: array<u32>;
// Triangle table (256 x 16, -1 terminated) as a buffer rather than a WGSL
// const array<i32, 4096>: naga's HLSL backend lowers a dynamically indexed
// module constant into indexable temporaries and FXC rejects the shader
// (X4505: sum of temp registers exceeds limit of 4096) on DX12 / WARP.
@group(0) @binding(5) var<storage, read> TRI_TABLE: array<i32>;
@group(0) @binding(6) var<storage, read> point_offsets: array<u32>;
{EDGE_COMMON}
// Lower corner offset (xyz) and axis (w) of cube edge e, Bourke numbering
// (corners 0-3 on the y = 0 face, 4-7 on y = 1, as CORNER_OFFSETS).
fn edge_lower_axis(e: i32) -> vec4<u32> {{
    switch e {{
        case 0: {{ return vec4<u32>(0u, 0u, 0u, 0u); }}
        case 1: {{ return vec4<u32>(1u, 0u, 0u, 2u); }}
        case 2: {{ return vec4<u32>(0u, 0u, 1u, 0u); }}
        case 3: {{ return vec4<u32>(0u, 0u, 0u, 2u); }}
        case 4: {{ return vec4<u32>(0u, 1u, 0u, 0u); }}
        case 5: {{ return vec4<u32>(1u, 1u, 0u, 2u); }}
        case 6: {{ return vec4<u32>(0u, 1u, 1u, 0u); }}
        case 7: {{ return vec4<u32>(0u, 1u, 0u, 2u); }}
        case 8: {{ return vec4<u32>(0u, 0u, 0u, 1u); }}
        case 9: {{ return vec4<u32>(1u, 0u, 0u, 1u); }}
        case 10: {{ return vec4<u32>(1u, 0u, 1u, 1u); }}
        default: {{ return vec4<u32>(0u, 0u, 1u, 1u); }}
    }}
}}

fn edge_vertex(cx: u32, cy: u32, cz: u32, e: i32) -> u32 {{
    let g = uniforms.resolution + 1u;
    let la = edge_lower_axis(e);
    let x = cx + la.x;
    let y = cy + la.y;
    let z = cz + la.z;
    return point_offsets[x + y * g + z * g * g] + crossings_below(x, y, z, la.w);
}}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution;
    if (gid.x >= res || gid.y >= res || gid.z >= res) {{
        return;
    }}
    let cell_idx = gid.x + gid.y * res + gid.z * res * res;
    let cube_index = cell_cube_indices[cell_idx];
    if (cube_index == 0u || cube_index == 255u) {{
        return;
    }}
    var slot = cell_offsets[cell_idx];
    let row = cube_index * 16u;
    for (var k = 0u; k < 16u; k += 1u) {{
        let e = TRI_TABLE[row + k];
        if (e < 0) {{
            break;
        }}
        output_indices[slot] = edge_vertex(gid.x, gid.y, gid.z, e);
        slot += 1u;
    }}
}}
"
    )
}

/// The flat triangle table Pass 4 reads through its `TRI_TABLE` storage
/// binding (256 rows × 16, `-1` terminated).
pub fn tri_table_flat() -> Vec<i32> {
    get_tri_table()
        .iter()
        .flat_map(|row| row.iter().map(|&v| i32::from(v)))
        .collect()
}

/// Generate EDGE_TABLE as WGSL const array
fn generate_edge_table_wgsl() -> String {
    let table: [u16; 256] = [
        0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a,
        0xd03, 0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895,
        0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435,
        0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa,
        0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
        0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
        0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff,
        0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
        0x2cf, 0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
        0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
        0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256,
        0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc,
        0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f,
        0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3,
        0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a,
        0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795,
        0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
        0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
    ];

    let entries: Vec<String> = table.iter().map(|&v| format!("{}u", v)).collect();
    format!(
        "const EDGE_TABLE: array<u32, 256> = array<u32, 256>(\n    {}\n);",
        entries.join(", ")
    )
}

/// Generate vertex count per cube_index as WGSL const
fn generate_vertex_count_table_wgsl() -> String {
    // Pre-computed: count of vertices (not triangles) per cube_index
    // This is derived from TRI_TABLE by counting non-(-1) entries
    let tri_table = get_tri_table();
    let mut counts = [0u32; 256];

    for (i, row) in tri_table.iter().enumerate() {
        let mut count = 0u32;
        for j in (0..16).step_by(3) {
            if row[j] == -1 {
                break;
            }
            count += 3;
        }
        counts[i] = count;
    }

    let entries: Vec<String> = counts.iter().map(|&v| format!("{}u", v)).collect();
    format!(
        "const VERTEX_COUNT_TABLE: array<u32, 256> = array<u32, 256>(\n    {}\n);",
        entries.join(", ")
    )
}

/// Workgroup size for prefix sum shaders
pub const PREFIX_SUM_WG: u32 = 256;

/// Generate Prefix Sum Scan shader (Hillis-Steele, exclusive)
///
/// Each workgroup scans a block of `PREFIX_SUM_WG` elements.
/// Block sums are written to `block_sums` for recursive scanning.
pub fn generate_prefix_sum_scan_shader() -> String {
    format!(
        r"// ALICE-SDF GPU Prefix Sum - Scan Pass (Hillis-Steele)

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = element_count

var<workgroup> s_a: array<u32, {wg}>;
var<workgroup> s_b: array<u32, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let tid = lid.x;
    let block_offset = wid.x * {wg}u;
    let global_id = block_offset + tid;
    let n = params.x;

    // Load input into shared memory (0 for out-of-bounds)
    if (global_id < n) {{
        s_a[tid] = input[global_id];
    }} else {{
        s_a[tid] = 0u;
    }}
    workgroupBarrier();

    // Hillis-Steele inclusive scan with ping-pong buffers
    var read_buf = 0u; // 0 = s_a, 1 = s_b
    var stride = 1u;
    loop {{
        if (stride >= {wg}u) {{
            break;
        }}
        if (read_buf == 0u) {{
            if (tid >= stride) {{
                s_b[tid] = s_a[tid] + s_a[tid - stride];
            }} else {{
                s_b[tid] = s_a[tid];
            }}
        }} else {{
            if (tid >= stride) {{
                s_a[tid] = s_b[tid] + s_b[tid - stride];
            }} else {{
                s_a[tid] = s_b[tid];
            }}
        }}
        read_buf = 1u - read_buf;
        stride = stride * 2u;
        workgroupBarrier();
    }}

    // Read inclusive scan result from the current read buffer
    var inclusive: u32;
    if (read_buf == 0u) {{
        inclusive = s_a[tid];
    }} else {{
        inclusive = s_b[tid];
    }}

    // Convert inclusive scan to exclusive: shift right, first element = 0
    if (global_id < n) {{
        if (tid == 0u) {{
            output[global_id] = 0u;
        }} else {{
            if (read_buf == 0u) {{
                output[global_id] = s_a[tid - 1u];
            }} else {{
                output[global_id] = s_b[tid - 1u];
            }}
        }}
    }}

    // Last thread in block writes the block's total (inclusive scan of last element)
    if (tid == {wg}u - 1u) {{
        block_sums[wid.x] = inclusive;
    }}
}}
",
        wg = PREFIX_SUM_WG,
    )
}

/// Generate Prefix Sum Propagate shader
///
/// Adds scanned block sums back to each element (skipping block 0).
pub fn generate_prefix_sum_propagate_shader() -> String {
    format!(
        r"// ALICE-SDF GPU Prefix Sum - Propagate Pass

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = element_count

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    // Block 0 already has correct values; only propagate to blocks 1+
    if (wid.x == 0u) {{
        return;
    }}

    let global_id = wid.x * {wg}u + lid.x;
    let n = params.x;

    if (global_id < n) {{
        data[global_id] = data[global_id] + block_offsets[wid.x];
    }}
}}
",
        wg = PREFIX_SUM_WG,
    )
}

/// Standard Marching Cubes TRI_TABLE (Lorensen & Cline, 1987)
const fn get_tri_table() -> [[i8; 16]; 256] {
    [
        [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    ]
}
