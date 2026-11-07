struct P { first: u32, count: u32, }
@group(0) @binding(0) var<uniform> params: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

fn work(seed: u32) -> u32 {
    var lut = array<u32, 16>(4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 24u, 25u, 26u, 27u, 16u, 10u, 18u, 9u);
    var x = seed * 2654435761u + 1u; var sum = 0u;
    for (var n = 0u; n < 20000u; n += 1u) {
        x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u;
        sum = sum * 31u + lut[x & 15u];
    }
    return sum;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count { return; }
    out[id.x] = work(params.first + id.x);
}
