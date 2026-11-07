struct P { first: u32, count: u32, }
@group(0) @binding(0) var<uniform> params: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

fn work(seed: u32) -> u32 {
    var st: array<u32, 64>; var top = 0u; var x = seed * 2654435761u + 1u; var sum = 0u;
    for (var n = 0u; n < 20000u; n += 1u) {
        x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u;
        if (x & 3u) != 0u && top < 64u { st[top] = x; top += 1u; }
        else if top > 0u { top -= 1u; sum = sum * 31u + st[top]; }
    }
    return sum;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count { return; }
    out[id.x] = work(params.first + id.x);
}
