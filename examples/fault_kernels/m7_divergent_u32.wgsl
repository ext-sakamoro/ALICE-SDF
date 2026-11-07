struct P { first: u32, count: u32, }
@group(0) @binding(0) var<uniform> params: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
fn next(x: u32) -> u32 { var y = x; y ^= y << 13u; y ^= y >> 17u; y ^= y << 5u; return y; }

fn leaf(x0: u32, n: u32) -> u32 {
    var x = x0; var best = 0xffffffffu;
    for (var i = 0u; i < n; i += 1u) {
        x = next(x);
        var v = x >> 4u;
        if (x & 1u) != 0u { if (x & 2u) != 0u { v = v / 3u; } else { v = v + 7u; } }
        else if (x & 4u) != 0u { v = v * 5u; }
        if v < best { best = v; }
    }
    return best ^ x;
}
fn walk(x0: u32) -> u32 {
    var x = x0; var sum = 0u; var left = 40u + (x & 127u); var going = true;
    loop {
        if !going { if left == 0u { break; } left -= 1u; x = next(x); if (x & 3u) == 0u { continue; } going = true; }
        x = next(x);
        if (x & 7u) == 0u { sum = sum * 31u + leaf(x, 1u + (x >> 29u)); going = false; continue; }
        if (x & 8u) != 0u { sum += 1u; } else { going = false; }
    }
    return sum ^ x;
}
fn work(seed: u32) -> u32 {
    var x = seed * 2654435761u + 1u; var sum = 0u; var e = 0u;
    loop {
        x = next(x);
        if e >= 2u + (x >> 28u) { break; }
        e += 1u;
        let r = walk(x);
        sum = sum * 33u + r;
        if (r & 63u) == 0u { break; }
    }
    return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count { return; }
    out[id.x] = work(params.first + id.x);
}
