# Shader Target Reference

ALICE-SDF transpiles `SdfNode` → GLSL (Unity / OpenGL / Vulkan), WGSL (WebGPU / wgpu), or HLSL (Unreal Engine 5-6 / DirectX). Emit is driven by `--shader {glsl|wgsl|hlsl}`.

## GLSL (default, feature `glsl`)

Target: Unity URP / HDRP, OpenGL 4.x, Vulkan GLSL. Emits `float sdf(vec3 p)` and `vec3 color(vec3 p, vec3 n)`.

- `#version 450 core` header assumed
- `float` precision default; caller may prepend `precision highp float;` for GLES3
- Structs / uniforms named `alice_*` to avoid collision
- Compatible with Shadertoy style (`mainImage(out vec4 fragColor, in vec2 fragCoord)`) via optional wrapper

## WGSL (feature `gpu`, transpiler always available)

Target: WebGPU, wgpu-rs, Deno / Chrome / Firefox / Safari (browser raymarch), Rust native via `wgpu` crate.

- `fn sdf(p: vec3<f32>) -> f32`
- `@group(0) @binding(N)` uniform layout, `N` auto-assigned in emit order
- naga-verified — the transpiler runs `naga::valid::Validator` before returning
- Storage buffers used for parameter arrays; push constants avoided (WebGPU compat)

## HLSL (feature `hlsl`)

Target: Unreal Engine 5-6 (`.usf` / `.ush` Custom Material), DirectX 12, Slang.

- `float sdf(float3 p)` and `float3 color(float3 p, float3 n)`
- `cbuffer AliceParams : register(b0)` uniform layout
- UE5 `.usf` compatibility: emit avoids features not in UE5 material node HLSL (see `~/ALICE-SDF/CHANGELOG.md` for version matrix)

## Cross-target gotchas

- **`vec3` alignment**: WGSL and HLSL both pad `vec3` to 16 bytes in uniform buffers. Follow up with a `float` field to fill the pad; ALICE-SDF emit handles this automatically for its own uniforms.
- **`inversesqrt` / `rsqrt`**: GLSL `inversesqrt`, HLSL `rsqrt`, WGSL `inverseSqrt` — transpiler handles.
- **`fract` sign convention**: GLSL/HLSL/WGSL all use `p - floor(p)`, consistent.
- **Sampler binding**: WGSL requires separate `sampler` and `texture` bindings; GLSL/HLSL can combine. Use only procedural textures inside the SDF emit path to avoid sampler complexity.
- **Loop bound**: HLSL under UE5 prefers `[loop]` attribute with dynamic bound. WGSL and GLSL 4.x support dynamic bounds directly. Ray marching iteration count is emitted as a compile-time constant to match all three.

## Selection guide

| Use case | Target |
|--|--|
| Browser raymarch demo, WebGPU | WGSL |
| Unity URP procedural material | GLSL |
| Unreal Engine 5-6 Custom node | HLSL |
| Vulkan native compute | WGSL (via naga → SPIR-V) or GLSL |
| Godot 4 spatial shader | GLSL (Godot Vulkan renderer accepts GLSL 450) |
| VRChat world / avatar | GLSL (Unity URP HLSL under the hood, but ALICE emits GLSL that VRChat's shader compiler accepts) |

## Related

See `karikari-shader` and `alice-sdf-shader-discipline` skills in `~/claude-config/claude-skills/` for shader-level code-quality gates and known SDF raymarch pitfalls (Y-axis conflict, Bezier 16-sample discretization, box-style z-thickness constraint, etc.).
