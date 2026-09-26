//! MSL (Metal Shading Language) emit for Apple platforms
//!
//! # Why this is not a fourth `ShaderLang`
//!
//! [`ShaderLang`](crate::compiled::transpiler_common::ShaderLang) is sealed and
//! implemented by `WgslLang` / `GlslLang` / `HlslLang` only. Each of those
//! carries its own copy of every law: `helper_source` alone supplies 56 helper
//! functions (`sdf_bezier`, `sdf_icosahedron`, `alice_atan2`, ...). Adding a
//! hand-written `MslLang` would make that **four** copies of the same 56 laws
//! to keep in step, and law drift between paths is the failure this crate keeps
//! paying for (the mirrored taper that survived in all three transpilers until
//! the GPU parity oracle caught it, `Terrain` emitting GLSL syntax from the
//! language-neutral walker, `Elongate` disagreeing with the CPU).
//!
//! So MSL is **derived, not written**: the WGSL emit — already verified against
//! the CPU for every corpus node by the GPU parity oracle — is the single
//! source of law, and [`naga`] translates it to MSL. A law fixed once in the
//! WGSL path is fixed in MSL for free, and MSL cannot drift from WGSL by
//! construction.
//!
//! This also matches ALICE-SDF-LAWS §同期の法 §5 (Port Parity Oracle Rule):
//! single source generation first, oracle second.
//!
//! # Buffer slots
//!
//! WGSL `@group(0) @binding(N)` maps to Metal `buffer(N)`, so the compute
//! pipeline binds:
//!
//! | slot | WGSL binding | contents |
//! |------|--------------|----------|
//! | 0 | `input_points` | `array<InputPoint>` (storage, read) |
//! | 1 | `output_distances` / `output` | storage, read_write |
//! | 2 | `point_count` | uniform `u32` |
//! | 3 | `sdf_params` | uniform `Params` (only in [`TranspileMode::Dynamic`]) |
//! | [`MslShader::sizes_buffer_slot`] | — | `u32` lengths of the runtime-sized arrays, in declaration order (one past the highest binding) |
//!
//! The sizes buffer is required because the WGSL uses runtime-sized arrays;
//! Metal has no equivalent, so naga passes the lengths in explicitly.
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::msl::MslShader;
//! use alice_sdf::compiled::TranspileMode;
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! let shader = MslShader::transpile(&shape, TranspileMode::Hardcoded)?;
//! // Feed shader.source to MTLDevice::newLibraryWithSource, then look up
//! // shader.entry_point and dispatch shader.workgroup_size threads per group.
//! println!("{}", shader.source);
//! ```
//!
//! Author: Moroya Sakamoto

mod transpiler;

pub use transpiler::{MslError, MslShader};
