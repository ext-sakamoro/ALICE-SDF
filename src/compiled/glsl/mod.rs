//! GLSL Transpiler (Deep Fried Edition)
//!
//! This module provides GLSL code generation for SDF trees.
//! Output is compatible with:
//!
//! - **OpenGL 4.x**: Compute shaders and fragment shaders
//! - **Vulkan**: compute shaders (`to_vulkan_compute_shader`)
//!
//! Unity Shader Graph takes HLSL: use `HlslShader::to_unity_custom_function`
//! / `HlslShader::export_unity_shader_graph` (`hlsl` feature).
//! - **Shadertoy**: Fragment shader for web-based visualization
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::glsl::GlslShader;
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // Generate GLSL code
//! let shader = GlslShader::transpile(&shape);
//!
//! // For Shadertoy-style fragment shader
//! let fragment = shader.to_fragment_shader();
//! println!("{}", fragment);
//!
//! // For OpenGL / Vulkan Compute Shader
//! let compute = shader.to_compute_shader();
//! let vulkan = shader.to_vulkan_compute_shader();
//! println!("{}", compute);
//! ```
//!
//! Author: Moroya Sakamoto

pub mod render_pipeline;
mod transpiler;

pub use render_pipeline::RenderConfig;
pub use transpiler::{GlslShader, GlslTranspileMode};
