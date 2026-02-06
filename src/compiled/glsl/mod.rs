//! GLSL Transpiler (Deep Fried Edition)
//!
//! This module provides GLSL code generation for SDF trees.
//! Output is compatible with:
//!
//! - **Unity**: Shader Graph Custom Function, URP/HDRP shaders
//! - **OpenGL 4.x**: Compute shaders and fragment shaders
//! - **Vulkan**: GLSL shaders via glslang/SPIRV-Cross
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
//! // For Unity Shader Graph Custom Function
//! let unity_code = shader.to_unity_custom_function();
//! println!("{}", unity_code);
//!
//! // For Shadertoy-style fragment shader
//! let fragment = shader.to_fragment_shader();
//! println!("{}", fragment);
//!
//! // For OpenGL Compute Shader
//! let compute = shader.to_compute_shader();
//! println!("{}", compute);
//! ```
//!
//! Author: Moroya Sakamoto

mod transpiler;

pub use transpiler::{GlslShader, GlslTranspileMode};
