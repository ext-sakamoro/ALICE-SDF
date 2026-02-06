//! HLSL Transpiler (Deep Fried Edition)
//!
//! This module provides HLSL code generation for SDF trees.
//! Output is compatible with:
//!
//! - **Unreal Engine 5**: Custom Material Expression nodes
//! - **DirectX 11/12**: Compute shaders and pixel shaders
//! - **Unity HDRP**: Custom shader passes (with minor modifications)
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::compiled::hlsl::HlslShader;
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! // Generate HLSL code
//! let shader = HlslShader::transpile(&shape);
//!
//! // For UE5 Custom Material Expression
//! let ue5_code = shader.to_ue5_custom_node();
//! println!("{}", ue5_code);
//!
//! // For DirectX Compute Shader
//! let compute = shader.to_compute_shader();
//! println!("{}", compute);
//! ```
//!
//! Author: Moroya Sakamoto

mod transpiler;

pub use transpiler::{HlslShader, HlslTranspileMode};
