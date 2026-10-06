# ALICE-SDF Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

🟡 **481 baseline items** — Permitted violations, ratchet in place

---

## 📋 Baseline (481 permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
| `src/material.rs` | 39 |
| `src/npr/compiled_color.rs` | 32 |
| `src/compiled/glsl/render_pipeline.rs` | 16 |
| `src/crispy.rs` | 16 |
| `src/autodiff.rs` | 15 |
| `src/compiled/wgsl/gpu_eval.rs` | 14 |
| `src/sim_bridge.rs` | 14 |
| `src/mesh/lod.rs` | 13 |
| `src/mesh/meshopt_filter.rs` | 13 |
| `src/mesh/nanite.rs` | 13 |
| `src/codec_bridge.rs` | 12 |
| `src/mesh/mesh_to_sdf.rs` | 10 |
| `src/mesh/primitive_fitting.rs` | 10 |
| `src/cache/chunked.rs` | 9 |
| `src/compiled/instanced.rs` | 9 |
| `src/destruction/mod.rs` | 9 |
| `src/mesh/collision.rs` | 9 |
| `src/compiled/aabb.rs` | 8 |
| `src/compiled/glsl/transpiler.rs` | 6 |
| `src/compiled/wgsl/transpiler.rs` | 6 |
| `src/mesh/quantization.rs` | 6 |
| `src/npr/dsl.rs` | 6 |
| `src/terrain/splatmap.rs` | 6 |
| `src/volume/export.rs` | 6 |
| `src/font_bridge.rs` | 5 |
| `src/mesh/hermite.rs` | 5 |
| `src/mesh/lod_persist.rs` | 5 |
| `src/mesh/manifold.rs` | 5 |
| `src/mesh/mesh_codec.rs` | 5 |
| `src/mesh/stripifier.rs` | 5 |
| `src/terrain/clipmap.rs` | 5 |
| `src/terrain/heightmap.rs` | 5 |
| `src/asp_bridge.rs` | 4 |
| `src/compiled/hlsl/transpiler.rs` | 4 |
| `src/compiled/jit/runtime.rs` | 4 |
| `src/compiled/jit/simd/mod.rs` | 4 |
| `src/compiled/opcode.rs` | 4 |
| `src/compiled/simd.rs` | 4 |
| `src/eval/parallel.rs` | 4 |
| `src/ffi/registry.rs` | 4 |
| `src/mesh/meshlet.rs` | 4 |
| `src/mesh/optimize.rs` | 4 |
| `src/soa.rs` | 4 |
| `src/svo/linearize.rs` | 4 |
| `src/svo/mod.rs` | 4 |
| `src/svo/streaming.rs` | 4 |
| `src/volume/mod.rs` | 4 |
| `src/cache/mod.rs` | 3 |
| `src/ffi/types.rs` | 3 |
| `src/gi/irradiance.rs` | 3 |
| `src/interval.rs` | 3 |
| `src/mesh/overdraw.rs` | 3 |
| `src/mesh/point_cloud_sdf.rs` | 3 |
| `src/physics_bridge.rs` | 3 |
| `src/primitives/mod.rs` | 3 |
| `src/svo/query.rs` | 3 |
| `src/cache_bridge.rs` | 2 |
| `src/compiled/compiler.rs` | 2 |
| `src/compiled/eval_bvh.rs` | 2 |
| `src/destruction/debris.rs` | 2 |
| `src/destruction/operations.rs` | 2 |
| `src/mesh/bvh.rs` | 2 |
| `src/mesh/decimate.rs` | 2 |
| `src/mesh/lightmap.rs` | 2 |
| `src/mesh/mesh_sign.rs` | 2 |
| `src/mesh/meshopt_vertex_codec.rs` | 2 |
| `src/mesh/spatial_order.rs` | 2 |
| `src/npr/outline.rs` | 2 |
| `src/npr/scene_composer.rs` | 2 |
| `src/python/helpers.rs` | 2 |
| `src/terrain/caves.rs` | 2 |
| `src/volume/bake.rs` | 2 |
| `src/bin/main.rs` | 1 |
| `src/compiled/blinkscript/transpiler.rs` | 1 |
| `src/compiled/eval.rs` | 1 |
| `src/compiled/eval_simd.rs` | 1 |
| `src/compiled/eval_soa.rs` | 1 |
| `src/compiled/instruction.rs` | 1 |
| `src/compiled/jit/codegen.rs` | 1 |
| `src/compiled/transpiler_common.rs` | 1 |
| `src/eval/mod.rs` | 1 |
| `src/gi/cone_trace.rs` | 1 |
| `src/gi/mod.rs` | 1 |
| `src/io/asdf.rs` | 1 |
| `src/mesh/dual_contouring.rs` | 1 |
| `src/mesh/meshopt_index_codec.rs` | 1 |
| `src/mesh/mod.rs` | 1 |
| `src/mesh/sdf_to_mesh.rs` | 1 |
| `src/mesh/uv_unwrap.rs` | 1 |
| `src/modifiers/surface_roughness.rs` | 1 |
| `src/neural.rs` | 1 |
| `src/optimize.rs` | 1 |
| `src/terrain/mod.rs` | 1 |
| `src/texture/fitting.rs` | 1 |
| `src/tight_aabb.rs` | 1 |
| `src/volume/gpu_bake.rs` | 1 |
| `src/volume/mipchain.rs` | 1 |

### Dead Code (13)

```
dead_code src/bin/main.rs 1
dead_code src/compiled/aabb.rs 5
dead_code src/compiled/glsl/transpiler.rs 2
dead_code src/compiled/hlsl/transpiler.rs 2
dead_code src/compiled/jit/simd/mod.rs 2
dead_code src/compiled/wgsl/transpiler.rs 2
dead_code src/ffi/registry.rs 3
dead_code src/io/asdf.rs 2
dead_code src/mesh/mesh_to_sdf.rs 1
dead_code src/neural.rs 1
dead_code src/npr/compiled_color.rs 1
dead_code src/python/helpers.rs 1
dead_code src/volume/export.rs 1
```

### Unwired Items (468)

```
unwired src/asp_bridge.rs::create_sdf_d_packet
unwired src/asp_bridge.rs::create_sdf_i_packet
unwired src/asp_bridge.rs::decode_sdf_i_packet
unwired src/asp_bridge.rs::estimate_packet_size
unwired src/autodiff.rs::dual3_box
unwired src/autodiff.rs::dual3_plane
unwired src/autodiff.rs::dual3_point
unwired src/autodiff.rs::dual3_sphere
unwired src/autodiff.rs::dual3_torus
unwired src/autodiff.rs::eval_dual3
unwired src/autodiff.rs::eval_with_gradient
unwired src/autodiff.rs::from_val_grad
unwired src/autodiff.rs::gaussian_curvature
unwired src/autodiff.rs::gradient
unwired src/autodiff.rs::gradient_magnitude
unwired src/autodiff.rs::length2
unwired src/autodiff.rs::length3
unwired src/autodiff.rs::principal_curvatures
unwired src/autodiff.rs::variable
unwired src/cache/chunked.rs::cached_chunks
unwired src/cache/chunked.rs::chunk_bounds
unwired src/cache/chunked.rs::get_chunk
unwired src/cache/chunked.rs::invalidate_all
unwired src/cache/chunked.rs::load_chunk
unwired src/cache/chunked.rs::merge_all
unwired src/cache/chunked.rs::persist_dirty
unwired src/cache/chunked.rs::set_chunk
unwired src/cache/chunked.rs::update_sdf_hash
unwired src/cache/mod.rs::compute_cache_key
unwired src/cache/mod.rs::get_or_generate
unwired src/cache/mod.rs::hash_sdf_node
unwired src/cache_bridge.rs::hit_rate
unwired src/cache_bridge.rs::put
unwired src/codec_bridge.rs::CompressResult
unwired src/codec_bridge.rs::VolumeStats
unwired src/codec_bridge.rs::compress_sdf
unwired src/codec_bridge.rs::compression_ratio
unwired src/codec_bridge.rs::decode_sdf_volume
unwired src/codec_bridge.rs::decompress_sdf
unwired src/codec_bridge.rs::encode_sdf_volume
unwired src/codec_bridge.rs::lossless
unwired src/codec_bridge.rs::volume_stats
unwired src/codec_bridge.rs::voxelize_sdf
unwired src/codec_bridge.rs::voxelize_sdf_uniform
unwired src/codec_bridge.rs::world_pos
unwired src/compiled/aabb.rs::distance_to_point_fast
unwired src/compiled/aabb.rs::from_half_size
unwired src/compiled/aabb.rs::hex_prism_aabb
unwired src/compiled/aabb.rs::link_aabb
unwired src/compiled/aabb.rs::octahedron_aabb
unwired src/compiled/aabb.rs::pyramid_aabb
unwired src/compiled/aabb.rs::rounded_cone_aabb
unwired src/compiled/blinkscript/transpiler.rs::extract_params
unwired src/compiled/compiler.rs::aux_data
unwired src/compiled/compiler.rs::memory_size
unwired src/compiled/eval.rs::eval_compiled_distance_and_normal
unwired src/compiled/eval_bvh.rs::memory_size
unwired src/compiled/eval_bvh.rs::refit_all_from_bytecode
unwired src/compiled/eval_simd.rs::eval_gradient_simd
unwired src/compiled/eval_soa.rs::eval_compiled_batch_soa_into
unwired src/compiled/glsl/render_pipeline.rs::BIOME_SYSTEM
unwired src/compiled/glsl/render_pipeline.rs::DESTRUCTION_SYSTEM
unwired src/compiled/glsl/render_pipeline.rs::DESTRUCTION_UNIFORMS
unwired src/compiled/glsl/render_pipeline.rs::INTERIOR_MAPPING_LIB
unwired src/compiled/glsl/render_pipeline.rs::MICRO_NORMAL_LIB
unwired src/compiled/glsl/render_pipeline.rs::NOISE_LIB
unwired src/compiled/glsl/render_pipeline.rs::NORMAL_AO_SHADOW
unwired src/compiled/glsl/render_pipeline.rs::PBR_BRDF
unwired src/compiled/glsl/render_pipeline.rs::POST_PROCESS
unwired src/compiled/glsl/render_pipeline.rs::SKY_ATMOSPHERE
unwired src/compiled/glsl/render_pipeline.rs::SPECTRAL_LIB
unwired src/compiled/glsl/render_pipeline.rs::UNIFORMS
unwired src/compiled/glsl/render_pipeline.rs::VFX_LIB
unwired src/compiled/glsl/render_pipeline.rs::VOLUMETRIC_LIGHT
unwired src/compiled/glsl/render_pipeline.rs::build_full_shader
unwired src/compiled/glsl/render_pipeline.rs::build_main_function
unwired src/compiled/glsl/transpiler.rs::export_unity_shader_graph
unwired src/compiled/glsl/transpiler.rs::extract_params
unwired src/compiled/glsl/transpiler.rs::to_fragment_shader
unwired src/compiled/glsl/transpiler.rs::to_fragment_shader_full
unwired src/compiled/glsl/transpiler.rs::to_unity_custom_function
unwired src/compiled/hlsl/transpiler.rs::export_ue5_material_function
unwired src/compiled/hlsl/transpiler.rs::extract_params
unwired src/compiled/hlsl/transpiler.rs::to_ue5_custom_node
unwired src/compiled/instanced.rs::add_at
unwired src/compiled/instanced.rs::add_instance
unwired src/compiled/instanced.rs::eval_min
unwired src/compiled/instanced.rs::eval_min_batch
unwired src/compiled/instanced.rs::eval_min_batch_simd
unwired src/compiled/instanced.rs::eval_min_simd
unwired src/compiled/instanced.rs::eval_per_instance
unwired src/compiled/instanced.rs::instance_count
unwired src/compiled/instanced.rs::to_instanced_wgsl
unwired src/compiled/instruction.rs::next_instruction_index
unwired src/compiled/jit/codegen.rs::extract_jit_params
unwired src/compiled/jit/runtime.rs::JitCompiledSdf::eval_batch_parallel
unwired src/compiled/jit/runtime.rs::JitCompiledSdfDynamic::eval_batch_parallel
unwired src/compiled/jit/runtime.rs::params
unwired src/compiled/jit/runtime.rs::update_params
unwired src/compiled/jit/simd/mod.rs::extract_simd_params
unwired src/compiled/jit/simd/mod.rs::params
unwired src/compiled/jit/simd/mod.rs::update_params
unwired src/compiled/opcode.rs::is_modifier
unwired src/compiled/opcode.rs::is_post_process
unwired src/compiled/opcode.rs::is_transform
unwired src/compiled/opcode.rs::modifies_point
unwired src/compiled/simd.rs::max_component
unwired src/compiled/simd.rs::max_zero
unwired src/compiled/simd.rs::min_component
unwired src/compiled/simd.rs::mul_vec3
unwired src/compiled/transpiler_common.rs::SHADER_UNSUPPORTED
unwired src/compiled/wgsl/gpu_eval.rs::create_buffer_pool
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_async
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_auto
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_full
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_pooled
unwired src/compiled/wgsl/gpu_eval.rs::eval_batch_submit
unwired src/compiled/wgsl/gpu_eval.rs::from_glsl_compute
unwired src/compiled/wgsl/gpu_eval.rs::from_shader_async
unwired src/compiled/wgsl/gpu_eval.rs::from_wgsl_async
unwired src/compiled/wgsl/gpu_eval.rs::new_async
unwired src/compiled/wgsl/gpu_eval.rs::new_dynamic
unwired src/compiled/wgsl/gpu_eval.rs::resolve
unwired src/compiled/wgsl/gpu_eval.rs::update_params
unwired src/compiled/wgsl/gpu_eval.rs::wait
unwired src/compiled/wgsl/transpiler.rs::extract_params
unwired src/compiled/wgsl/transpiler.rs::to_compute_shader_with_normals
unwired src/compiled/wgsl/transpiler.rs::to_volume_shader
unwired src/compiled/wgsl/transpiler.rs::transpile_material
unwired src/compiled/wgsl/transpiler.rs::with_workgroup_size
unwired src/crispy.rs::BitMask64::test
unwired src/crispy.rs::BloomFilter::test
unwired src/crispy.rs::EMPTY
unwired src/crispy.rs::FULL
unwired src/crispy.rs::and
unwired src/crispy.rs::branchless_abs
unwired src/crispy.rs::branchless_clamp
unwired src/crispy.rs::branchless_max
unwired src/crispy.rs::branchless_min
unwired src/crispy.rs::fast_recip
unwired src/crispy.rs::fast_recip_vec3
unwired src/crispy.rs::from_items
unwired src/crispy.rs::or
unwired src/crispy.rs::round_half_up_vec3
unwired src/crispy.rs::select_f32
unwired src/crispy.rs::test_hash
unwired src/destruction/debris.rs::DebrisPiece
unwired src/destruction/debris.rs::generate_debris
unwired src/destruction/mod.rs::ChunkMesh
unwired src/destruction/mod.rs::chunk_size
unwired src/destruction/mod.rs::chunks_per_axis
unwired src/destruction/mod.rs::clear_dirty
unwired src/destruction/mod.rs::get_material
unwired src/destruction/mod.rs::is_chunk_dirty
unwired src/destruction/mod.rs::remesh_all_dirty
unwired src/destruction/mod.rs::remesh_chunk
unwired src/destruction/mod.rs::set_distance
unwired src/destruction/operations.rs::carve_batch
unwired src/destruction/operations.rs::explode
unwired src/eval/mod.rs::gradient
unwired src/eval/parallel.rs::eval_grid
unwired src/eval/parallel.rs::eval_grid_with_normals
unwired src/eval/parallel.rs::grid_coords
unwired src/eval/parallel.rs::grid_index
unwired src/ffi/registry.rs::clear_all
unwired src/ffi/registry.rs::compiled_count
unwired src/ffi/registry.rs::node_count
unwired src/ffi/types.rs::EvalStats
unwired src/ffi/types.rs::ShaderType
unwired src/ffi/types.rs::SoaBatchConfig
unwired src/font_bridge.rs::FontMetrics
unwired src/font_bridge.rs::char_to_sdf2d
unwired src/font_bridge.rs::font_metrics
unwired src/font_bridge.rs::glyph_to_sdf2d
unwired src/font_bridge.rs::text_to_sdf2d
unwired src/gi/cone_trace.rs::trace_hemisphere
unwired src/gi/irradiance.rs::get_probe
unwired src/gi/irradiance.rs::get_probe_mut
unwired src/gi/irradiance.rs::probe_count
unwired src/gi/mod.rs::PointLight
unwired src/interval.rs::is_negative
unwired src/interval.rs::is_positive
unwired src/interval.rs::overlaps
unwired src/material.rs::aluminum
unwired src/material.rs::chrome
unwired src/material.rs::concrete
unwired src/material.rs::copper
unwired src/material.rs::default_material
unwired src/material.rs::dielectric
unwired src/material.rs::emissive
unwired src/material.rs::find_by_name
unwired src/material.rs::from_material
unwired src/material.rs::glow
unwired src/material.rs::gold
unwired src/material.rs::marble
unwired src/material.rs::material_lerp
unwired src/material.rs::metal
unwired src/material.rs::plastic_red
unwired src/material.rs::plastic_white
unwired src/material.rs::rubber
unwired src/material.rs::skin
unwired src/material.rs::solid
unwired src/material.rs::to_particle
unwired src/material.rs::velvet
unwired src/material.rs::water
unwired src/material.rs::wet_asphalt
unwired src/material.rs::with_albedo_map
unwired src/material.rs::with_anisotropy
unwired src/material.rs::with_ao_map
unwired src/material.rs::with_clearcoat
unwired src/material.rs::with_color
unwired src/material.rs::with_emission
unwired src/material.rs::with_emissive_map
unwired src/material.rs::with_metallic
unwired src/material.rs::with_metallic_roughness_map
unwired src/material.rs::with_normal_map
unwired src/material.rs::with_sheen
unwired src/material.rs::with_subsurface
unwired src/material.rs::with_tiling
unwired src/material.rs::with_transmission
unwired src/material.rs::with_uv_channel
unwired src/material.rs::with_volume
unwired src/mesh/bvh.rs::expand_aabb
unwired src/mesh/bvh.rs::unsigned_distance_batch
unwired src/mesh/collision.rs::CollisionMesh
unwired src/mesh/collision.rs::compute_bounding_sphere
unwired src/mesh/collision.rs::compute_convex_hull
unwired src/mesh/collision.rs::convex_decomposition
unwired src/mesh/collision.rs::convex_hull_from_points
unwired src/mesh/collision.rs::simplify_collision
unwired src/mesh/collision.rs::total_triangles
unwired src/mesh/collision.rs::total_vertices
unwired src/mesh/collision.rs::volume
unwired src/mesh/decimate.rs::aggressive
unwired src/mesh/decimate.rs::conservative
unwired src/mesh/dual_contouring.rs::dual_contouring_compiled
unwired src/mesh/hermite.rs::HermiteExtractor::extract_edge_crossings
unwired src/mesh/hermite.rs::extract_edge_crossings
unwired src/mesh/hermite.rs::extract_hermite
unwired src/mesh/hermite.rs::extract_surface_points
unwired src/mesh/hermite.rs::t
unwired src/mesh/lightmap.rs::generate_lightmap_uvs
unwired src/mesh/lightmap.rs::generate_lightmap_uvs_fast
unwired src/mesh/lod.rs::DecimationLodConfig::distance_range
unwired src/mesh/lod.rs::LodConfig::distance_range
unwired src/mesh/lod.rs::balanced
unwired src/mesh/lod.rs::base_triangle_count
unwired src/mesh/lod.rs::generate_lod_chain
unwired src/mesh/lod.rs::generate_lod_chain_decimated
unwired src/mesh/lod.rs::get_blend_pair
unwired src/mesh/lod.rs::get_level
unwired src/mesh/lod.rs::get_lod
unwired src/mesh/lod.rs::get_render_meshes
unwired src/mesh/lod.rs::high_res
unwired src/mesh/lod.rs::resolution_at_level
unwired src/mesh/lod.rs::select_by_error
unwired src/mesh/lod_persist.rs::LodChainSummary
unwired src/mesh/lod_persist.rs::level_count
unwired src/mesh/lod_persist.rs::select_lod
unwired src/mesh/lod_persist.rs::summary
unwired src/mesh/lod_persist.rs::total_memory_bytes
unwired src/mesh/manifold.rs::MeshQuality
unwired src/mesh/manifold.rs::compute_quality
unwired src/mesh/manifold.rs::drop_specks
unwired src/mesh/manifold.rs::fill_holes
unwired src/mesh/manifold.rs::orient_faces
unwired src/mesh/mesh_codec.rs::decode_indices
unwired src/mesh/mesh_codec.rs::decode_positions
unwired src/mesh/mesh_codec.rs::encode_indices
unwired src/mesh/mesh_codec.rs::encode_mesh
unwired src/mesh/mesh_codec.rs::encode_positions
unwired src/mesh/mesh_sign.rs::cell_size
unwired src/mesh/mesh_sign.rs::dims
unwired src/mesh/mesh_to_sdf.rs::accurate
unwired src/mesh/mesh_to_sdf.rs::eval_unsigned
unwired src/mesh/mesh_to_sdf.rs::eval_unsigned_batch
unwired src/mesh/mesh_to_sdf.rs::gradient
unwired src/mesh/mesh_to_sdf.rs::hybrid
unwired src/mesh/mesh_to_sdf.rs::mesh_to_sdf_exact
unwired src/mesh/mesh_to_sdf.rs::sign_mode
unwired src/mesh/mesh_to_sdf.rs::to_sdf_node
unwired src/mesh/mesh_to_sdf.rs::topology_robust
unwired src/mesh/meshlet.rs::build_meshlets
unwired src/mesh/meshlet.rs::build_meshlets_adjacency
unwired src/mesh/meshlet.rs::build_meshlets_scan
unwired src/mesh/meshlet.rs::quality
unwired src/mesh/meshopt_filter.rs::decode_filter_exp_u32_in_place
unwired src/mesh/meshopt_filter.rs::decode_filter_oct_i16_in_place
unwired src/mesh/meshopt_filter.rs::decode_filter_quat_i16_in_place
unwired src/mesh/meshopt_filter.rs::encode_filter_exp_one
unwired src/mesh/meshopt_filter.rs::encode_filter_exp_u32
unwired src/mesh/meshopt_filter.rs::encode_filter_oct_i16
unwired src/mesh/meshopt_filter.rs::encode_filter_oct_one
unwired src/mesh/meshopt_filter.rs::encode_filter_quat_i16
unwired src/mesh/meshopt_filter.rs::encode_filter_quat_one
unwired src/mesh/meshopt_filter.rs::quantize_snorm
unwired src/mesh/meshopt_filter.rs::try_decode_filter_oct_i16_in_place
unwired src/mesh/meshopt_filter.rs::try_encode_filter_oct_i16
unwired src/mesh/meshopt_filter.rs::try_encode_filter_quat_i16
unwired src/mesh/meshopt_index_codec.rs::decode_index_buffer
unwired src/mesh/meshopt_vertex_codec.rs::decode_vertex_buffer
unwired src/mesh/meshopt_vertex_codec.rs::encode_vertex_buffer
unwired src/mesh/mod.rs::with_all
unwired src/mesh/nanite.rs::CLUSTER_MAX_VERTICES
unwired src/mesh/nanite.rs::from_normals
unwired src/mesh/nanite.rs::from_normals_and_positions
unwired src/mesh/nanite.rs::get_cluster
unwired src/mesh/nanite.rs::high_detail
unwired src/mesh/nanite.rs::is_backface_culled
unwired src/mesh/nanite.rs::is_visible
unwired src/mesh/nanite.rs::medium_detail
unwired src/mesh/nanite.rs::preview
unwired src/mesh/nanite.rs::select_clusters
unwired src/mesh/nanite.rs::should_render
unwired src/mesh/nanite.rs::total_vertices
unwired src/mesh/nanite.rs::unbounded
unwired src/mesh/optimize.rs::compute_acmr
unwired src/mesh/optimize.rs::compute_atvr
unwired src/mesh/optimize.rs::optimize_vertex_cache
unwired src/mesh/optimize.rs::optimize_vertex_fetch
unwired src/mesh/overdraw.rs::default_view_directions
unwired src/mesh/overdraw.rs::optimize_overdraw
unwired src/mesh/overdraw.rs::optimize_overdraw_with_views
unwired src/mesh/point_cloud_sdf.rs::accurate
unwired src/mesh/point_cloud_sdf.rs::point_cloud_to_sdf
unwired src/mesh/point_cloud_sdf.rs::point_count
unwired src/mesh/primitive_fitting.rs::PrimitiveType
unwired src/mesh/primitive_fitting.rs::compute_error
unwired src/mesh/primitive_fitting.rs::detect_primitive
unwired src/mesh/primitive_fitting.rs::fit_box
unwired src/mesh/primitive_fitting.rs::fit_cylinder
unwired src/mesh/primitive_fitting.rs::fit_plane
unwired src/mesh/primitive_fitting.rs::fit_sphere
unwired src/mesh/primitive_fitting.rs::primitive_type
unwired src/mesh/primitive_fitting.rs::primitives_to_csg
unwired src/mesh/primitive_fitting.rs::to_sdf_node
unwired src/mesh/quantization.rs::half_decode
unwired src/mesh/quantization.rs::half_encode
unwired src/mesh/quantization.rs::snorm_i16_decode
unwired src/mesh/quantization.rs::snorm_i8_decode
unwired src/mesh/quantization.rs::unorm_u16_decode
unwired src/mesh/quantization.rs::unorm_u8_decode
unwired src/mesh/sdf_to_mesh.rs::adaptive_marching_cubes_compiled
unwired src/mesh/spatial_order.rs::morton_3d
unwired src/mesh/spatial_order.rs::optimize_spatial_order
unwired src/mesh/stripifier.rs::stripify
unwired src/mesh/stripifier.rs::stripify_bound
unwired src/mesh/stripifier.rs::try_stripify
unwired src/mesh/stripifier.rs::unstripify
unwired src/mesh/stripifier.rs::unstripify_bound
unwired src/mesh/uv_unwrap.rs::compute_uv_density
unwired src/modifiers/surface_roughness.rs::hash3_xyz
unwired src/npr/compiled_color.rs::ADD
unwired src/npr/compiled_color.rs::BLOOM
unwired src/npr/compiled_color.rs::FRESNEL
unwired src/npr/compiled_color.rs::HATCH
unwired src/npr/compiled_color.rs::MULTIPLY
unwired src/npr/compiled_color.rs::N_DOT_L
unwired src/npr/compiled_color.rs::N_DOT_V
unwired src/npr/compiled_color.rs::OUTLINE_OVER
unwired src/npr/compiled_color.rs::PALETTE3
unwired src/npr/compiled_color.rs::PALETTE5
unwired src/npr/compiled_color.rs::POSTERIZE_COLOR
unwired src/npr/compiled_color.rs::PUSH_CONSTANT
unwired src/npr/compiled_color.rs::SATURATE
unwired src/npr/compiled_color.rs::SCALE
unwired src/npr/compiled_color.rs::SDF
unwired src/npr/compiled_color.rs::SOFT_TOON
unwired src/npr/compiled_color.rs::SPEED_LINE
unwired src/npr/compiled_color.rs::TIME_CYCLE
unwired src/npr/compiled_color.rs::TONEMAP
unwired src/npr/compiled_color.rs::TOON
unwired src/npr/compiled_color.rs::TWO_TONE
unwired src/npr/compiled_color.rs::UV_Y
unwired src/npr/compiled_color.rs::VIGNETTE
unwired src/npr/compiled_color.rs::as_words
unwired src/npr/compiled_color.rs::byte_len
unwired src/npr/compiled_color.rs::deserialize
unwired src/npr/compiled_color.rs::emit_wgsl_bytecode_evaluator
unwired src/npr/compiled_color.rs::fallback_op_count
unwired src/npr/compiled_color.rs::native_op_count
unwired src/npr/compiled_color.rs::opcode_word_count
unwired src/npr/compiled_color.rs::serialize
unwired src/npr/dsl.rs::bloom
unwired src/npr/dsl.rs::multiply
unwired src/npr/dsl.rs::plus
unwired src/npr/dsl.rs::posterize
unwired src/npr/dsl.rs::with_hatch
unwired src/npr/dsl.rs::with_speed_lines
unwired src/npr/outline.rs::depth_step_outline
unwired src/npr/outline.rs::distance_field_outline
unwired src/npr/scene_composer.rs::with_camera
unwired src/npr/scene_composer.rs::with_shading
unwired src/optimize.rs::optimization_stats
unwired src/physics_bridge.rs::arc
unwired src/physics_bridge.rs::sdf_to_physics_field
unwired src/physics_bridge.rs::with_epsilon
unwired src/primitives/mod.rs::PrimitiveType
unwired src/primitives/mod.rs::eval_primitive
unwired src/primitives/mod.rs::eval_primitive_unchecked
unwired src/python/helpers.rs::numpy_to_vec3_fast
unwired src/sim_bridge.rs::GpuPhysicsBundle
unwired src/sim_bridge.rs::add_erosion
unwired src/sim_bridge.rs::add_fracture
unwired src/sim_bridge.rs::add_modifier
unwired src/sim_bridge.rs::add_phase_change
unwired src/sim_bridge.rs::add_pressure
unwired src/sim_bridge.rs::add_thermal
unwired src/sim_bridge.rs::attach_physics
unwired src/sim_bridge.rs::clear_modifiers
unwired src/sim_bridge.rs::gpu_mesh_with_physics
unwired src/sim_bridge.rs::modifier_count
unwired src/sim_bridge.rs::modifier_mut
unwired src/sim_bridge.rs::simulate_sdf
unwired src/sim_bridge.rs::with_bounds
unwired src/soa.rs::SIMD_ALIGNMENT
unwired src/soa.rs::as_ptrs
unwired src/soa.rs::load_simd_unchecked
unwired src/soa.rs::store_simd_unchecked
unwired src/svo/linearize.rs::compact_svo
unwired src/svo/linearize.rs::linearize_svo
unwired src/svo/linearize.rs::nodes_at_level
unwired src/svo/linearize.rs::validate_linearized
unwired src/svo/mod.rs::child_count
unwired src/svo/mod.rs::linearize
unwired src/svo/mod.rs::nearest_surface
unwired src/svo/mod.rs::ray_query
unwired src/svo/query.rs::SvoRayHit
unwired src/svo/query.rs::svo_nearest_surface
unwired src/svo/query.rs::svo_ray_query
unwired src/svo/streaming.rs::hit_rate
unwired src/svo/streaming.rs::memory_used
unwired src/svo/streaming.rs::split_into_chunks
unwired src/svo/streaming.rs::with_memory_budget
unwired src/terrain/caves.rs::generate_cave_sdf
unwired src/terrain/caves.rs::generate_chamber
unwired src/terrain/clipmap.rs::ClipmapMesh
unwired src/terrain/clipmap.rs::generate_level_mesh
unwired src/terrain/clipmap.rs::generate_meshes
unwired src/terrain/clipmap.rs::level_count
unwired src/terrain/clipmap.rs::total_vertices
unwired src/terrain/heightmap.rs::from_data
unwired src/terrain/heightmap.rs::from_image
unwired src/terrain/heightmap.rs::from_image_bytes
unwired src/terrain/heightmap.rs::normal_at
unwired src/terrain/heightmap.rs::sample_bicubic
unwired src/terrain/mod.rs::terrain_sdf
unwired src/terrain/splatmap.rs::add_layer
unwired src/terrain/splatmap.rs::auto_splat_from_heightmap
unwired src/terrain/splatmap.rs::dominant_material
unwired src/terrain/splatmap.rs::get_weight
unwired src/terrain/splatmap.rs::layer_count
unwired src/terrain/splatmap.rs::set_weight
unwired src/texture/fitting.rs::reconstruct
unwired src/tight_aabb.rs::preset_medium
unwired src/volume/bake.rs::bake_volume_compiled
unwired src/volume/bake.rs::bake_volume_with_normals
unwired src/volume/export.rs::DdsFormat
unwired src/volume/export.rs::export_dds_3d
unwired src/volume/export.rs::export_dds_3d_distgrad
unwired src/volume/export.rs::export_raw
unwired src/volume/export.rs::export_raw_with_mips
unwired src/volume/gpu_bake.rs::gpu_bake_volume_with_normals
unwired src/volume/mipchain.rs::generate_mip_chain_distgrad
unwired src/volume/mod.rs::Volume3D<VoxelDistGrad>::sample_trilinear
unwired src/volume/mod.rs::Volume3D<f32>::sample_trilinear
unwired src/volume/mod.rs::mip_count
unwired src/volume/mod.rs::voxel_to_world
```

---

## What is the Wiring Guard?

The wiring guard ensures that all public items in `src/` are actually called from production code:

- **Dead Code Guard**: Verifies that `#[allow(dead_code)]` has a documented reason
- **Unwired Items**: Detects public functions, structs, etc. that are never called (except in tests)
- **Stale Baseline**: Ensures baseline entries are still needed
- **Brace Balance**: Checks syntax integrity

### Resolving Violations

1. **New violations**: Either implement/wire the item, or add to `scripts/wiring-baseline.txt`
2. **Baseline cleanup**: Remove lines from baseline as violations are resolved
3. **Comments**: Add `// ALLOW-DEAD:` or `// ALLOW-UNWIRED:` with reason (12+ chars)

For details: see `scripts/wiring_guard.py`
