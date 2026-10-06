# Security Policy

## Supported versions

Security fixes are made on the latest minor release of the newest major
version of `alice-sdf` on crates.io. Older versions are not patched.

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Report it privately by email
to <sakamoro@alicelaw.net> with the subject
`[SECURITY] ALICE-SDF: <short description>`.

Include the version, the enabled features, a way to reproduce the problem (a
tree or file that triggers it is ideal) and the impact you expect. You will get
an acknowledgement, and the fix and its release are coordinated with you before
the details are published.

## Things to know when you handle untrusted input

### Loading trees and files

- `.asdf` (binary) input is decoded with a size limit, so a forged length
  prefix is rejected instead of allocating without bound. Other formats
  (`.asdf.json`, meshes, voxel and splat files) have no such limit; check the
  size of untrusted files before loading them.
- A tree is evaluated recursively, so a very deep tree from an untrusted source
  can exhaust the stack during evaluation or compilation. Dropping a deep tree
  does not recurse (`tests/deep_tree_drop.rs`). Limit the depth of trees you
  accept.
- The decoders and the evaluators are fuzzed (`fuzz/fuzz_targets/`).

### Generated code

The shader transpilers and the `rust` feature emit source code from a tree.
Numeric parameters are written as literals, but treat code generated from an
untrusted tree like any other untrusted code before you compile and run it.

### `unsafe` code

`unsafe` is used for SIMD loads in the structure-of-arrays evaluators, for
`primitives::eval_primitive_unchecked` (the caller guarantees the parameter
count), for calling code produced by the Cranelift JIT (`jit` feature), for the
C ABI (`ffi` feature), for the Python bindings (`python` feature) and for
writing volume textures (`volume` feature).

### Dependencies

CI runs `cargo audit` and `cargo deny` in `.github/workflows/security-audit.yml`.
The `physics`, `codec` and `sdf-cache` features pull in additional crates;
audit the build with the features you actually enable.
