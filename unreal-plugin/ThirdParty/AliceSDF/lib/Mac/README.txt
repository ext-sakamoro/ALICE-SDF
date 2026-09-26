libalice_sdf.dylib belongs here.

It is a build product and is NOT in the repository: a committed binary
goes stale silently (the one that used to sit here was seven months and one
law change behind the source, and every user who built the plugin from a
clone linked against it).

Get them one of two ways:

  1. The release zip — https://github.com/ext-sakamoro/ALICE-SDF/releases
     AliceSDF-UE5-Plugin-macOS.zip already contains this directory filled in.

  2. Build them:
       cargo build --release --features unreal --target aarch64-apple-darwin
       copy target\x86_64-pc-windows-msvc\release\alice_sdf.dll     <here>\alice_sdf.dll
     `scripts/build_ue5_plugin.sh` does this for the host platform.

The `unreal` feature is the right one: it is ffi + hlsl + glsl + gpu, and the
plugin calls all four (GenerateHlsl / GenerateGlsl / GenerateWgsl). A library
built with a narrower set fails to link.

The module logs which library it loaded and its version at startup:
  LogTemp: ALICE-SDF: native library 3.1.0 loaded from ...
