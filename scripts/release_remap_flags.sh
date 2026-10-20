#!/usr/bin/env bash
# scripts/release_remap_flags.sh — the rustc flags that keep the build machine's paths out of a
# shipped library.
#
# A release build records source paths in panic messages and debug info: this checkout's (and
# the sibling checkouts next to it), and the dependency sources under the cargo home
# (`<home>/.cargo/registry/src/...`), which name the user. Printed on one line for RUSTFLAGS:
#
#   <parent of this checkout> -> /src     <cargo home> -> /cargo     /root/.cargo -> /cargo
#
# (/root/.cargo: the cargo home inside the manylinux container the Linux wheels are built in).
# On Windows the paths are written as rustc sees them (C:\...). A path with a space cannot be
# passed in RUSTFLAGS and exits 2. The symbols are stripped by `[profile.release] strip = true`.
# scripts/release_artifact_audit.py checks what is shipped.
#
# usage: RUSTFLAGS="$(scripts/release_remap_flags.sh)" cargo build --release ...
set -euo pipefail
cd "$(dirname "$0")/.."

native() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

parent=$(native "$(dirname "$PWD")")
cargo_home=$(native "${CARGO_HOME:-$HOME/.cargo}")
for p in "$parent" "$cargo_home"; do
  if [[ "$p" == *" "* ]]; then
    echo "release_remap_flags.sh: '$p' has a space and cannot be passed in RUSTFLAGS" >&2
    exit 2
  fi
done
printf -- '--remap-path-prefix=%s=/src --remap-path-prefix=%s=/cargo --remap-path-prefix=/root/.cargo=/cargo\n' \
  "$parent" "$cargo_home"
