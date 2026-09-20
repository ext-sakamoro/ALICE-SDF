#!/usr/bin/env bash
# Host-side parity of the VRChat interactive samples: Rust golden -> C# collider,
# one (example, HostTests~ project, golden line count) triple per sample.
# Mirrors the `vrchat-host` job of .github/workflows/ci.yml; preflight.sh
# calls this, so it must stay runnable with only cargo + dotnet installed.
set -euo pipefail
cd "$(dirname "$0")/.."

# Homebrew formula install location, for shells (git hooks) whose PATH lacks it
if ! command -v dotnet >/dev/null 2>&1 && [[ -x /opt/homebrew/opt/dotnet/bin/dotnet ]]; then
    export PATH="/opt/homebrew/opt/dotnet/bin:$PATH"
    export DOTNET_ROOT="/opt/homebrew/opt/dotnet/libexec"
fi
if ! command -v dotnet >/dev/null 2>&1; then
    # Homebrew formula (no sudo); the dotnet-sdk cask needs an interactive sudo
    echo "vrchat-host-parity: dotnet not found (brew install dotnet, then export PATH=/opt/homebrew/opt/dotnet/bin:\$PATH)" >&2
    exit 1
fi

golden="$(mktemp -t vrchat-golden.XXXXXX)"
trap 'rm -f "$golden"' EXIT

# example : HostTests~ project : lines the golden must have (the grid size; a
# silently truncated golden would otherwise pass)
samples=(
    "vrchat_mochi_golden:MochiParity:1521"
    "vrchat_terrain_sculpt_golden:TerrainSculptParity:4335"
    "vrchat_deformable_wall_golden:DeformableWallParity:5100"
    "vrchat_basic_golden:StaticParity:2535"
    "vrchat_cosmic_golden:StaticParity:2907"
    "vrchat_fractal_golden:StaticParity:3375"
    "vrchat_mix_golden:StaticParity:1989"
)

for entry in "${samples[@]}"; do
    IFS=: read -r example project expected <<< "$entry"
    echo "== $project ($example)"
    cargo run -q --release --example "$example" > "$golden"
    lines=$(wc -l < "$golden")
    [[ "$lines" -eq "$expected" ]] || { echo "$example golden has $lines lines, expected $expected" >&2; exit 1; }

    proj="vrchat-package/HostTests~/$project"
    dotnet build "$proj" -c Release --nologo -v q
    # the example name lets a project that hosts several samples pick one
    dotnet run --no-build -c Release --project "$proj" -- "$golden" "$example"
done

# The Kit product scripts: the eight sample / base colliders renamed the way
# KitProductBuilder renames them, compiled together against the stub and
# checked by reflection (HostTests~/KitCompile/Compiled). Catches a rename
# that leaves a dangling reference without a Unity licence.
echo "== KitCompile (Kit-renamed scripts)"
kit="vrchat-package/HostTests~/KitCompile"
dotnet build "$kit" -c Release --nologo -v q
dotnet run --no-build -c Release --project "$kit"
dotnet build "$kit/Compiled" -c Release --nologo -v q
dotnet run --no-build -c Release --project "$kit/Compiled"
