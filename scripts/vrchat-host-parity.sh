#!/usr/bin/env bash
# Host-side parity of the VRChat Mochi sample: Rust golden -> C# collider.
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

golden="$(mktemp -t mochi-golden.XXXXXX)"
trap 'rm -f "$golden"' EXIT

cargo run -q --release --example vrchat_mochi_golden > "$golden"
lines=$(wc -l < "$golden")
[[ "$lines" -eq 1521 ]] || { echo "golden has $lines lines, expected 1521" >&2; exit 1; }

proj="vrchat-package/HostTests~/MochiParity"
dotnet build "$proj" -c Release --nologo -v q
dotnet run --no-build -c Release --project "$proj" -- "$golden"
