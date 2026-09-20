#!/usr/bin/env bash
# Creates the throw-away VRChat Worlds project the `unity-vrchat` CI job runs
# the Unity checks in: Unity 2022.3.22f1 project skeleton from
# vrchat-package/CI~, com.vrchat.worlds resolved with vrc-get (the VPM
# resolver as a single binary), com.alice.sdf linked as a file: dependency.
#
#   scripts/unity-ci-project.sh <project dir>     (default: ci-unity)
#
# Idempotent: re-running keeps an existing Library/. The SDK version is
# pinned so a VRChat release does not turn CI red on its own; bump it here.
set -euo pipefail
cd "$(dirname "$0")/.."
proj="${1:-ci-unity}"
sdk_version="3.10.5"
vrc_get_version="1.9.2"

mkdir -p "$proj/Assets" "$proj/Packages" "$proj/ProjectSettings"
cp vrchat-package/CI~/ProjectVersion.txt "$proj/ProjectSettings/ProjectVersion.txt"
cp vrchat-package/CI~/manifest.json "$proj/Packages/manifest.json"
# the template links ../../vrchat-package relative to Packages/: true for a
# project directly under the repository root
[[ -d "$proj/../vrchat-package" ]] || { echo "unity-ci-project: $proj must sit directly under the repository root (file: link)" >&2; exit 1; }

bin="$(mktemp -d)/vrc-get"
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  asset="x86_64-unknown-linux-musl-vrc-get" ;;
    Darwin-arm64)  asset="aarch64-apple-darwin-vrc-get" ;;
    Darwin-x86_64) asset="x86_64-apple-darwin-vrc-get" ;;
    *) echo "unity-ci-project: no vrc-get binary for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
curl -sSL --retry 3 -o "$bin" "https://github.com/vrc-get/vrc-get/releases/download/v${vrc_get_version}/${asset}"
chmod +x "$bin"
"$bin" install com.vrchat.worlds "$sdk_version" --project "$proj" --yes
[[ -d "$proj/Packages/com.vrchat.worlds" && -d "$proj/Packages/com.vrchat.base" ]] || { echo "unity-ci-project: SDK packages not installed" >&2; exit 1; }
echo "unity-ci-project: $proj ready (com.vrchat.worlds $sdk_version, com.alice.sdf -> vrchat-package)"
