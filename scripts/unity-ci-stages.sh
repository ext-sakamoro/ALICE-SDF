#!/usr/bin/env bash
# Runs the Unity checks of vrchat-package (AliceSDF.Editor.AliceSDF_CiChecks
# .RunBatch, one Unity invocation per stage) in the GameCI editor image, the
# way the `unity-vrchat` CI job does. scripts/unity-preflight.ps1 is the
# local (installed Unity) equivalent; keep the stage list in step.
#
#   UNITY_LICENSE=<contents of a Unity_lic.ulf> scripts/unity-ci-stages.sh <project dir>
#
# Each stage's Unity log lands in ci-unity-logs/<stage>.log.
#
# The licence file is a Personal licence activated for the editor version
# (Unity Hub / the GameCI activation page); it is written into the container
# only. Exit 0 = every stage passed, 1 = a stage failed (its log is the
# report: grep for [ALICE-CI]).
set -euo pipefail
cd "$(dirname "$0")/.."
proj="${1:-ci-unity}"
image="unityci/editor:ubuntu-2022.3.22f1-android-3"
[[ -n "${UNITY_LICENSE:-}" ]] || { echo "unity-ci-stages: UNITY_LICENSE is empty" >&2; exit 1; }
[[ -f "$proj/Packages/vpm-manifest.json" ]] || { echo "unity-ci-stages: $proj is not set up (scripts/unity-ci-project.sh)" >&2; exit 1; }

# Unity rejects a -logFile path containing ".." ("... is not a valid directory
# name"), so the stage logs get their own directory next to the project
logdir="ci-unity-logs"
mkdir -p "$logdir"

run_stage() {  # stage, build target
    local stage="$1" target="$2" log="$logdir/${1}.log"
    echo "== stage $stage ($target)"
    # the editor writes as root inside the container; the checkout is chowned back below
    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e UNITY_LICENSE -e ALICE_CI_STAGE="$stage" \
        "$image" bash -c '
            set -e
            mkdir -p "$HOME/.local/share/unity3d/Unity"
            printf "%s" "$UNITY_LICENSE" > "$HOME/.local/share/unity3d/Unity/Unity_lic.ulf"
            unity-editor -batchmode -nographics -quit \
                -projectPath "/work/'"$proj"'" -buildTarget "'"$target"'" \
                -executeMethod AliceSDF.Editor.AliceSDF_CiChecks.RunBatch \
                -logFile "/work/'"$logdir"'/'"$stage"'.log"' \
        && rc=0 || rc=$?
    grep -E '\[ALICE-CI\]|error CS|Shader error' "$log" || true
    if [[ $rc -ne 0 ]]; then echo "unity-ci-stages: stage $stage FAILED (exit $rc), log $log" >&2; return 1; fi
}

marker="$proj/Library/alice_ci_kit.txt"
rm -f "$marker"
run_stage setup Linux64
run_stage compile Linux64
run_stage samples Linux64
run_stage scenes Linux64
for i in 1 2 3 4 5 6; do
    run_stage kit Linux64
    state="$(cat "$marker" 2>/dev/null || true)"
    case "$state" in
        done*) echo "  kit: $state"; break ;;
        failed) echo "unity-ci-stages: kit build failed" >&2; exit 1 ;;
    esac
    [[ $i -lt 6 ]] || { echo "unity-ci-stages: kit build did not finish in 6 runs" >&2; exit 1; }
done
run_stage verify Linux64
run_stage android Android
echo "unity-ci-stages: all stages passed"
