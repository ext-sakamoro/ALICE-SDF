#!/usr/bin/env python3
"""The GPU slot guard is in every test build of this crate and in no build of an application.

`builds <cargo-json>`: the `--message-format=json` output of `cargo test --no-run --features gpu ...`. Every
alice-sdf library artifact in it must carry the `gpu-slot-guard` feature (this crate's dev-dependency on itself
turns it on), and at least one test executable must have been built. Zero executables, or zero alice-sdf library
artifacts, is a failure.

`consumer <bin-without> <bin-with>`: two release binaries of a small application that depends on alice-sdf with
`gpu` (and, for the second, also `gpu-slot-guard`) and requests a GPU context. The first must not contain the
variable name ALICE_GPU_SLOT (no application build can be refused), the second must (the check has teeth).

`artifact <shipped-library> <bin-with>`: this crate's own release library, built with `cargo build --release` in
this repository as the plugins ship it, must not contain ALICE_GPU_SLOT (the guard reaches only test, example and
bench builds through the dev-dependency on itself, never a `cargo build`); `<bin-with>` is the consumer build with
the feature, which must contain it (the check has teeth). A missing or empty library is a failure.
"""
import json
import pathlib
import sys

VAR = b"ALICE_GPU_SLOT"
FEATURE = "gpu-slot-guard"
LIB_KINDS = {"lib", "rlib", "cdylib", "staticlib", "dylib"}


def builds(path: str) -> int:
    libs = 0
    without = []
    executables = 0
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("reason") != "compiler-artifact":
            continue
        target = msg.get("target", {})
        # this crate's library target is ["cdylib", "rlib"], not "lib"
        if target.get("name") == "alice_sdf" and LIB_KINDS & set(target.get("kind", [])):
            libs += 1
            if FEATURE not in msg.get("features", []):
                without.append(msg.get("package_id", "?"))
        if msg.get("executable") and msg.get("profile", {}).get("test"):
            executables += 1
    print(f"gpu-slot-guard builds: {libs} alice-sdf library artifacts, {executables} test executables, "
          f"{len(without)} library artifacts without {FEATURE}")
    if libs == 0 or executables == 0:
        print("gpu-slot-guard builds: nothing was checked")
        return 1
    return 1 if without else 0


def consumer(bin_without: str, bin_with: str) -> int:
    a = pathlib.Path(bin_without).read_bytes().count(VAR)
    b = pathlib.Path(bin_with).read_bytes().count(VAR)
    print(f"gpu-slot-guard consumer: {VAR.decode()} occurs {a} times without the feature, {b} times with it")
    if b == 0:
        print("gpu-slot-guard consumer: the build with the feature has no reference either: the check has no teeth")
        return 1
    return 1 if a else 0


def artifact(library: str, bin_with: str) -> int:
    data = pathlib.Path(library).read_bytes()
    a = data.count(VAR)
    b = pathlib.Path(bin_with).read_bytes().count(VAR)
    print(f"gpu-slot-guard artifact: {library} ({len(data)} bytes): {VAR.decode()} occurs {a} times; "
          f"{b} times in the build with the feature")
    if not data or b == 0:
        print("gpu-slot-guard artifact: nothing compared (empty library, or the reference has no reference)")
        return 1
    return 1 if a else 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "builds":
        return builds(sys.argv[2])
    if len(sys.argv) == 4 and sys.argv[1] == "consumer":
        return consumer(sys.argv[2], sys.argv[3])
    if len(sys.argv) == 4 and sys.argv[1] == "artifact":
        return artifact(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
