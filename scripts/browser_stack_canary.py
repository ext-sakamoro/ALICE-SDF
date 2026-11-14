#!/usr/bin/env python3
"""Stack canary in a browser background thread: how deep a chain the tree walkers run in headless Chromium

Builds `examples/stack_canary_wasm.rs` for wasm32-unknown-unknown (the module's own stack is the
rust-lld default 1 MiB, as in the browser background thread that evaluates scenes), serves it with
`scripts/browser_stack_canary/` on a local port, and opens the page in headless Chromium. A
background thread calls `canary_eval` and `canary_compile` with growing depths and reports the
deepest chain each ran before the engine stopped it. Each must reach `--min-depth` (the depth the
wasm stack budget test holds: (1 MiB - 128 KiB) / 512 B per level = 1792).

Exit 1 when an export stays below the minimum, when no result came back (timeout), when nothing
was measured, or when no Chromium is found (`--chrome` or `$CHROME`, else the usual install paths).
`--known-defect NAME` reports that export's shortfall without failing, and fails once it reaches
the minimum (the defect is fixed: drop the flag); `--only NAME` measures and checks that export
alone.

Author: Moroya Sakamoto
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "scripts" / "browser_stack_canary"
MIN_DEPTH = (1024 * 1024 - 128 * 1024) // 512

CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def find_chrome(explicit: str | None) -> str | None:
    for c in [explicit, os.environ.get("CHROME"), *CANDIDATES]:
        if not c:
            continue
        p = shutil.which(c) or (c if Path(c).is_file() else None)
        if p:
            return p
    return None


def build_module() -> Path:
    cmd = [
        "cargo", "build", "--release", "--example", "stack_canary_wasm",
        "--target", "wasm32-unknown-unknown", "--no-default-features", "--message-format=json",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"browser-canary: FAIL the wasm build exited {r.returncode}")
    out = r.stdout
    for line in out.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") == "compiler-artifact" and m["target"]["name"] == "stack_canary_wasm":
            wasm = [f for f in m["filenames"] if f.endswith(".wasm")]
            if wasm:
                return Path(wasm[0])
    raise SystemExit("browser-canary: FAIL the build gave no stack_canary_wasm .wasm")


def serve_and_run(site: Path, chrome: str, timeout: float) -> dict | None:
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(site), **k)

        def do_POST(self):  # noqa: N802 (http.server naming)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            result["data"] = json.loads(body or b"{}")
            self.send_response(204)
            self.end_headers()
            done.set()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen(
            [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
             f"--user-data-dir={profile}", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            done.wait(timeout)
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            httpd.shutdown()
    return result.get("data")


EXPORTS = ("canary_eval", "canary_compile")


def check(
    data: dict | None, min_depth: int, known: frozenset[str] = frozenset(), only: str | None = None
) -> tuple[list[str], list[str]]:
    report, fails = [], []
    if data is None:
        return report, ["no result from the background thread (timeout)"]
    if "error" in data:
        return report, [f"thread error: {data['error']}"]
    results = data.get("results") or []
    wanted = (only,) if only else EXPORTS
    names = {r.get("name") for r in results}
    for name in wanted:
        if name not in names:
            fails.append(f"{name}: not measured")
    for r in results:
        name = r.get("name")
        if name not in wanted:
            continue
        depth, stopped = r.get("deepest", 0), r.get("stopped")
        report.append(f"{name}: deepest {depth} levels, stopped by {stopped}")
        if name in known:
            if depth >= min_depth:
                fails.append(
                    f"{name}: known defect no longer reproduces ({depth} >= {min_depth} levels): "
                    "remove --known-defect"
                )
            else:
                report.append(f"{name}: known defect, {depth} levels < {min_depth} (not failing)")
        elif depth < min_depth:
            fails.append(f"{name}: {depth} levels < {min_depth}")
    return report, fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--chrome")
    ap.add_argument("--min-depth", type=int, default=MIN_DEPTH)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--known-defect", action="append", default=[], choices=EXPORTS)
    ap.add_argument("--only", choices=EXPORTS)
    a = ap.parse_args(argv)
    chrome = find_chrome(a.chrome)
    if chrome is None:
        print("browser-canary: FAIL no Chromium found (--chrome or $CHROME)", file=sys.stderr)
        return 1
    wasm = build_module()
    with tempfile.TemporaryDirectory() as site:
        for f in ("index.html", "thread.js"):
            shutil.copy(PAGE / f, Path(site) / f)
        shutil.copy(wasm, Path(site) / "stack_canary.wasm")
        data = serve_and_run(Path(site), chrome, a.timeout)
    report, fails = check(data, a.min_depth, frozenset(a.known_defect), a.only)
    if data and "userAgent" in data:
        print(f"browser-canary: {data['userAgent']}")
    for line in report:
        print(f"browser-canary: {line}")
    for line in fails:
        print(f"browser-canary: FAIL {line}", file=sys.stderr)
    print(f"browser-canary: {len(report)} measured, {len(fails)} failure(s), minimum {a.min_depth} levels")
    return 1 if fails or not report else 0


if __name__ == "__main__":
    sys.exit(main())
