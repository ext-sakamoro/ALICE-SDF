// Background thread of the page: the deepest chain each export of stack_canary.wasm evaluates before the engine
// stops it (a RangeError when the engine's stack runs out, a RuntimeError trap when the module's
// own stack runs out). Every attempt uses a fresh instance (a trap can leave the module's memory
// in any state). A call that ran must return a finite negative distance.
"use strict";

async function attempt(module, name, depth) {
  const instance = await WebAssembly.instantiate(module, {}); // a Module gives the Instance itself
  try {
    const d = instance.exports[name](depth);
    if (!(Number.isFinite(d) && d < 0)) return { ok: false, kind: "bad value " + d };
    return { ok: true };
  } catch (e) {
    return { ok: false, kind: e.constructor.name + ": " + e.message };
  }
}

async function deepest(module, name, ceiling) {
  // grow by doubling, then bisect between the last depth that ran and the first that did not
  let good = 0, bad = null, failure = null;
  for (let d = 64; d <= ceiling; d *= 2) {
    const r = await attempt(module, name, d);
    if (r.ok) good = d; else { bad = d; failure = r.kind; break; }
  }
  if (bad === null) return { name, deepest: good, stopped: null, attempts_capped_at: ceiling };
  while (bad - good > 1) {
    const mid = Math.floor((good + bad) / 2);
    const r = await attempt(module, name, mid);
    if (r.ok) good = mid; else { bad = mid; failure = r.kind; }
  }
  return { name, deepest: good, stopped: failure };
}

(async () => {
  try {
    const bytes = await (await fetch("stack_canary.wasm")).arrayBuffer();
    const module = await WebAssembly.compile(bytes);
    const results = [];
    for (const name of ["canary_eval", "canary_compile"]) {
      results.push(await deepest(module, name, 1 << 20));
    }
    postMessage({ userAgent: navigator.userAgent, results });
  } catch (e) {
    postMessage({ error: String(e) });
  }
})();
