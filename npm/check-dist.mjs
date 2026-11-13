// After a build: the files package.json names as the package's entry points exist in dist/
// (wasm-pack names its output after the crate unless --out-name is given; a mismatch installs a
// package whose entry is missing). Fails with the missing files listed.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(fs.readFileSync(path.join(here, 'package.json'), 'utf8'));
const named = [pkg.main, pkg.module, pkg.types, 'dist/alice_sdf_bg.wasm'].filter(Boolean);
const missing = named.filter((f) => !fs.existsSync(path.join(here, f)));
if (named.length === 0 || missing.length > 0) {
  console.error(`check-dist: missing ${missing.join(', ') || '(package.json names no entry)'}`);
  process.exit(1);
}
console.log(`check-dist: ${named.length} entry files present (${named.join(', ')})`);
