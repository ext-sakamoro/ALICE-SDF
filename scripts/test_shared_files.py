#!/usr/bin/env python3
"""Tests of scripts/shared_files.sh (python3 scripts/test_shared_files.py)

The script fetches ALICE-Law at the pinned revision. Without an argument it checks the
committed tree, which passes; given an export of HEAD with a flipped byte in a shared copy it
fails, and a pin with a wrong SHA-256 or a malformed pin fails before anything is compared.

The same file runs in every crate that carries the script, public (listed in ALICE-Law's table)
or private (`visibility private` in its pin, never listed): which shared copies a tree holds and
which repository names the table lists are read from the table at the pinned revision, not
written here.

ALICE-Law is fetched once per run: setUpClass fetches the pinned revision (bounded retries with
backoff) into a local repository, and every run of the script reads it from there through a git
url rewrite in its environment. A fetch that still fails is one named error, never a skip and
never one failure per test. The table's rows are of two kinds: shared rows (`repos = [...]`) and
own rows (`own = "..."`, a path each repository holds as its own); a row of neither kind fails,
and so does a table with no shared row."""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / ".github/shared-files-pin"
# the copies these tests rewrite themselves: never the probe of a flipped byte
SELF = ("scripts/shared_files.sh", "scripts/test_shared_files.py")
LAW_URL = "https://github.com/ext-sakamoro/ALICE-Law"
# the table paths under it are shared workflows: a private holder does not hold them
HOSTED_TRIGGER = ".github/workflows/"
# waits between the fetch attempts of the pinned revision (seconds): 4 attempts in all
FETCH_BACKOFF = (2, 4, 8)


def bash() -> str:
    """Git's bash on Windows (`bash` on PATH there can be the WSL launcher, which has no
    distribution on the runners), the PATH one elsewhere."""
    if os.name == "nt":
        exec_path = subprocess.run(["git", "--exec-path"], check=True, capture_output=True, text=True).stdout.strip()
        for up in (3, 2):
            candidate = Path(exec_path).parents[up - 1] / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(f"no bash.exe of Git near {exec_path}")
    found = shutil.which("bash")
    if found is None:
        raise RuntimeError("no bash on PATH")
    return found


def pin_value(key: str) -> str:
    values = [l.split()[1] for l in PIN.read_text(encoding="utf-8").splitlines()
              if l.startswith(key + " ") and len(l.split()) == 2]
    if len(values) != 1:
        raise RuntimeError(f"{PIN}: no single '{key} <value>' line")
    return values[0]


def fetch_law(dest: Path, rev: str, backoff=FETCH_BACKOFF, run=subprocess.run, sleep=time.sleep) -> None:
    """ALICE-Law at `rev` into the bare repository `dest` (branch `pinned`), retried after each wait
    of `backoff`; a fetch that still fails raises one error naming the revision and the last output"""
    run(["git", "init", "-q", "--bare", str(dest)], check=True)
    last = ""
    for wait in (*backoff, None):
        r = run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", LAW_URL, rev], capture_output=True, text=True)
        if r.returncode == 0:
            run(["git", "-C", str(dest), "update-ref", "refs/heads/pinned", "FETCH_HEAD"], check=True)
            return
        last = (r.stderr or r.stdout or "").strip()
        if wait is not None:
            sleep(wait)
    raise RuntimeError(f"ALICE-Law {rev}: fetch failed after {len(backoff) + 1} attempts: {last[-300:]}")


def law_env(law: Path) -> dict[str, str]:
    """the environment that sends git's fetches of ALICE-Law to the local repository `law`"""
    env = dict(os.environ)
    n = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
    env[f"GIT_CONFIG_KEY_{n}"] = f"url.{law.resolve().as_uri()}.insteadOf"
    env[f"GIT_CONFIG_VALUE_{n}"] = LAW_URL
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    return env


def parse_table(data: bytes) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """(shared rows as (path, listed repositories), own paths) of the table's bytes: a row is shared
    (`repos = [...]`) or own (`own = "..."`); a row of neither kind, of both, or a table with no
    shared row is refused. Fills UNIFORM with each uniform row's SHA-256"""
    rows, own = [], []
    for block in data.decode("utf-8").split("[[file]]")[1:]:
        path = re.search(r'^path = "([^"]+)"$', block, re.M)
        repos = re.search(r"^repos = \[([^\]]*)\]$", block, re.M)
        mine = re.search(r'^own = "([^"]+)"$', block, re.M)
        if path is None or (repos is None) == (mine is None):
            raise RuntimeError(f"a table row that is neither shared (a repos line) nor own (an own line): {block[:80]!r}")
        if mine is not None:
            own.append(path.group(1))
            continue
        rows.append((path.group(1), re.findall(r'"([^"]+)"', repos.group(1))))
        sha = re.search(r'^sha256 = "([0-9a-f]{64})"$', block, re.M)
        if sha is not None:
            UNIFORM[path.group(1)] = sha.group(1)
    if not rows:
        raise RuntimeError(f"the pinned table has no shared row ({len(own)} own row(s))")
    return rows, own


def pinned_table(law: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """the rows of ALICE-Law's table at the pinned revision (read from the local fetch `law`), refused
    unless its SHA-256 is the pin's (the script fetches the same bytes)"""
    data = subprocess.run(["git", "-C", str(law), "show", "pinned:conformance/shared-files.toml"], check=True,
                          capture_output=True).stdout
    if hashlib.sha256(data).hexdigest() != pin_value("table-sha256"):
        raise RuntimeError("the table at the pinned revision is not the pin's table-sha256")
    return parse_table(data)


# path -> the bytes' SHA-256 of each uniform row of the pinned table (filled by pinned_table)
UNIFORM: dict[str, str] = {}


class SharedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.law_tmp = tempfile.TemporaryDirectory()
        law = Path(cls.law_tmp.name) / "law.git"
        try:
            fetch_law(law, pin_value("rev"))
            cls.rows, cls.own = pinned_table(law)
        except BaseException:
            cls.law_tmp.cleanup()
            raise
        cls.env = law_env(law)
        print(f"pinned table: {len(cls.rows)} shared row(s), {len(cls.own)} own row(s)")

    @classmethod
    def tearDownClass(cls):
        cls.law_tmp.cleanup()

    def present(self, tree: Path) -> list[str]:
        """the table paths this tree holds"""
        return [p for p, _ in self.rows if (tree / p).is_file()]

    def probes(self, tree: Path) -> list[str]:
        """the table paths this tree holds whose row records bytes (some repository holds it: a row no
        repository holds is per-repository data, not compared), in table order"""
        compared = [p for p, repos in self.rows if repos and p not in SELF]
        found = [p for p in compared if (tree / p).is_file()]
        self.assertTrue(found, f"{tree} holds no shared copy whose row records bytes")
        return found

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.tree.mkdir()
        # no line-ending conversion: on Windows git archive would otherwise write CRLF
        archive = subprocess.run(["git", "-C", str(ROOT), "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                                  "archive", "HEAD"], check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", str(self.tree)], input=archive.stdout, check=True)
        # the committed bytes, as the rest of the tree: a working copy checked out with CRLF (Windows without an
        # eol attribute) would be a finding of its own
        for rel in ("scripts/shared_files.sh", ".github/shared-files-pin"):
            shown = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], check=True, capture_output=True)
            (self.tree / rel).write_bytes(shown.stdout)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args: str, script_root: Path | None = None):
        root = script_root or self.tree
        r = subprocess.run([bash(), str(root / "scripts/shared_files.sh"), *args], capture_output=True, text=True,
                           env=self.env)
        return r.returncode, r.stdout + r.stderr

    def test_the_committed_tree_passes_and_compares_something(self):
        code, out = self.run_script(script_root=ROOT)
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"copies [1-9][0-9]*, pending [0-9]+, 0 finding")

    def test_an_export_of_head_passes(self):
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 0, out)

    def test_a_flipped_byte_in_a_shared_copy_fails(self):
        # the first shared copy this tree holds whose row records bytes
        rel = self.probes(self.tree)[0]
        f = self.tree / rel
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(bytes(data))
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 1, out)
        self.assertIn(rel, out)
        # exactly the flipped file: a red for every file would also be exit 1
        self.assertIn(", 1 finding(s)", out)

    def test_a_wrong_checker_sha256_in_the_pin_fails(self):
        pin = self.tree / ".github/shared-files-pin"
        lines = pin.read_text(encoding="utf-8").splitlines()
        lines = [f"checker-sha256 {'0' * 64}" if l.startswith("checker-sha256 ") else l for l in lines]
        pin.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)

    def test_a_malformed_pin_fails(self):
        (self.tree / ".github/shared-files-pin").write_text("rev main\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)

    def write_pin(self, name: str, visibility: str | None) -> None:
        """The committed pin with another repository name and, if given, a visibility line."""
        pin = self.tree / ".github/shared-files-pin"
        lines = [l for l in pin.read_text(encoding="utf-8").splitlines()
                 if not l.startswith(("name ", "visibility "))]
        lines.insert(0, f"name {name}")
        if visibility is not None:
            lines.append(f"visibility {visibility}")
        pin.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def unlisted_name(self) -> str:
        """a repository name the table lists nowhere (as a private holder's)"""
        listed = {r for _, repos in self.rows for r in repos}
        name = "ALICE-PrivateProbe"
        while name in listed:
            name += "X"
        return name

    def private_tree(self) -> Path:
        """A private holder's tree: every table path the committed tree holds (shared copies and
        per-repository data alike, as a private repository carries them), at the committed bytes,
        and nothing else. Shared workflows are left out: they trigger a hosted run, which a private
        repository does not hold (the private mode makes holding one a finding)."""
        tree = Path(self.tmp.name) / "private"
        for rel in self.present(self.tree):
            if rel.startswith(HOSTED_TRIGGER):
                continue
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.tree / rel, tree / rel)
        return tree

    def test_a_private_repository_compares_its_copies_with_the_rows(self):
        # a name the table never lists: the private mode compares each table path present
        self.write_pin(self.unlisted_name(), "private")
        code, out = self.run_script(str(self.private_tree()))
        self.assertEqual(code, 0, out)
        self.assertIn("private copy, the row's bytes", out)
        self.assertRegex(out, r"copies [1-9][0-9]*, pending [0-9]+, 0 finding")

    def test_a_shared_workflow_in_a_private_tree_is_a_finding(self):
        # a shared workflow row's path (the bytes do not matter: holding one is the finding)
        self.write_pin(self.unlisted_name(), "private")
        tree = self.private_tree()
        rel = next(p for p, repos in self.rows if repos and p.startswith(HOSTED_TRIGGER))
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        src = self.tree / rel
        (tree / rel).write_bytes(src.read_bytes() if src.is_file() else b"on: push\n")
        code, out = self.run_script(str(tree))
        self.assertEqual(code, 1, out)
        self.assertIn(f"{rel}: a shared workflow is not held by a private repository", out)
        self.assertIn(", 1 finding(s)", out)

    def test_a_flipped_byte_fails_in_the_private_mode_too(self):
        self.write_pin(self.unlisted_name(), "private")
        tree = self.private_tree()
        rel = self.probes(tree)[0]
        f = tree / rel
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(bytes(data))
        code, out = self.run_script(str(tree))
        self.assertEqual(code, 1, out)
        self.assertIn(rel, out)
        self.assertIn(", 1 finding(s)", out)

    def test_every_shared_copy_of_a_private_holder_is_compared(self):
        # a flipped byte in each probe of the private tree is a finding on its own (not only the first)
        self.write_pin(self.unlisted_name(), "private")
        for rel in self.probes(self.private_tree()):
            with self.subTest(rel=rel):
                tree = self.private_tree()
                f = tree / rel
                data = bytearray(f.read_bytes())
                data[-1] ^= 1
                f.write_bytes(bytes(data))
                code, out = self.run_script(str(tree))
                self.assertEqual(code, 1, out)
                self.assertIn(f"{rel}: sha256 ", out)
                self.assertIn(", 1 finding(s)", out)
                shutil.rmtree(tree)

    def test_a_listed_public_repository_cannot_use_the_private_mode(self):
        # a repository the table lists (the first named by a row), so the private mode would dodge
        # its rows: refused
        name = next(r for _, repos in self.rows for r in repos)
        self.write_pin(name, "private")
        code, out = self.run_script(str(self.tree))
        self.assertEqual(code, 2, out)

    def converged_tree(self) -> tuple[Path, list[str]]:
        """A tree holding only shared copies whose bytes are their uniform row's (taken from the committed
        tree): converged by construction, whatever state the host repository's own copies are in."""
        tree = Path(self.tmp.name) / "converged"
        held = []
        for rel in self.probes(self.tree):
            data = (self.tree / rel).read_bytes()
            if UNIFORM.get(rel) == hashlib.sha256(data).hexdigest():
                (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                (tree / rel).write_bytes(data)
                held.append(rel)
        self.assertTrue(held, "the committed tree holds no copy with its uniform row's bytes")
        return tree, held

    def test_a_converged_copy_of_an_unlisted_name_is_adopting_in_the_public_mode(self):
        # the table's bytes under a name it does not list yet are being adopted (the next table lists
        # them): pending, not a finding
        self.write_pin(self.unlisted_name(), None)
        tree, held = self.converged_tree()
        code, out = self.run_script(str(tree))
        self.assertEqual(code, 0, out)
        self.assertIn("adopting", out)
        self.assertNotIn("private copy", out)

    def test_a_pin_without_a_visibility_line_is_public(self):
        # no line is the public mode: a name the table never lists then holds a shared path with other
        # bytes than its row's, which fails, and the private mode's wording does not appear (the private
        # mode is never the default). The other bytes are made here, on a converged tree, so the result
        # does not depend on whether the host repository's copies are at their targets.
        self.write_pin(self.unlisted_name(), None)
        tree, held = self.converged_tree()
        f = tree / held[0]
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(bytes(data))
        code, out = self.run_script(str(tree))
        self.assertEqual(code, 1, out)
        self.assertIn(held[0], out)
        self.assertNotIn("private copy", out)

    def test_an_unknown_visibility_fails(self):
        self.write_pin(self.unlisted_name(), "secret")
        self.assertEqual(self.run_script(str(self.tree))[0], 2)

    def test_a_visibility_line_without_a_value_fails(self):
        # a bare `visibility` line must not read as public
        self.write_pin(self.unlisted_name(), "")
        self.assertEqual(self.run_script(str(self.private_tree()))[0], 2)

    def test_the_private_mode_comparing_nothing_fails(self):
        self.write_pin(self.unlisted_name(), "private")
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        (empty / "README.md").write_text("nothing shared\n", encoding="utf-8")
        self.assertEqual(self.run_script(str(empty))[0], 2)



class TableAndFetch(unittest.TestCase):
    """the table reader and the one fetch, without the network"""

    SHARED = b'[[file]]\npath = "scripts/a.py"\nsha256 = "' + b"a" * 64 + b'"\nrepos = ["ALICE-X"]\n'
    OWN_ROW = b'[[file]]\npath = ".github/workflows/ci.yml"\nown = "each repository\'s own"\n'

    def setUp(self):
        self.saved = dict(UNIFORM)

    def tearDown(self):
        UNIFORM.clear()
        UNIFORM.update(self.saved)

    def test_own_rows_are_their_own_kind(self):
        rows, own = parse_table(self.SHARED + b"\n" + self.OWN_ROW)
        self.assertEqual(rows, [("scripts/a.py", ["ALICE-X"])])
        self.assertEqual(own, [".github/workflows/ci.yml"])

    def test_a_table_without_own_rows_reads(self):
        rows, own = parse_table(self.SHARED)
        self.assertEqual((len(rows), own), (1, []))

    def test_a_row_of_neither_kind_fails(self):
        with self.assertRaisesRegex(RuntimeError, "neither shared"):
            parse_table(self.SHARED + b'[[file]]\npath = "scripts/b.py"\nsha256 = "' + b"b" * 64 + b'"\n')

    def test_a_row_of_both_kinds_fails(self):
        with self.assertRaisesRegex(RuntimeError, "neither shared"):
            parse_table(self.SHARED + b'[[file]]\npath = "scripts/b.py"\nrepos = ["ALICE-X"]\nown = "x"\n')

    def test_a_table_of_own_rows_only_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no shared row"):
            parse_table(self.OWN_ROW)

    def test_an_empty_table_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no shared row"):
            parse_table(b"# nothing\n")

    def fake_run(self, fails: int):
        """a subprocess.run whose `git fetch` fails `fails` times, then succeeds"""
        calls = {"fetch": 0}

        def run(cmd, **kw):
            if "fetch" in cmd:
                calls["fetch"] += 1
                code = 1 if calls["fetch"] <= fails else 0
                return subprocess.CompletedProcess(cmd, code, "", "fatal: unable to access: 503" if code else "")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return run, calls

    def test_a_fetch_that_keeps_failing_is_one_named_error(self):
        run, calls = self.fake_run(fails=99)
        waits = []
        with self.assertRaisesRegex(RuntimeError, r"ALICE-Law r0: fetch failed after 4 attempts: .*503"):
            fetch_law(Path("unused"), "r0", run=run, sleep=waits.append)
        self.assertEqual((calls["fetch"], waits), (4, list(FETCH_BACKOFF)))

    def test_a_fetch_that_recovers_is_retried_with_backoff(self):
        run, calls = self.fake_run(fails=2)
        waits = []
        fetch_law(Path("unused"), "r0", run=run, sleep=waits.append)
        self.assertEqual((calls["fetch"], waits), (3, list(FETCH_BACKOFF[:2])))

    def test_the_url_rewrite_is_added_after_existing_git_config_entries(self):
        old = os.environ.get("GIT_CONFIG_COUNT")
        os.environ["GIT_CONFIG_COUNT"] = "2"
        try:
            env = law_env(Path("law.git"))
        finally:
            if old is None:
                del os.environ["GIT_CONFIG_COUNT"]
            else:
                os.environ["GIT_CONFIG_COUNT"] = old
        self.assertEqual(env["GIT_CONFIG_COUNT"], "3")
        self.assertEqual(env["GIT_CONFIG_VALUE_2"], LAW_URL)
        self.assertTrue(env["GIT_CONFIG_KEY_2"].startswith("url.file:") and env["GIT_CONFIG_KEY_2"].endswith(".insteadOf"))

if __name__ == "__main__":
    unittest.main()
