#!/usr/bin/env python3
"""tests for scripts/entropy_deps_guard.py (ALICE-Law G10): a configured member's normal or build dependency on
a source of entropy or wall-clock time fails (published or not), a dev-dependency or a member outside the config
does not, a configured member that is not read fails, reading no configured member fails, and a config that is
missing or holds nothing fails

    python3 scripts/test_entropy_deps_guard.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entropy_deps_guard", HERE / "entropy_deps_guard.py")
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g  # (dataclasses look the module up while the class is made)
spec.loader.exec_module(g)

DENY = ("rand", "rand_*", "getrandom", "uuid", "fastrand", "chrono", "time", "oorandom", "nanorand")
ALL = g.Config(("member-a", "member-b"), DENY, "all")
DEFAULT = g.Config(("member-a", "member-b"), DENY, "default")


def meta(*pkgs: dict) -> dict:
    return {"workspace_members": [p["id"] for p in pkgs], "packages": list(pkgs)}


def pkg(name: str, deps: list[dict], publish=None, features=None) -> dict:
    return {"id": f"{name} 0.1.0", "name": name, "publish": publish, "dependencies": deps, "features": features or {}}


def dep(name: str, kind=None, target=None, rename=None, optional=False) -> dict:
    return {"name": name, "kind": kind, "target": target, "rename": rename, "optional": optional}


class Guard(unittest.TestCase):
    def test_a_clean_configured_member_passes(self):
        read, problems = g.denied_deps(meta(pkg("member-a", [dep("glam"), dep("sha2")])), ALL)
        self.assertEqual((read, problems), (["member-a"], []))

    def test_each_denied_crate_fails_as_a_normal_or_build_dependency(self):
        for name in ("rand", "rand_core", "rand_chacha", "getrandom", "uuid", "fastrand", "chrono", "time",
                     "oorandom", "nanorand"):
            for kind in (None, "build"):
                _, problems = g.denied_deps(meta(pkg("member-a", [dep(name, kind=kind)])), ALL)
                self.assertEqual(len(problems), 1, (name, kind))
                self.assertIn(f"({name})", problems[0])

    def test_a_target_specific_or_renamed_dependency_still_fails(self):
        _, problems = g.denied_deps(meta(pkg("member-a", [dep("rand", target="cfg(unix)", rename="rng")])), ALL)
        self.assertEqual(len(problems), 1)
        self.assertIn("rng", problems[0])

    def test_dev_dependencies_and_members_outside_the_config_are_not_checked(self):
        read, problems = g.denied_deps(meta(pkg("member-a", [dep("rand", kind="dev")]),
                                            pkg("tool-c", [dep("rand")])), ALL)
        self.assertEqual((read, problems), (["member-a"], []))

    def test_a_configured_member_with_publish_false_is_still_checked(self):
        # cargo metadata gives publish = false as publish = []; the configured members are checked regardless
        read, problems = g.denied_deps(meta(pkg("member-b", [dep("getrandom")], publish=[])), ALL)
        self.assertEqual(read, ["member-b"])
        self.assertEqual(len(problems), 1, problems)

    def test_names_that_only_contain_a_denied_word_pass(self):
        _, problems = g.denied_deps(meta(pkg("member-a", [dep("timely"), dep("operand"), dep("random-access"),
                                                           dep("randomize"), dep("uuidx")])), ALL)
        self.assertEqual(problems, [])

    def test_transitive_reach_follows_normal_and_build_edges_only(self):
        def node(i, *deps):
            return {"id": i, "deps": [{"pkg": d, "dep_kinds": [{"kind": k}]} for d, k in deps]}
        m = {"workspace_members": ["a"], "packages": [
            {"id": "a", "name": "member-a", "publish": None},
            {"id": "w", "name": "wgpu", "version": "1"}, {"id": "r", "name": "rand", "version": "0.8.5"},
            {"id": "u", "name": "uuid", "version": "1.2"}, {"id": "t", "name": "time", "version": "0.3"}],
            "resolve": {"nodes": [node("a", ("w", None), ("t", "dev")), node("w", ("r", "build")),
                                  node("r", ("u", None)), node("u"), node("t")]}}
        self.assertEqual(g.transitive(m, ALL), ["rand 0.8.5", "uuid 1.2"])

    def test_every_configured_member_must_be_read(self):
        self.assertEqual(g.members_differ(["member-a", "member-b"], ALL), [])
        # mutant: a configured member is not read (renamed, dropped from the workspace, or a typo in the config)
        out = g.members_differ(["member-a"], ALL)
        self.assertEqual(len(out), 1, out)
        self.assertIn("member-b", out[0])


class Features(unittest.TestCase):
    """`features = "all"` counts an optional dependency under any feature, "default" only when enabled by default"""

    def optional_member(self, default: list[str]) -> dict:
        return pkg("member-a", [dep("rand", optional=True), dep("glam")],
                   features={"default": default, "rng": ["dep:rand"], "fast": ["rng"], "ext": ["rand?/std"]})

    def test_an_optional_dependency_counts_under_all(self):
        self.assertEqual(len(g.denied_deps(meta(self.optional_member([])), ALL)[1]), 1)

    def test_an_optional_dependency_off_by_default_passes_under_default(self):
        for default in ([], ["ext"]):  # `rand?/std` does not enable rand
            with self.subTest(default=default):
                self.assertEqual(g.denied_deps(meta(self.optional_member(default)), DEFAULT)[1], [])

    def test_an_optional_dependency_the_default_features_enable_fails_under_default(self):
        for default in (["rng"], ["fast"], ["rand"]):  # direct, through another feature, the implicit feature
            with self.subTest(default=default):
                m = meta(self.optional_member(default))
                if default == ["rand"]:
                    m["packages"][0]["features"] = {"default": ["rand"]}
                self.assertEqual(len(g.denied_deps(m, DEFAULT)[1]), 1, default)

    def test_a_feature_of_the_dependency_enables_it(self):
        m = meta(pkg("member-a", [dep("rand", optional=True)], features={"default": ["rand/std"]}))
        self.assertEqual(len(g.denied_deps(m, DEFAULT)[1]), 1)


class Config(unittest.TestCase):
    def write(self, text: str) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return Path(f.name)

    def test_a_valid_config_is_read(self):
        c = g.read_config(self.write('members = ["a"]\ndeny = ["rand", "rand_*"]\nfeatures = "default"\n'))
        self.assertEqual(c, g.Config(("a",), ("rand", "rand_*"), "default"))
        self.assertEqual(g.read_config(self.write('members = ["a"]\ndeny = ["rand"]\n')).features, "all")

    def test_a_bad_config_is_an_error(self):
        for text in ['members = []\ndeny = ["rand"]\n', 'deny = ["rand"]\n', 'members = ["a"]\ndeny = []\n',
                     'members = ["a"]\n', 'members = ["a"]\ndeny = ["rand"]\nfeatures = "some"\n',
                     'members = ["a"]\ndeny = ["rand"]\nmember = ["b"]\n', 'members = "a"\ndeny = ["rand"]\n',
                     'members = [""]\ndeny = ["rand"]\n', "members = [\n"]:
            with self.subTest(text=text):
                with self.assertRaises(g.ConfigError):
                    g.read_config(self.write(text))

    def test_a_missing_config_is_an_error(self):
        with self.assertRaises(g.ConfigError):
            g.read_config(HERE / "no-such-config.toml")

    def run_guard(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(HERE / "entropy_deps_guard.py"), *args],
                              capture_output=True, text=True)

    def test_the_guard_exits_2_without_a_usable_config(self):
        manifest = str(HERE / "clippy-lib-canary" / "Cargo.toml")
        for args in (["--config", str(HERE / "no-such-config.toml")],
                     ["--config", str(self.write('members = []\ndeny = ["rand"]\n'))]):
            with self.subTest(args=args):
                p = self.run_guard("--manifest-path", manifest, *args)
                self.assertEqual(p.returncode, 2, p.stderr)

    def test_a_configured_member_that_is_not_in_the_workspace_fails(self):
        cfg = self.write('members = ["clippy-lib-canary", "no-such-member"]\ndeny = ["rand"]\n')
        p = self.run_guard("--manifest-path", str(HERE / "clippy-lib-canary" / "Cargo.toml"), "--config", str(cfg))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("no-such-member is configured but was not read", p.stderr)

    def test_no_configured_member_read_is_compared_nothing(self):
        cfg = self.write('members = ["no-such-member"]\ndeny = ["rand"]\n')
        p = self.run_guard("--manifest-path", str(HERE / "clippy-lib-canary" / "Cargo.toml"), "--config", str(cfg))
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("compared nothing", p.stderr)

    def test_this_workspace_passes(self):
        p = self.run_guard("--manifest-path", str(HERE.parent / "Cargo.toml"))
        self.assertEqual(p.returncode, 0, p.stderr)
        cfg = g.read_config(HERE.parent / ".github" / "entropy-deps.toml")
        self.assertIn(", ".join(sorted(cfg.members)), p.stdout)


if __name__ == "__main__":
    unittest.main()
