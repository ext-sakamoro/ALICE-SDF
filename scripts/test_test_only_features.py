#!/usr/bin/env python3
"""Tests for scripts/test_only_features.py: the repository's list passes, and each way a listed feature can reach a
production build is refused, in scratch repositories and on this repository's own manifest."""
import importlib.util
import os
import shutil
import subprocess
import tempfile
import tomllib
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("test_only_features", os.path.join(HERE, "test_only_features.py"))
tof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tof)

BASE = """[package]
name = "crate"
version = "0.0.0"
edition = "2021"

[features]
default = ["a"]
a = []
b = ["a"]
hooks = []

[dev-dependencies]
crate = { path = ".", features = ["hooks"] }
"""


class Scratch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "scripts"))
        subprocess.run(["git", "init", "-q", self.d], check=True)

    def tearDown(self):
        shutil.rmtree(self.d)

    def write(self, rel, text):
        path = os.path.join(self.d, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        subprocess.run(["git", "-C", self.d, "add", rel], check=True)

    def run_check(self, names, manifest=BASE, others=()):
        self.write("Cargo.toml", manifest)
        self.write(tof.LIST, "# comment\n" + "".join(n + "\n" for n in names))
        for rel, text in others:
            self.write(rel, text)
        return tof.check(self.d)

    def test_a_test_only_feature_passes(self):
        problems, _, compared = self.run_check(["hooks"])
        self.assertEqual(problems, [])
        self.assertGreater(compared, 0)

    def test_an_empty_list_compares_nothing_and_exits_2(self):
        self.write("Cargo.toml", BASE)
        self.write(tof.LIST, "# only a comment\n")
        self.assertEqual(tof.main([self.d]), 2)

    def test_a_name_that_is_not_a_feature_is_refused(self):
        problems, _, _ = self.run_check(["nope"])
        self.assertTrue(any("not a feature" in p for p in problems), problems)

    def test_a_default_feature_is_refused(self):
        m = BASE.replace('features = ["hooks"]', 'features = ["hooks", "a"]')
        problems, _, _ = self.run_check(["a"], m)
        self.assertTrue(any("default features" in p for p in problems), problems)

    def test_a_feature_another_feature_enables_is_refused(self):
        m = BASE.replace('b = ["a"]', 'b = ["a", "hooks"]')
        problems, _, _ = self.run_check(["hooks"], m)
        self.assertTrue(any("enabled by crate's feature `b`" in p for p in problems), problems)

    def test_a_feature_the_self_dev_dependency_does_not_enable_is_refused(self):
        problems, _, _ = self.run_check(["hooks", "b"])
        self.assertTrue(any(p.startswith("b: ") and "dev-dependencies" in p for p in problems), problems)

    def test_a_dependent_enabling_it_in_its_dependency_entry_is_refused(self):
        dep = '[package]\nname = "app"\nversion = "0.0.0"\n\n[dependencies]\ncrate = { path = "..", features = ["hooks"] }\n'
        problems, _, _ = self.run_check(["hooks"], others=[("app/Cargo.toml", dep)])
        self.assertTrue(any("app/Cargo.toml enables it in its dependencies" in p for p in problems), problems)

    def test_a_dependent_enabling_it_through_its_default_is_refused(self):
        dep = ('[package]\nname = "app"\nversion = "0.0.0"\n\n[dependencies]\ncrate = { path = ".." }\n\n'
               '[features]\ndefault = ["x"]\nx = ["crate/hooks"]\n')
        problems, _, _ = self.run_check(["hooks"], others=[("app/Cargo.toml", dep)])
        self.assertTrue(any("through its default feature `x`" in p for p in problems), problems)

    def test_a_dependent_opting_in_behind_its_own_feature_is_a_note(self):
        dep = ('[package]\nname = "app"\nversion = "0.0.0"\n\n[dependencies]\ncrate = { path = ".." }\n\n'
               '[features]\nguard = ["crate/hooks"]\n')
        problems, notes, _ = self.run_check(["hooks"], others=[("app/Cargo.toml", dep)])
        self.assertEqual(problems, [])
        self.assertTrue(any("non-default feature `guard`" in n for n in notes), notes)

    def test_a_renamed_dependency_is_followed(self):
        dep = ('[package]\nname = "app"\nversion = "0.0.0"\n\n[dependencies]\nsdf = { package = "crate", path = "..", '
               'features = ["hooks"] }\n')
        problems, _, _ = self.run_check(["hooks"], others=[("app/Cargo.toml", dep)])
        self.assertTrue(any("enables it in its dependencies entry `sdf`" in p for p in problems), problems)


class ThisRepository(unittest.TestCase):
    def test_the_list_passes_and_compares(self):
        problems, _, compared = tof.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(compared, len(tof.listed(ROOT)))

    def test_every_production_feature_added_to_the_list_is_refused(self):
        # the mutant: a production feature named in the list would hide its code from the reach check
        with open(os.path.join(ROOT, "Cargo.toml"), "rb") as f:
            features = tomllib.load(f)["features"]
        listed = tof.listed(ROOT)
        production = [f for f in features if f != "default" and f not in listed]
        self.assertGreater(len(production), 0)
        real = tof.listed
        try:
            for name in production:
                tof.listed = lambda root, name=name: listed + [name]
                problems, _, _ = tof.check(ROOT)
                self.assertTrue(any(p.startswith(f"{name}: ") for p in problems), f"{name} was not refused")
        finally:
            tof.listed = real


if __name__ == "__main__":
    unittest.main()
