#!/usr/bin/env python3
"""Tests for scripts/docs_lint.py.

Each case builds a small tree in a temporary directory, breaks exactly one
thing, and asserts that the linter reports it. The first case runs the linter
against this repository.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docs_lint as dl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARGO = '[package]\nname = "demo"\nversion = "1.5.0"\n\n[features]\n'

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- `world::step_n`: run several steps

### Changed

- **Behavior change:** `RigidBody::new` friction default 0.3 -> 0.5

### Fixed

- `fluid::kernel` normalisation

## [1.4.0] - 2026-09-17

### Added — old style heading kept as history

## [1.3.0] - 2026-09-16
"""

README = "# demo\n\nA physics engine. The memory footprint is small.\n"


def tree(overrides: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> str:
    files = {
        "Cargo.toml": CARGO,
        "CHANGELOG.md": CHANGELOG,
        "README.md": README,
        "README_JP.md": README,
        "docs/MODULES.md": "# Modules\n",
    }
    files.update(overrides or {})
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        if rel in drop:
            continue
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    return d


def errors(overrides=None, drop=()):
    return dl.check(tree(overrides, drop))[0]


class RealRepo(unittest.TestCase):
    def test_this_repository_passes(self):
        errs, counts = dl.check(ROOT)
        self.assertEqual(errs, [])
        self.assertGreater(counts["versions"], 5)


class Vocabulary(unittest.TestCase):
    def test_clean_tree_passes(self):
        self.assertEqual(errors(), [])

    def test_agent_process_word(self):
        e = errors({"README.md": README + "Measured by worker 3.\n"})
        self.assertTrue(any("agent process `worker`" in x for x in e), e)

    def test_every_hit_on_a_line_is_reported(self):
        # one hit must not hide a second hit of the same kind on the same line
        e = errors({"README.md": README + "worker 1 and subagent 2 measured it\n"})
        self.assertEqual(len([x for x in e if "agent process" in x]), 2, e)

    def test_internal_tracker_word(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("run several steps", "run several steps (Backlog item)")})
        self.assertTrue(any("internal tracker `Backlog`" in x for x in e), e)

    def test_session_name(self):
        e = errors({"README_JP.md": README + "ys-3a が検出\n"})
        self.assertTrue(any("session name" in x for x in e), e)

    def test_private_names_are_matched_by_hash_including_phrases(self):
        # stand-in names: the real ones are only stored as hashes in docs_lint.py
        import hashlib
        fake = {hashlib.sha256(n.encode()).hexdigest() for n in ("zorblax", "quux-lab", "acme rocket works")}
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = fake
            e = errors({"docs/MODULES.md": "consumers: Zorblax, quux-lab and the Acme Rocket Works team\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        hits = sorted(x.split("(`")[1].rstrip("`)") for x in e if "private or internal name" in x)
        self.assertEqual(hits, ["Acme Rocket Works", "Zorblax", "quux-lab"], e)

    def test_private_names_are_found_in_every_file_and_path_of_the_tree(self):
        # comments, generated ledgers, workflows and scripts are published too
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = {hashlib.sha256(b"zorblax").hexdigest()}
            e = errors({"src/a.rs": "// see the Zorblax notes\nfn a() {}\n",
                        ".github/workflows/ci.yml": "      # Zorblax rule\n",
                        "docs/oracle-status.md": "For details: [ZORBLAX.md](../ZORBLAX.md)\n",
                        "src/zorblax_glue.rs": "fn b() {}\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        where = sorted({x.split(":")[0] for x in e if "private or internal name" in x})
        self.assertEqual(where, [".github/workflows/ci.yml", "docs/oracle-status.md", "src/a.rs",
                                 "src/zorblax_glue.rs"], e)
        self.assertTrue(any("in a file path" in x for x in e), e)

    def test_binary_files_are_not_read_as_text(self):
        import hashlib
        d = tree()
        with open(os.path.join(d, "blob.bin"), "wb") as f:
            f.write(b"zorblax\0\x01\x02")
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = {hashlib.sha256(b"zorblax").hexdigest()}
            e, counts = dl.check(d)
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertEqual(e, [])
        self.assertGreater(counts["tree files"], 0)

    def test_a_japanese_name_glued_to_a_particle_is_still_found(self):
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256("ゾルバクス".encode()).hexdigest(),
                                              hashlib.sha256(b"zorb").hexdigest()}
            for text in ("ゾルバクスの世界", "第二章ゾルバクス編", "zorbの設定"):
                self.assertTrue(dl.private_names(text), text)
            # a name only as part of a longer katakana word is not that name
            self.assertEqual(dl.private_names("ゾルバクスター"), [])
        finally:
            dl.PRIVATE_NAME_HASHES = saved

    def test_the_hash_list_is_not_empty_and_holds_no_plain_names(self):
        self.assertGreaterEqual(len(dl.PRIVATE_NAME_HASHES), 10)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in dl.PRIVATE_NAME_HASHES))

    def test_private_address_and_device(self):
        e = errors({"README.md": README + "runs on Jetson at 100.79.87.42\n"})
        self.assertTrue(any("device `Jetson`" in x for x in e), e)
        self.assertTrue(any("private address `100.79.87.42`" in x for x in e), e)

    def test_ordinary_english_is_not_flagged(self):
        # "memory footprint" and a netcode "session" are ordinary words
        self.assertEqual(errors({"README.md": README + "A rollback session keeps memory low.\n"}), [])

    def test_identifiers_inside_code_blocks_are_ignored(self):
        self.assertEqual(errors({"README.md": README + "```rust\nlet worker = 1;\n```\n"}), [])

    def test_reported_line_number_matches_the_file(self):
        text = README + "```\nx\ny\n```\nBacklog here\n"
        e = errors({"README.md": text})
        line = text.split("\n").index("Backlog here") + 1
        self.assertTrue(any(x.startswith(f"README.md:{line}:") for x in e), e)

    def test_missing_document(self):
        e = errors(drop=("README_JP.md",))
        self.assertTrue(any("README_JP.md: missing" in x for x in e), e)


class DevelopmentVocabulary(unittest.TestCase):
    """How the work was organised: in the documents and in every other tracked file."""

    def test_an_english_word_next_to_japanese_is_found(self):
        # `\b` sees no boundary between "worker" and "が"; this is the case that slipped through
        e = errors({"README_JP.md": README + "3 件とも thin_wall workerが見つけた\n"})
        self.assertTrue(any("agent process `worker`" in x for x in e), e)
        e = errors({"README_JP.md": README + "詳細はBacklogに記録\n"})
        self.assertTrue(any("internal tracker `Backlog`" in x for x in e), e)

    def test_worktree_and_multi_agent_words(self):
        for text, hit in (("each change was made in a worktree", "worktree"),
                          ("a multi-agent setup", "multi-agent"),
                          ("マルチエージェントで並列化", "マルチエージェント"),
                          ("エージェントが測定", "エージェント")):
            e = errors({"README.md": README + text + "\n"})
            self.assertTrue(any(f"agent process `{hit}`" in x for x in e), (text, e))

    def test_a_git_worktree_command_is_not_a_description(self):
        wf = "jobs:\n  x:\n    steps:\n      - run: git worktree add /tmp/base origin/main\n"
        self.assertEqual(errors({".github/workflows/x.yml": wf}), [])

    def test_physics_words_that_share_letters_are_not_flagged(self):
        # a numerical integrator and a CCD agent radius are physics, not process
        src = "// the integrator's order\n// zero agent radius still converges\nlet reference_temp_k = 1;\nproject_pressure_multigrid();\n"
        self.assertEqual(errors({"src/a.rs": src}), [])

    def test_vocabulary_is_checked_in_every_tracked_file(self):
        for rel, text, label in (("src/a.rs", "// see the Backlog\n", "internal tracker"),
                                 ("docs/ROADMAP.md", "worker 5 本で実施\n", "agent process"),
                                 ("tests/b.rs", "// (ys-08 判断)\n", "session name"),
                                 ("examples/c.rs", "//! user 裁定: 追加しない\n", "instruction source"),
                                 ("src/d.rs", "/// oracle: `sakamoro-ff`'s derivation\n", "session name")):
            e = errors({rel: text})
            self.assertTrue(any(x.startswith(f"{rel}:1: {label}") for x in e), (rel, e))

    def test_private_note_names(self):
        for text in ("// see [[feedback_degenerate_case_as_silent_wrong_answer]]",
                     "//! `feedback_mms_degree_blind_on_structured_lattice` measured",
                     "# canonical CI template: [[reference_alice_ci_canonical_template]]",
                     "/// (`project_alice_physics_world_auditor_engine_gaps`)",
                     "// `memory/feedback_x.md`"):
            e = errors({"src/a.rs": text + "\n"})
            self.assertTrue(any("internal note" in x or "internal tracker" in x for x in e), (text, e))

    def test_internal_rule_and_template_references(self):
        for text, label in (("# 罠 #22 に従う", "internal note"),
                            ("# canonical CI template", "internal note")):
            e = errors({"src/a.rs": text + "\n"})
            self.assertTrue(any(label in x for x in e), (text, e))

    def test_internal_rule_names_are_matched_by_hash(self):
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256(b"zorb-quux-rules").hexdigest()}
            e = errors({"src/a.rs": "/// (`zorb-quux-rules` §11.2)\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertTrue(any("private or internal name" in x for x in e), e)

    def test_technical_terms_that_share_letters_are_not_flagged(self):
        # `skills/` is a shipped directory here; Unity runs jobs on worker threads;
        # `max_workers` is a Python API argument
        src = ("# - skills: skill files for coding tools\n"
               "// first touched from its worker thread\n"
               "with ThreadPoolExecutor(max_workers=16) as pool:\n")
        self.assertEqual(errors({"src/a.rs": src}), [])
        e = errors({"src/a.rs": "// measured by the worker\n"})
        self.assertTrue(any("agent process `worker`" in x for x in e), e)

    def test_an_assistant_name_is_allowed_only_as_a_product_mention(self):
        # stand-in name: the real ones are only stored as hashes in docs_lint.py
        import hashlib
        fake = {hashlib.sha256(b"zorblax").hexdigest()}
        saved = (dl.PRIVATE_NAME_HASHES, dl.ASSISTANT_NAME_HASHES)
        try:
            dl.PRIVATE_NAME_HASHES = saved[0] | fake
            dl.ASSISTANT_NAME_HASHES = fake
            ok = errors({"server/llm.py": "MODEL = 'zorblax-small'\n",
                         "docs/TEXT_TO_3D.md": "| Zorblax API |\n",
                         "docs/USAGE.md": "User prompt -> LLM (Zorblax/Gemini) -> tree\n",
                         "README.md": README + "A skill for Zorblax Code / Codex.\n"})
            bad = errors({".gitignore": "# Zorblax Code\nZORBLAX.md\n",
                          ".github/workflows/x.yml": "# see ZORBLAX.md\n",
                          "README.md": README + "Written with Zorblax.\n"})
        finally:
            dl.PRIVATE_NAME_HASHES, dl.ASSISTANT_NAME_HASHES = saved
        self.assertEqual(ok, [])
        where = sorted({x.split(":")[0] for x in bad if "private or internal name" in x})
        self.assertEqual(where, [".github/workflows/x.yml", ".gitignore", "README.md"], bad)

    def test_a_private_name_on_a_product_line_is_still_reported(self):
        # the product exception covers the assistant names only
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256(b"quuxcorp").hexdigest()}
            e = errors({"server/llm.py": "# Gemini fallback for Quuxcorp\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertTrue(any("Quuxcorp" in x for x in e), e)

    def test_the_files_that_define_the_vocabulary_are_exempt(self):
        e = errors({"scripts/test_land.py": "msg = 'found by the worker'\n"})
        self.assertEqual(e, [])

    def test_tree_vocabulary_compared_something(self):
        _, counts = dl.check(tree({"src/a.rs": "fn a() {}\n"}))
        self.assertGreater(counts["tree vocabulary"], 0)


class Changelog(unittest.TestCase):
    def test_duplicate_version(self):
        e = errors({"CHANGELOG.md": CHANGELOG + "\n## [1.3.0] - 2026-09-16\n"})
        self.assertTrue(any("appear twice" in x for x in e), e)

    def test_unreleased_must_be_first(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("## [Unreleased]", "## [1.5.0] - 2026-10-04\n\n## [Unreleased]")})
        self.assertTrue(any("not [Unreleased]" in x for x in e), e)

    def test_versions_descending(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("## [1.3.0]", "## [1.4.1]")})
        self.assertTrue(any("is not newer" in x for x in e), e)

    def test_prerelease_sorts_below_its_release(self):
        cl = CHANGELOG + "\n## [1.0.0] - x\n\n## [1.0.0-preview.2] - x\n\n## [1.0.0-preview.1] - x\n"
        self.assertEqual(errors({"CHANGELOG.md": cl}), [])

    def test_cargo_version_without_section_or_unreleased(self):
        cl = CHANGELOG.replace("## [Unreleased]\n", "")
        cl = cl.replace("### Added\n\n- `world::step_n`", "## [1.4.5] - x\n\n### Added\n\n- `world::step_n`", 1)
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("no section and there is no [Unreleased]" in x for x in e), e)

    def test_cargo_version_older_than_newest_section(self):
        e = errors({"Cargo.toml": CARGO.replace('1.5.0', '1.2.0')})
        self.assertTrue(any("older than the newest section" in x for x in e), e)

    def test_category_twice_in_unreleased(self):
        cl = CHANGELOG.replace("### Fixed\n", "### Added\n\n- extra\n\n### Fixed\n")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("`### Added` appears 2 times" in x for x in e), e)

    def test_non_keep_a_changelog_category(self):
        cl = CHANGELOG.replace("### Fixed\n", "### Improvements\n\n- x\n\n### Fixed\n")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("`### Improvements` is not a Keep a Changelog category" in x for x in e), e)

    def test_emoji_marker_in_unreleased(self):
        cl = CHANGELOG.replace("**Behavior change:**", "⚠️")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("emoji status marker" in x for x in e), e)

    def test_released_sections_keep_their_old_headings(self):
        # "### Added — old style heading" in [1.4.0] is history, not checked
        self.assertEqual(errors(), [])

    def test_unreleased_entries_without_categories_compare_nothing(self):
        cl = CHANGELOG.split("### Added")[0] + "- an entry with no category heading\n\n## [1.4.0] - 2026-09-17\n"
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("compared nothing" in x and "categories" in x for x in e), e)

    def test_an_empty_unreleased_right_after_a_release_passes(self):
        cl = "# Changelog\n\n## [Unreleased]\n\n## [1.5.0] - 2026-10-07\n\n### Added\n\n- x\n\n## [1.4.0] - 2026-09-17\n"
        self.assertEqual(errors({"CHANGELOG.md": cl}), [])


if __name__ == "__main__":
    unittest.main()
