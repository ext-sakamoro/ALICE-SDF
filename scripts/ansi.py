"""Remove terminal escape sequences from tool output before parsing it.

cargo, libtest and the cargo-* tools colour their output when
CARGO_TERM_COLOR=always (the workflows set it) or `--color always` is passed.
Besides SGR colour codes (`ESC [ 32 m`), libtest ends a coloured word with a
character-set selector (`ESC ( B`, from terminfo `sgr0`), which an SGR-only
pattern leaves in place, so `test result: ok.` would read as
`test result: ok\\x1b(B.`. This module removes:

  * CSI sequences: `ESC [` parameters, intermediates, one final byte
    (colours, cursor movement, erase)
  * character-set selectors: `ESC (`, `)`, `*` or `+` and one designator byte
  * OSC sequences: `ESC ]` ... terminated by BEL or `ESC \\` (hyperlinks, titles)

The same pattern is used by ALICE-Physics `scripts/ansi.py`.
"""

from __future__ import annotations

import re

ANSI_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|[()*+][0-9A-Za-z]"  # character-set selector
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r")"
)


def strip(text: str) -> str:
    """`text` without terminal escape sequences."""
    return ANSI_RE.sub("", text)
