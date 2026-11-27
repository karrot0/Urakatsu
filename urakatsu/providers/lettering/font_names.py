"""A font's weight, read from its file name.

The font reader names one of 6,150 font files, and most of their names say how heavy they are:
``W3`` and ``W9`` (the Japanese foundries' scale), ``Light`` and ``Heavy``, a trailing ``-DB`` or
``-EB``, 细 and 粗. Weighing each named font by its probability gives the weight of the lettering
itself, which Koharu does not read (it calls a line bold from its stroke width alone). Names that
say nothing (about three in ten, mostly display faces cut in one weight) take no part.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Optional

# Longest names first, so "extra light" is not read as "light".
_WORDS: tuple[tuple[str, int], ...] = (
    (r"extra[\s-]?light|ultra[\s-]?light|exl\b", 200),
    (r"demi[\s-]?light|semi[\s-]?light", 350),
    (r"thin|hairline", 100),
    (r"light", 300),
    (r"demi[\s-]?bold|semi[\s-]?bold", 600),
    (r"extra[\s-]?bold|ultra[\s-]?bold|xbold", 800),
    (r"heavy|black|ultra", 900),
    (r"bold", 700),
    (r"medium", 500),
    (r"regular|normal|book", 400),
)
# A trailing abbreviation, after the family and its character set (``FOT-RodinPro-DB``).
_SUFFIXES: dict[str, int] = {
    "EL": 200, "UL": 200, "L": 300, "Lt": 300, "R": 400, "Reg": 400, "N": 400, "M": 500, "Md": 500,
    "D": 600, "DB": 600, "SB": 600, "B": 700, "Bd": 700, "E": 800, "EB": 800,
    "EH": 900, "H": 900, "U": 900, "UB": 900, "UH": 900, "BK": 900,
}  # fmt: skip
_CHARACTER_SET = re.compile(r"[-_ ]?(?:C\d|Pr\d?N?|Pro\d?N?|Std\d?N?|StdN|ProN)$")
_SUFFIX = re.compile(r"[-_ ]([A-Z][A-Za-z]{0,2})$")
_JOINED_SUFFIX = re.compile(r"(?:Std|Pro|Pr5|Pr6|Pr6N|ProN|Pr5N|StdN)-?([A-Z][A-Za-z]{0,2})$")
_W_NUMBER = re.compile(r"w(\d{1,2})(?!\d)")
# Chinese names: heaviest first, as 特粗 holds 粗.
_CHINESE: tuple[tuple[str, int], ...] = (
    ("特粗|超粗|特黑|超黑|极粗|極粗|特大|超大|大黑|大标|大標", 900),
    ("特细|特細|极细|極細", 200),
    ("粗", 700),
    ("中", 500),
    ("准|準", 400),
    ("细|細|纤|纖|幼", 300),
)


def weight_of(path: str) -> Optional[int]:
    """The weight (100 to 900) a font file's name states, or None when it states none."""
    # A collection lists its faces ("A-W5 - AP-W5 - AG-W5.ttc"): they share a weight.
    stem = PurePosixPath(path).stem.split(" - ")[0].strip()
    lower = stem.lower()
    numbered = _W_NUMBER.search(lower)
    if numbered is not None:
        # W1 to W9 are the hundreds; DynaFont runs on to W14, all of them black.
        return min(int(numbered.group(1)), 9) * 100 or None
    for pattern, weight in _WORDS:
        if re.search(pattern, lower):
            return weight
    for pattern in (_SUFFIX, _JOINED_SUFFIX):
        match = pattern.search(_CHARACTER_SET.sub("", stem) if pattern is _SUFFIX else stem)
        if match is not None and match.group(1) in _SUFFIXES:
            return _SUFFIXES[match.group(1)]
    for pattern, weight in _CHINESE:
        if re.search(pattern, stem):
            return weight
    return None
