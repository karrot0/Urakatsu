"""Lettering styles: the kinds of face a line is lettered in, and the font each is set in.

A manga letters each kind of line in its own kind of face: dialogue in antique (mincho-like kana
with gothic kanji), narration in mincho, shouts in a heavy gothic, thoughts in a rounded gothic,
and so on (https://manga.jp.net/knowhow/150924font/). A style is one of those kinds, named as
Koharu's kernels are (``crates/koharu-scene/src/text.rs``), which follow the classifications of
Morisawa, Founder and DynaComware. The lettering stage reads which one a source line is in
(``pipeline/lettering.py``); the renderer sets the translation in that style's font for the
translation's script (``StyleTable.face``).

The default table is Koharu 0.100's (``crates/koharu-renderer/src/config.rs``): for Latin, the
fonts scanlation groups' guides set each kind of line in; for Chinese and Japanese, a bundled face
of the same kind. Every family it names is in the bundled catalog (``renderer/fonts.py``).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# The styles, in the order the font model's kernel table and the settings list them.
STYLES: tuple[str, ...] = (
    "antique",
    "mincho",
    "fangsong",
    "gothic",
    "rounded",
    "kaisho",
    "gyokai",
    "gyosho",
    "sosho",
    "reisho",
    "tensho",
    "kointai",
    "weibei",
    "textbook",
    "brush",
    "handwriting",
    "pop",
    "design",
    "decorative",
)

# The style a text without one is set in: dialogue's.
DEFAULT_STYLE = "antique"

# ``Kernel::group``: the styles of one look, read and set together. The mincho kin sit under
# mincho, the calligraphic hands under brush, and pop and decorative faces under design.
_GROUPS: dict[str, str] = {
    "fangsong": "mincho",
    "kaisho": "mincho",
    "textbook": "mincho",
    "gyosho": "brush",
    "gyokai": "brush",
    "sosho": "brush",
    "reisho": "brush",
    "tensho": "brush",
    "kointai": "brush",
    "weibei": "brush",
    "pop": "design",
    "decorative": "design",
}

# What each style is used for, for whoever picks one (the type bar, Settings, the agent).
STYLE_USES: dict[str, str] = {
    "antique": "Dialogue",
    "mincho": "Narration",
    "gothic": "Shouts",
    "rounded": "Thoughts",
    "handwriting": "Handwriting",
    "brush": "Brush",
    "design": "Special voices",
    "pop": "Side text",
    "kointai": "Horror",
    "decorative": "Decorative",
}

# The scripts a style names a face for, as ISO 15924 codes.
SCRIPTS: tuple[str, ...] = ("Latn", "Cyrl", "Grek", "Hans", "Hant", "Jpan")


def group(style: str) -> str:
    """The style whose look ``style`` shares (itself, for most)."""
    return _GROUPS.get(style, style)


class StyleFace(BaseModel):
    """``Face``: the font a style sets for one script. What it leaves out is antique's."""

    family: Optional[str] = None
    weight: Optional[int] = None
    italic: Optional[bool] = None

    def over(self, other: "StyleFace") -> "StyleFace":
        """This face, with what it leaves out taken from ``other``."""
        return StyleFace(
            family=self.family if self.family else other.family,
            weight=self.weight if self.weight is not None else other.weight,
            italic=self.italic if self.italic is not None else other.italic,
        )


StyleTable = dict[str, dict[str, StyleFace]]


def _named(family: str) -> StyleFace:
    return StyleFace(family=family)


def default_styles() -> StyleTable:
    """``faces``: each style's default face by script. A group's first style's face is each of its
    styles' that names none of its own."""
    italic = StyleFace(italic=True)
    emphatic = StyleFace(weight=700, italic=True)
    # Narration in CC Victory Speech, shouts in CreativeBlock BB, thoughts in Anime Ace, italic,
    # handwriting in Ashcan BB, side text in Komika Slick, as Quicksand Scans', Vorbis's and
    # Gemelli's font guides set them; brush and horror in the Blambot fonts tagged so.
    latin = {
        "antique": _named("CCWildWords"),
        "mincho": _named("CCVictorySpeechLower"),
        "gothic": _named("CreativeBlock BB"),
        "rounded": StyleFace(family="Anime Ace 3 BB", italic=True),
        "handwriting": _named("Ashcan BB"),
        "brush": _named("Ninjutsu BB"),
        "kointai": _named("ManEater BB"),
        "design": _named("Voice Activated BB"),
        "pop": _named("Komika Slick"),
        "decorative": _named("ManEater BB"),
    }
    # The comic fonts lack Cyrillic and Greek: the dialogue font's italic for thoughts and
    # captions, and its bold italic for emphasis.
    cyrillic = {
        "antique": _named("CCWildWords"),
        "mincho": italic,
        "rounded": italic,
        "brush": emphatic,
        "design": emphatic,
    }
    simplified = {
        "antique": _named("Adobe 黑体 Std"),
        "mincho": _named("Adobe 宋体 Std"),
        # A gothic of every weight, so a heavy one is a real bold.
        "gothic": _named("思源黑体 CN"),
        "rounded": _named("方正准圆_GBK"),
        "handwriting": _named("国祥手写体"),
        "brush": _named("方正行楷_GBK"),
        "reisho": _named("方正隶变_GBK"),
        "design": _named("华康POP1体W5"),
    }
    traditional = {
        "antique": _named("微軟正黑體"),
        "mincho": _named("Adobe 明體 Std"),
        "rounded": _named("華康儷特圓"),
        "handwriting": _named("華康鋼筆體W2"),
        "brush": _named("華康超特楷體斜"),
        "reisho": _named("華康漆隸體斜W5"),
        "design": _named("華康POP1體W7"),
    }
    # The catalog has no Japanese antique or rounded face: antique takes a gothic.
    japanese = {
        "antique": _named("小塚ゴシック Pr6N"),
        "mincho": _named("小塚明朝 Pr6N"),
        "handwriting": _named("瀬戸フォント"),
        "brush": _named("CRＣ＆Ｇ由紀葉太楷書体"),
        "design": _named("FOT-スランプ Std"),
    }
    columns = {"Latn": latin, "Cyrl": cyrillic, "Grek": cyrillic, "Hans": simplified, "Hant": traditional, "Jpan": japanese}
    table: StyleTable = {}
    for script, column in columns.items():
        for style in STYLES:
            face = column.get(style) or column.get(group(style))
            if face is not None:
                table.setdefault(style, {})[script] = face.model_copy()
    return table


def face(styles: StyleTable, style: Optional[str], script: str) -> StyleFace:
    """The face ``style`` sets for ``script``: its own, with what that leaves out from antique's."""
    antique = styles.get(DEFAULT_STYLE, {}).get(script) or StyleFace()
    own = styles.get(style or DEFAULT_STYLE, {}).get(script)
    return own.over(antique) if own is not None else antique


def fallback_families(styles: StyleTable, script: str) -> list[str]:
    """The families a styled text's glyphs fall back on: antique's for its script, then Latin's."""
    families: list[str] = []
    for column in (script, "Latn"):
        family = (styles.get(DEFAULT_STYLE, {}).get(column) or StyleFace()).family
        if family and family not in families:
            families.append(family)
    return families


# Languages written in a script other than the one their text alone implies.
_CYRILLIC = {"ru", "uk", "be", "bg", "sr", "mk", "kk", "ky", "mn", "tg"}
_TRADITIONAL = {"zh-tw", "zh-hk", "zh-mo", "zh-hant"}


def script_of(text: str, language: Optional[str] = None) -> str:
    """The script, of ``SCRIPTS``, a text is set in: its language's, else what its characters
    imply (kana means Japanese; Han alone, Chinese), else Latin."""
    tag = (language or "").strip().lower().replace("_", "-")
    primary = tag.split("-")[0]
    if tag:
        if any(tag == value or tag.startswith(value + "-") for value in _TRADITIONAL):
            return "Hant"
        if primary == "zh":
            return "Hans"
        if primary == "ja":
            return "Jpan"
        if primary in _CYRILLIC:
            return "Cyrl"
        if primary == "el":
            return "Grek"
    han = False
    for character in text:
        code = ord(character)
        if 0x3040 <= code <= 0x30FF:
            return "Jpan"
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            han = True
        elif 0x0400 <= code <= 0x04FF:
            return "Cyrl"
        elif 0x0370 <= code <= 0x03FF:
            return "Grek"
    return "Hans" if han else "Latn"
