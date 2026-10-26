"""A chapter as a Komga book: a CBZ of its rendered pages with a ``ComicInfo.xml``.

Komga reads the book's title, number, writers, genres, language and reading direction from
``ComicInfo.xml`` (the Anansi Project schema, v2.0) when it scans the file, so every book
carries its own metadata and needs no API call.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Optional
from xml.etree import ElementTree


@dataclass(frozen=True)
class BookInfo:
    series: str
    title: str
    number: Optional[float] = None
    summary: str = ""
    writers: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    web: Optional[str] = None  # the series page on the site it came from
    language: Optional[str] = None  # the pages' language (the translation's), a BCP 47 tag
    right_to_left: bool = False


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def comic_info(book: BookInfo, pages: int) -> bytes:
    """``ComicInfo.xml`` for the book, its elements in the schema's order."""
    root = ElementTree.Element("ComicInfo", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", "xmlns:xsd": "http://www.w3.org/2001/XMLSchema"})

    def add(name: str, value: Optional[str]) -> None:
        if value:
            ElementTree.SubElement(root, name).text = value

    add("Title", book.title)
    add("Series", book.series)
    add("Number", _number(book.number) if book.number is not None else None)
    add("Summary", book.summary)
    add("Writer", ", ".join(book.writers))
    add("Genre", ", ".join(book.genres))
    add("Web", book.web)
    add("PageCount", str(pages))
    add("LanguageISO", book.language.split("-")[0].lower() if book.language else None)
    add("Manga", "YesAndRightToLeft" if book.right_to_left else "Yes")
    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def pack_cbz(pages: list[tuple[str, bytes]], book: BookInfo) -> bytes:
    """The pages, in order, and ``ComicInfo.xml``; stored uncompressed, as pages already are."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in pages:
            archive.writestr(name, data)
        archive.writestr("ComicInfo.xml", comic_info(book, len(pages)))
    return buffer.getvalue()


__all__ = ["BookInfo", "comic_info", "pack_cbz"]
