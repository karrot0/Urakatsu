"""Chapters from what a user hands over at once: loose images, ZIP/CBZ archives and folders.

The browser sends every file with the path it had (``Series/Ch 2/003.jpg`` when a folder was
picked or dropped, ``vol1.cbz`` for an archive, ``003.jpg`` for a loose image). :func:`gather`
sorts them into :class:`Batch`:

- Each archive and each top-level folder is a *root*. An archive inside a folder or inside another
  archive is a root of its own, so a folder (or a ZIP) of CBZ files gives one chapter per CBZ.
- A root whose images all sit in one directory is one chapter named after the root (the archive's
  name without ``.cbz``, or the folder's name). A root with images in several directories gives a
  chapter per directory, named by its path inside the root (``Vol 1 / Ch 3``); images at the root
  itself take the root's name.
- Loose images belong to no chapter of their own: they join the chapter they were dropped on.

Chapters and pages are in natural order (``2.jpg`` before ``10.jpg``). Files that are neither an
image nor an archive (``ComicInfo.xml``, ``Thumbs.db``), hidden files and macOS ``__MACOSX``
folders are skipped.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

ARCHIVE_SUFFIXES = frozenset({".zip", ".cbz"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"})
# Archives nested deeper than this are skipped (a ZIP of CBZs is two levels).
ARCHIVE_DEPTH = 3
# What one import may unpack, against archives that expand without end.
MAX_ENTRIES = 50_000
MAX_UNPACKED = 8 * 2**30


@dataclass
class Batch:
    """An import sorted into chapters (name, pages as (label, image bytes)) and loose pages."""

    chapters: list[tuple[str, list[tuple[str, bytes]]]] = field(default_factory=list)
    loose: list[tuple[str, bytes]] = field(default_factory=list)

    @property
    def pages(self) -> int:
        return len(self.loose) + sum(len(pages) for _, pages in self.chapters)


@dataclass
class _Root:
    key: str  # sorts the roots: the path the archive or folder had
    name: str
    # directory inside the root -> (file name, bytes)
    folders: dict[tuple[str, ...], list[tuple[str, bytes]]] = field(default_factory=dict)


def natural_key(text: str) -> list:
    """Sorts ``page 2`` before ``page 10``, and a path folder by folder. ``re.split`` with a group
    alternates text and digits, so two keys compare text with text and numbers with numbers. A
    name's extension only breaks ties, so ``Ch 1.cbz`` comes before ``Ch 1 (2).cbz``."""

    def chunks(part: str) -> list:
        return [int(piece) if index % 2 else piece.casefold() for index, piece in enumerate(re.split(r"(\d+)", part))]

    *folders, name = text.split("/")
    stem, suffix = posixpath.splitext(name)
    return [(chunks(folder), "") for folder in folders] + [(chunks(stem), suffix.casefold())]


def is_archive(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in ARCHIVE_SUFFIXES


def is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


def _parts(path: str) -> tuple[str, ...]:
    """A path's parts, whatever its separators, without empty, ``.`` and ``..`` parts."""
    return tuple(part for part in path.replace("\\", "/").split("/") if part not in ("", ".", ".."))


def _skipped(parts: tuple[str, ...]) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in parts)


def _entry_name(info: zipfile.ZipInfo) -> str:
    """An entry's name. ZIP tools that do not mark names as UTF-8 write them in the system's code
    page, which zipfile reads as CP437; Japanese archives are usually Shift-JIS (CP932)."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437", errors="replace")
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


class _Budget:
    def __init__(self) -> None:
        self.entries = 0
        self.unpacked = 0

    def take(self, archive: str, info: zipfile.ZipInfo) -> None:
        self.entries += 1
        self.unpacked += info.file_size
        if self.entries > MAX_ENTRIES:
            raise ValueError(f"{archive} holds more than {MAX_ENTRIES:,} files")
        if self.unpacked > MAX_UNPACKED:
            raise ValueError(f"the archives unpack to more than {MAX_UNPACKED // 2**30} GiB")


def gather(files: list[tuple[str, bytes]]) -> Batch:
    """Sorts uploaded ``(path, bytes)`` into chapters and loose pages (module docstring)."""
    roots: dict[str, _Root] = {}
    loose: list[tuple[str, bytes]] = []
    budget = _Budget()

    def root(key: str, name: str) -> _Root:
        return roots.setdefault(key, _Root(key, name))

    def unpack(key: str, data: bytes, depth: int) -> None:
        name = PurePosixPath(key).stem
        if depth > ARCHIVE_DEPTH:
            return
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise ValueError(f"{PurePosixPath(key).name} is not a ZIP archive (only ZIP and CBZ can be imported)") from None
        with archive:
            # Entries in name order, not archive order, so nested archives become roots in order too.
            entries = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: natural_key(_entry_name(info)))
            for info in entries:
                parts = _parts(_entry_name(info))
                if not parts or _skipped(parts):
                    continue
                if not (is_image(parts[-1]) or is_archive(parts[-1])):
                    continue
                budget.take(PurePosixPath(key).name, info)
                try:
                    content = archive.read(info)
                except RuntimeError as error:  # zipfile's word for an encrypted entry
                    raise ValueError(f"{PurePosixPath(key).name} is password protected") from error
                except (zipfile.BadZipFile, OSError, EOFError) as error:
                    raise ValueError(f"{PurePosixPath(key).name} is damaged: {error}") from error
                if is_archive(parts[-1]):
                    unpack(f"{key}/{'/'.join(parts)}", content, depth + 1)
                else:
                    root(key, name).folders.setdefault(parts[:-1], []).append((parts[-1], content))

    for path, data in files:
        parts = _parts(path)
        if not parts or _skipped(parts):
            continue
        if is_archive(parts[-1]):
            unpack("/".join(parts), data, 1)
        elif is_image(parts[-1]):
            if len(parts) == 1:
                loose.append((parts[0], data))
            else:
                root(parts[0], parts[0]).folders.setdefault(parts[1:-1], []).append((parts[-1], data))

    batch = Batch(loose=sorted(loose, key=lambda page: natural_key(page[0])))
    for found in sorted(roots.values(), key=lambda r: natural_key(r.key)):
        folders = sorted(found.folders.items(), key=lambda item: [natural_key(part) for part in item[0]])
        for folder, pages in folders:
            name = found.name if len(folders) == 1 or not folder else " / ".join(folder)
            batch.chapters.append((name, sorted(pages, key=lambda page: natural_key(page[0]))))
    return batch
