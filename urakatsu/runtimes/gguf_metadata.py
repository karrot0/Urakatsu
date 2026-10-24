"""Reads GGUF header metadata (key/value pairs) without loading tensors.

Used for the values llama-server does not expose but the chat-template rendering
needs, such as the EOS token's text. Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


def _read(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(handle: BinaryIO) -> str:
    return handle.read(_read(handle, "<Q")).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, kind: int) -> Any:
    if kind in _SCALARS:
        return _read(handle, _SCALARS[kind])
    if kind == _STRING:
        return _string(handle)
    if kind == _ARRAY:
        item_kind, count = _read(handle, "<I"), _read(handle, "<Q")
        return [_value(handle, item_kind) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: Path, keys: set[str] | None = None) -> dict[str, Any]:
    """Header key/value pairs; with ``keys``, only those (reading stops once all are found)."""
    result: dict[str, Any] = {}
    with Path(path).open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(handle, "<I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        _read(handle, "<Q")  # tensor count
        for _ in range(_read(handle, "<Q")):
            key = _string(handle)
            value = _value(handle, _read(handle, "<I"))
            if keys is None or key in keys:
                result[key] = value
                if keys is not None and keys <= result.keys():
                    break
    return result


def special_token_text(path: Path, name: str) -> str:
    """The text of a special token (``bos``, ``eos``) from the GGUF vocabulary, or ""."""
    key = f"tokenizer.ggml.{name}_token_id"
    meta = read_metadata(path, {key, "tokenizer.ggml.tokens"})
    token_id, tokens = meta.get(key), meta.get("tokenizer.ggml.tokens") or []
    return tokens[token_id] if isinstance(token_id, int) and 0 <= token_id < len(tokens) else ""
