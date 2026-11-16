"""A rendered page as a layered Photoshop file (PSD), so it can be edited further elsewhere.

Each layer of the render (``Frame.layers``) becomes a PSD layer, bottom to top: the page art, the
cleanup (inpainting), brush and other raster layers, then every text block as its own layer in a
"Text" group. Layers keep their visibility, opacity and name (a text layer is named after its
text). Text is raster: Photoshop's editable type needs its text engine data, which no format
outside Photoshop writes reliably, so the text keeps its rendered look and position instead.

Koharu exports PSD too (its ``psd`` export); this writer is Urakatsu's own: 8-bit RGB with an
alpha channel per layer, PackBits (RLE) compressed like Photoshop's own files, plus the flattened
image every PSD reader can fall back on. Spec: Adobe Photoshop File Formats Specification
("Layer and Mask Information Section"). PackBits rows come from Pillow's TIFF encoder (one strip
per row), which is the same per-row PackBits PSD uses.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

RLE = 1
# Layer flags: bit 1 hides the layer; bits 3 and 4 mark a group's divider as having no pixels.
HIDDEN = 0x02
NO_PIXELS = 0x18


@dataclass
class PsdLayer:
    name: str
    image: Optional[Image.Image]  # RGBA, cropped to the layer; None for a group marker
    left: int = 0
    top: int = 0
    visible: bool = True
    opacity: float = 1.0
    # "open"/"closed" starts a group (it sits above its layers), "end" closes one (below them).
    section: Optional[str] = None


def _packbits_rows(channel: np.ndarray) -> tuple[list[int], bytes]:
    """PackBits-compressed rows of one 8-bit channel: each row's byte count and the rows joined."""
    height = channel.shape[0]
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(channel), "L").save(buffer, format="TIFF", compression="packbits", tiffinfo={278: 1})
    data = buffer.getvalue()
    tiff = Image.open(io.BytesIO(data))
    offsets, counts = tiff.tag_v2[273], tiff.tag_v2[279]
    if len(offsets) != height:  # pragma: no cover - Pillow always honours one row per strip
        raise RuntimeError("could not compress the layer one row at a time")
    return list(counts), b"".join(data[o : o + n] for o, n in zip(offsets, counts))


def _channel_rle(channel: np.ndarray) -> bytes:
    """A layer channel as PSD channel image data: compression, row byte counts, rows."""
    if channel.size == 0:
        return struct.pack(">H", 0)
    counts, rows = _packbits_rows(channel)
    return struct.pack(">H", RLE) + struct.pack(f">{len(counts)}H", *counts) + rows


def _pascal(name: str) -> bytes:
    """The legacy layer name: a Pascal string in Latin-1 (others replaced), padded to 4 bytes."""
    raw = name.encode("latin-1", "replace")[:255]
    data = bytes([len(raw)]) + raw
    return data + b"\0" * (-len(data) % 4)


def _block(key: bytes, data: bytes) -> bytes:
    data += b"\0" * (len(data) % 2)
    return b"8BIM" + key + struct.pack(">I", len(data)) + data


def _record(layer: PsdLayer) -> tuple[bytes, bytes]:
    """One layer record and its channel image data."""
    if layer.image is None:
        channels = [(i, struct.pack(">H", 0)) for i in (-1, 0, 1, 2)]
        top = left = bottom = right = 0
    else:
        pixels = np.asarray(layer.image.convert("RGBA"))
        height, width = pixels.shape[:2]
        top, left, bottom, right = layer.top, layer.left, layer.top + height, layer.left + width
        channels = [(-1, _channel_rle(pixels[:, :, 3]))] + [(i, _channel_rle(pixels[:, :, i])) for i in (0, 1, 2)]
    record = struct.pack(">4iH", top, left, bottom, right, len(channels))
    for channel_id, data in channels:
        record += struct.pack(">hI", channel_id, len(data))
    flags = 0 if layer.visible else HIDDEN
    if layer.section == "end" or (layer.section and layer.image is None):
        flags |= NO_PIXELS
    blend = b"pass" if layer.section in ("open", "closed") else b"norm"
    opacity = max(0, min(255, round(layer.opacity * 255)))
    record += b"8BIM" + blend + struct.pack(">BBBB", opacity, 0, flags, 0)
    extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(layer.name)
    unicode_name = layer.name.encode("utf-16-be")
    extra += _block(b"luni", struct.pack(">I", len(unicode_name) // 2) + unicode_name)
    if layer.section:
        kind = {"open": 1, "closed": 2, "end": 3}[layer.section]
        extra += _block(b"lsct", struct.pack(">I", kind) + (b"8BIMpass" if kind != 3 else b""))
    record += struct.pack(">I", len(extra)) + extra
    return record, b"".join(data for _, data in channels)


def write_psd(width: int, height: int, layers: list[PsdLayer], flattened: Image.Image) -> bytes:
    """A PSD of ``layers`` (bottom first) over a ``width`` x ``height`` canvas, with ``flattened``
    as the composite image readers show without compositing the layers."""
    records, pixels = [], []
    for layer in layers:
        record, data = _record(layer)
        records.append(record)
        pixels.append(data)
    info = struct.pack(">h", len(layers)) + b"".join(records) + b"".join(pixels)
    info += b"\0" * (len(info) % 2)
    layer_info = struct.pack(">I", len(info)) + info
    section = layer_info + struct.pack(">I", 0)  # no global layer mask

    merged = np.asarray(flattened.convert("RGB"))
    rows = [_packbits_rows(merged[:, :, i]) for i in (0, 1, 2)]
    image_data = struct.pack(">H", RLE)
    image_data += b"".join(struct.pack(f">{len(counts)}H", *counts) for counts, _ in rows)
    image_data += b"".join(data for _, data in rows)

    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, height, width, 8, 3)
    return header + struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", len(section)) + section + image_data


def frame_psd(frame, names: Optional[dict[str, str]] = None) -> bytes:
    """A rendered page (``Frame``) as a PSD: its raster layers, then its text blocks in a "Text"
    group. ``names`` maps a raster layer's entity to the name the editor shows for it."""
    from .frame import ImageKind, _rounded_out
    from .svg import split_index

    names = names or {}
    split = split_index(frame)
    out: list[PsdLayer] = []

    def add(layer, name: str) -> None:
        left, top, right, bottom = _rounded_out(layer.bounds)
        if right <= left or bottom <= top:
            return
        out.append(PsdLayer(name=name, image=layer.image(), left=left, top=top, visible=layer.presentation.visible, opacity=layer.presentation.opacity))

    default = {ImageKind.SOURCE: "Original", ImageKind.CLEANUP: "Cleanup", ImageKind.PAINT: "Brush", ImageKind.EMBEDDED: "Image"}
    for layer in frame.layers[:split]:
        kind = getattr(layer.kind, "kind", None)
        add(layer, names.get(layer.entity) or getattr(layer.kind, "name", None) or default.get(kind, "Layer"))
    text = frame.layers[split:]
    if text:
        out.append(PsdLayer(name="</Layer group>", image=None, section="end"))
        for index, layer in enumerate(text, start=1):
            words = " ".join(getattr(layer.kind, "text", "").split())
            name = words[:60] + ("…" if len(words) > 60 else "") if words else f"Text {index}"
            add(layer, name if layer.is_text() else (names.get(layer.entity) or f"Layer {index}"))
        out.append(PsdLayer(name="Text", image=None, section="open"))
    flattened = Image.new("RGBA", (frame.width, frame.height), (255, 255, 255, 255))
    flattened.alpha_composite(frame.composite())
    return write_psd(frame.width, frame.height, out, flattened)
