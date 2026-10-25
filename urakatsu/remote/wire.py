"""The encoding the remote runtime's client and worker share.

Everything that crosses the wire is lossless, so a remote run sees the same input as a local one:

- Images are PNG, which keeps Pillow's mode (RGB, RGBA, L). Compression level 1: pages are
  large and the network, not the size, is the cost worth paying for.
- Arrays (inpainting masks, detection instance masks) are ``.npz``: any dtype and shape,
  compressed, and read back with ``allow_pickle=False`` so a reply cannot run code.
- Detections are JSON with each instance mask as a base64 ``.npz``.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask

PNG = "image/png"
NPZ = "application/x-npz"


def encode_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def encode_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, a=np.ascontiguousarray(array))
    return buffer.getvalue()


def decode_array(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return archive["a"]


def encode_detections(result: LayoutDetections) -> dict[str, Any]:
    return {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "detections": [
            {
                "label_id": d.label_id,
                "label": d.label,
                "score": d.score,
                "bbox": list(d.bbox),
                "area": d.area,
                "mask": {
                    "x": d.mask.x,
                    "y": d.mask.y,
                    "width": d.mask.width,
                    "height": d.mask.height,
                    "pixels": base64.b64encode(encode_array(d.mask.pixels)).decode("ascii"),
                },
            }
            for d in result.detections
        ],
    }


def decode_detections(data: dict[str, Any]) -> LayoutDetections:
    detections = []
    for d in data["detections"]:
        mask = d["mask"]
        detections.append(
            LayoutDetection(
                label_id=int(d["label_id"]),
                label=str(d["label"]),
                score=float(d["score"]),
                bbox=tuple(float(v) for v in d["bbox"]),  # type: ignore[arg-type]
                area=int(d["area"]),
                mask=LayoutMask(
                    x=int(mask["x"]),
                    y=int(mask["y"]),
                    width=int(mask["width"]),
                    height=int(mask["height"]),
                    pixels=decode_array(base64.b64decode(mask["pixels"])),
                ),
            )
        )
    return LayoutDetections(image_width=int(data["image_width"]), image_height=int(data["image_height"]), detections=detections)
