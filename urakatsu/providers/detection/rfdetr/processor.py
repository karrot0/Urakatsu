"""RF-DETR image preprocessing and instance segmentation postprocessing.

Port of ``processor.rs``. Upstream:
https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/detr.py
https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/models/postprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from urakatsu.core.errors import ProcessingError
from urakatsu.core.results import LayoutDetection, LayoutDetections, LayoutMask, LayoutThresholds

from .config import InferenceConfig

# processor.rs preprocess: ImageNet statistics.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# model.rs: NUM_QUERIES and NUM_CLASSES_WITH_BACKGROUND (the fifth logit is the background slot).
NUM_QUERIES = 300
NUM_CLASSES_WITH_BACKGROUND = 5
PATCH_SIZE_TIMES_WINDOWS = 24  # patch_size 12 * num_windows 2


@dataclass
class NetworkOutput:
    """Last decoder layer outputs of the RF-DETR segmentation network (``model.rs`` ``Output``)."""

    pred_logits: torch.Tensor  # [1, 300, 5]
    pred_boxes: torch.Tensor  # [1, 300, 4] normalised cxcywh
    pred_masks: torch.Tensor  # [1, 300, R/4, R/4] mask logits


class KoharuLayoutRFDetrImageProcessor:
    """``KoharuLayoutRFDetrImageProcessor``."""

    def __init__(self, config: InferenceConfig):
        # processor.rs new(): fail closed on an unusable checkpoint configuration.
        if config.resolution <= 0:
            raise ValueError("RF-DETR resolution must be positive")
        if config.resolution % PATCH_SIZE_TIMES_WINDOWS != 0:
            raise ValueError("RF-DETR resolution must be divisible by patch_size * num_windows")
        if config.num_select <= 0:
            raise ValueError("RF-DETR num_select must be positive")
        self.resolution = config.resolution
        self.num_select = config.num_select
        self.class_names = config.class_names()
        self.recommended_thresholds = config.thresholds()

    def preprocess(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        """``preprocess``: RGB8 -> float in [0,1] -> bilinear resize (no antialias) -> ImageNet normalise."""
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ProcessingError("cannot segment an empty image")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        pixel_values = (
            torch.from_numpy(rgb).to(device).view(1, height, width, 3).permute(0, 3, 1, 2).to(torch.float32) / 255.0
        )
        if width != self.resolution or height != self.resolution:
            # Latest upstream disables antialiasing to match the bilinear Albumentations resize used in training.
            pixel_values = F.interpolate(
                pixel_values, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
            )
        mean = torch.tensor(MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        return (pixel_values - mean) / std

    def postprocess(
        self,
        output: NetworkOutput,
        image_width: int,
        image_height: int,
        thresholds: LayoutThresholds,
    ) -> LayoutDetections:
        """``postprocess``: top-k over (query, class) pairs, per-class thresholds, mask geometry."""
        thresholds.validate()
        side = self.resolution // 4
        if tuple(output.pred_logits.shape) != (1, NUM_QUERIES, NUM_CLASSES_WITH_BACKGROUND):
            raise ProcessingError(f"unexpected RF-DETR logits shape {tuple(output.pred_logits.shape)}")
        if tuple(output.pred_boxes.shape) != (1, NUM_QUERIES, 4):
            raise ProcessingError(f"unexpected RF-DETR box shape {tuple(output.pred_boxes.shape)}")
        if tuple(output.pred_masks.shape) != (1, NUM_QUERIES, side, side):
            raise ProcessingError(f"unexpected RF-DETR mask shape {tuple(output.pred_masks.shape)}")

        # PostProcess ranks every query/class pair, including the checkpoint's fifth background logit
        # slot, before applying the caller threshold.
        scores_all, indexes = output.pred_logits.sigmoid().reshape(1, -1).topk(self.num_select, dim=1, largest=True, sorted=True)
        scores = scores_all[0]
        indexes = indexes[0]
        query_indexes = torch.div(indexes, 5, rounding_mode="floor")
        labels = torch.remainder(indexes, 5)
        # with_background(): the background slot never passes an infinite threshold.
        per_class = torch.tensor(
            [thresholds.text, thresholds.onomatopoeia, thresholds.bubble, thresholds.panel, float("inf")],
            dtype=torch.float32,
            device=scores.device,
        )
        keep = scores > per_class[labels]
        selected = keep.nonzero().view(-1)
        if selected.numel() == 0:
            return LayoutDetections(image_width=image_width, image_height=image_height, detections=[])

        scores = scores[selected]
        labels = labels[selected]
        query_indexes = query_indexes[selected]
        boxes = output.pred_boxes[0].index_select(0, query_indexes)
        center = boxes[:, 0:2]
        half_size = boxes[:, 2:4] / 2.0
        boxes = torch.cat([center - half_size, center + half_size], 1) * torch.tensor(
            [image_width, image_height, image_width, image_height], dtype=torch.float32, device=boxes.device
        )
        floats = torch.cat([scores.unsqueeze(1), boxes], 1).to(torch.float32).cpu().numpy()
        label_values = labels.cpu().numpy()

        # Gathering before interpolation avoids resizing masks for candidates rejected by threshold.
        masks = output.pred_masks[0].index_select(0, query_indexes)
        detections: list[LayoutDetection] = []
        for index, label in enumerate(label_values.tolist()):
            # RF-DETR defines masks by bilinearly projecting each native mask to the source image and
            # thresholding at zero. One mask at a time keeps memory bounded, then only the non-zero
            # page-space extent is kept.
            mask = (
                F.interpolate(
                    masks[index].unsqueeze(0).unsqueeze(0),
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                )
                > 0.0
            ).to(torch.uint8)[0, 0]
            area = int(mask.count_nonzero().item())
            if area == 0:
                x = y = right = bottom = 0
            else:
                columns = (mask.count_nonzero(dim=0) > 0).to(torch.int64)
                rows = (mask.count_nonzero(dim=1) > 0).to(torch.int64)
                x = int(torch.argmax(columns).item())
                y = int(torch.argmax(rows).item())
                right = image_width - int(torch.argmax(columns.flip(0)).item())
                bottom = image_height - int(torch.argmax(rows.flip(0)).item())
            width = max(right - x, 0)
            height = max(bottom - y, 0)
            if width == 0 or height == 0:
                pixels = np.zeros((0, 0), dtype=np.uint8)
            else:
                pixels = (mask[y:bottom, x:right] * 255).cpu().numpy().astype(np.uint8)
            detections.append(
                LayoutDetection(
                    label_id=int(label),
                    label=self.class_names[label] if label < len(self.class_names) else "__background__",
                    score=float(floats[index, 0]),
                    bbox=(float(floats[index, 1]), float(floats[index, 2]), float(floats[index, 3]), float(floats[index, 4])),
                    area=area,
                    mask=LayoutMask(x=x, y=y, width=width, height=height, pixels=pixels),
                )
            )
        return LayoutDetections(image_width=image_width, image_height=image_height, detections=detections)
