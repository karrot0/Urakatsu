"""Comic Text & Bubble Detector: image processing, slicing and detection postprocessing.

Port of Koharu's ``comic_text_bubble_detector/processor.rs`` and the inference flow of ``mod.rs``.
Function names follow the Rust ones. Upstream sources Koharu ported:

- transformers ``RTDetrImageProcessor`` (``preprocess``, ``post_process_object_detection``)
- ogkalu/comic-translate ``modules/detection/utils/slicer.py``, ``content.py``, ``geometry.py``

Deviations from processor.rs, in one place:

- Resizing uses Pillow's bilinear filter. Koharu uses ``fast_image_resize`` bilinear, which is
  close to Pillow's antialiased bilinear but not bit-identical.
- Box geometry is evaluated in float64. Koharu uses f32; only exact threshold ties can differ.
- Scores are carried through every merge. An output box keeps the maximum score of the raw boxes
  folded into it (merge, containment, duplicate, and text-merge duplicate). Koharu drops scores here.
- ``TextBlock`` carries that score. Koharu's FontEngine fields (``direction``, ``font_color``) are
  not produced: they are the upstream font model's job, and Koharu's default path leaves them empty.

Post-processing runs in the network's output dtype, as Koharu's ``f32``-to-``Vec`` conversion
happens after ``boxes *= scale`` and ``sigmoid``/``topk`` on the bfloat16 logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import torch

Box = tuple[float, float, float, float]
IntBox = tuple[int, int, int, int]

# processor.rs: PIL resampling filter codes used by preprocessor_config.json ``resample``.
PIL_RESAMPLING = {
    0: Image.Resampling.NEAREST,
    1: Image.Resampling.LANCZOS,
    2: Image.Resampling.BILINEAR,
    3: Image.Resampling.BICUBIC,  # Koharu: CatmullRom
    4: Image.Resampling.BOX,
    5: Image.Resampling.HAMMING,
}


class ScoredBox(NamedTuple):
    """A box with the score it carries through merges (Koharu carries boxes only)."""

    xyxy: Box
    score: float


@dataclass
class SizeDict:
    height: int = 640
    width: int = 640


@dataclass
class RTDetrImageProcessor:
    """``RTDetrImageProcessor`` (``#[serde(default)]``: keys missing from the JSON keep these values)."""

    do_resize: bool = True
    size: SizeDict = field(default_factory=SizeDict)
    resample: int = 2
    do_rescale: bool = True
    rescale_factor: float = 1.0 / 255.0
    do_normalize: bool = False
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    do_pad: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RTDetrImageProcessor":
        defaults = cls()
        size = data.get("size")
        if isinstance(size, dict):
            size_value = SizeDict(height=int(size.get("height", 640)), width=int(size.get("width", 640)))
        else:
            size_value = SizeDict()
        return cls(
            do_resize=bool(data.get("do_resize", defaults.do_resize)),
            size=size_value,
            resample=int(data.get("resample", defaults.resample)),
            do_rescale=bool(data.get("do_rescale", defaults.do_rescale)),
            rescale_factor=float(data.get("rescale_factor", defaults.rescale_factor)),
            do_normalize=bool(data.get("do_normalize", defaults.do_normalize)),
            image_mean=tuple(float(v) for v in data.get("image_mean", defaults.image_mean)),
            image_std=tuple(float(v) for v in data.get("image_std", defaults.image_std)),
            do_pad=bool(data.get("do_pad", defaults.do_pad)),
        )

    def preprocess(self, image: Image.Image, device: "torch.device") -> "torch.Tensor":
        """``RTDetrImageProcessor::preprocess``: RGB8, resize, rescale, optional normalise, NCHW float32."""
        import torch

        rgb = image.convert("RGB")
        if self.do_resize:
            resample = PIL_RESAMPLING.get(self.resample)
            if resample is None:
                raise ValueError(f"unsupported PIL resampling filter {self.resample}")
            rgb = rgb.resize((self.size.width, self.size.height), resample)
        values = np.asarray(rgb, dtype=np.uint8).astype(np.float32)
        if self.do_rescale:
            values = values * np.float32(self.rescale_factor)
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            values = (values - mean) / std
        # For a single image Transformers pads to the image's own size, so do_pad changes nothing (as in Rust).
        height, width = values.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device)
        return tensor.view(1, height, width, 3).permute(0, 3, 1, 2).contiguous()

    def post_process_object_detection(
        self,
        logits: "torch.Tensor",
        pred_boxes: "torch.Tensor",
        threshold: float,
        target_sizes: Sequence[tuple[int, int]],
        use_focal_loss: bool,
    ) -> list["ObjectDetectionOutput"]:
        """``post_process_object_detection``: (height, width) target sizes, focal-loss top-k or softmax."""
        import torch

        batch_size = int(logits.size(0))
        if len(target_sizes) != batch_size:
            raise ValueError(f"target size count {len(target_sizes)} does not match detector batch size {batch_size}")

        boxes = center_to_corners_format(pred_boxes)
        scale_values: list[float] = []
        for height, width in target_sizes:
            scale_values.extend([float(width), float(height), float(width), float(height)])
        scale = torch.tensor(scale_values, dtype=torch.float32, device=boxes.device).view(batch_size, 1, 4)
        # In-place as in Rust: the product is taken in float32 and stored in the boxes' dtype.
        boxes *= scale

        num_top_queries = int(logits.size(1))
        num_classes = int(logits.size(2))
        if use_focal_loss:
            scores, index = torch.topk(logits.sigmoid().flatten(1), num_top_queries, dim=-1, largest=True, sorted=True)
            labels = torch.remainder(index, num_classes)
            index = torch.floor_divide(index, num_classes)
            boxes = boxes.gather(1, index.unsqueeze(-1).repeat(1, 1, boxes.size(2)))
        else:
            probabilities = torch.softmax(logits, dim=-1)[:, :, : num_classes - 1]
            scores, labels = probabilities.max(dim=-1)

        # Koharu compares the f32 score with the f32 threshold.
        threshold32 = float(np.float32(threshold))
        results: list[ObjectDetectionOutput] = []
        for batch in range(batch_size):
            batch_scores = scores[batch].float().cpu().tolist()
            batch_labels = labels[batch].cpu().tolist()
            batch_boxes = boxes[batch].float().cpu().tolist()
            result = ObjectDetectionOutput()
            for query, score in enumerate(batch_scores):
                if score <= threshold32:
                    continue
                result.scores.append(score)
                result.labels.append(int(batch_labels[query]))
                result.boxes.append(tuple(batch_boxes[query]))
            results.append(result)
        return results


@dataclass
class ObjectDetectionOutput:
    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)


def center_to_corners_format(boxes: "torch.Tensor") -> "torch.Tensor":
    """``center_to_corners_format``: cxcywh to xyxy on the last axis."""
    import torch

    centers = boxes[..., 0:2]
    dimensions = boxes[..., 2:4]
    return torch.cat([centers - dimensions * 0.5, centers + dimensions * 0.5], dim=-1)


@dataclass
class TextBlock:
    """``TextBlock`` (xyxy and bubble_xyxy are whole pixels, as Koharu's ``as i32``)."""

    xyxy: IntBox
    bubble_xyxy: Optional[IntBox]
    text_class: str
    score: float


def _div_ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class ImageSlicer:
    """``ImageSlicer``: tall pages are cut into overlapping 3:1 slices and their boxes merged back."""

    height_to_width_ratio_threshold: float = 3.5
    target_slice_ratio: float = 3.0
    overlap_height_ratio: float = 0.2
    min_slice_height_ratio: float = 0.7
    merge_iou_threshold: float = 0.2
    duplicate_iou_threshold: float = 0.5
    merge_y_distance_threshold: float = 0.1
    containment_threshold: float = 0.85

    def should_slice(self, image: Image.Image) -> bool:
        return image.height / image.width > self.height_to_width_ratio_threshold

    def calculate_slice_params(self, image: Image.Image) -> tuple[int, int, int, int]:
        """(slice_width, slice_height, effective_slice_height, num_slices). The f32 truncations mirror ``as u32``."""
        slice_width = image.width
        slice_height = int(np.float32(slice_width) * np.float32(self.target_slice_ratio))
        effective_slice_height = int(np.float32(slice_height) * (np.float32(1.0) - np.float32(self.overlap_height_ratio)))
        if effective_slice_height <= 0:
            raise ValueError("image is too narrow to slice")
        num_slices = _div_ceil(image.height, effective_slice_height)
        last_slice_start = (num_slices - 1) * effective_slice_height
        last_slice_height = image.height - last_slice_start
        if (np.float32(last_slice_height) / np.float32(slice_height)) < np.float32(self.min_slice_height_ratio) and num_slices > 1:
            num_slices -= 1
        return slice_width, slice_height, effective_slice_height, num_slices

    def get_slice(self, image: Image.Image, slice_number: int, effective_slice_height: int, slice_height: int) -> tuple[Image.Image, int, int]:
        start_y = slice_number * effective_slice_height
        if slice_number == _div_ceil(image.height, effective_slice_height) - 1:
            end_y = image.height
        else:
            end_y = min(start_y + slice_height, image.height)
        return image.crop((0, start_y, image.width, end_y)), start_y, end_y

    def adjust_box_coordinates(self, boxes: Sequence[ScoredBox], start_y: int) -> list[ScoredBox]:
        return [ScoredBox((x0, y0 + start_y, x1, y1 + start_y), score) for (x0, y0, x1, y1), score in boxes]

    def box_contained(self, box1: Box, box2: Box) -> tuple[bool, float, int]:
        """(contained, containment ratio, which: 1 if box1 is the larger, 2 if box2 is, 0 if not contained)."""
        area1 = box_area(box1)
        area2 = box_area(box2)
        intersection = intersection_area(box1, box2)
        if intersection <= 0.0:
            return False, 0.0, 0
        smaller = min(area1, area2)
        containment_ratio = intersection / smaller if smaller != 0.0 else math.inf
        if containment_ratio >= self.containment_threshold:
            return True, containment_ratio, 1 if area1 > area2 else 2
        return False, containment_ratio, 0

    def merge_overlapping_boxes(self, boxes: Sequence[ScoredBox], image_height: int) -> list[ScoredBox]:
        """Merges one slice-level box list: containment, duplicates, and vertically adjacent fragments."""
        boxes = list(boxes)
        y_distance_threshold = self.merge_y_distance_threshold * image_height
        index = 0
        while index + 1 < len(boxes):
            nxt = index + 1
            while nxt < len(boxes):
                box1 = boxes[index]
                box2 = boxes[nxt]
                xyxy1 = box1.xyxy
                xyxy2 = box2.xyxy
                area1 = box_area(xyxy1)
                area2 = box_area(xyxy2)
                score = max(box1.score, box2.score)

                contained, _, which_contains = self.box_contained(xyxy1, xyxy2)
                if contained:
                    keeper = xyxy2 if which_contains == 2 else xyxy1
                    boxes[index] = ScoredBox(keeper, score)
                    del boxes[nxt]
                    continue

                if calculate_iou(xyxy1, xyxy2) >= self.duplicate_iou_threshold:
                    keeper = xyxy2 if area2 > area1 else xyxy1
                    boxes[index] = ScoredBox(keeper, score)
                    del boxes[nxt]
                    continue

                width1 = xyxy1[2] - xyxy1[0]
                height1 = xyxy1[3] - xyxy1[1]
                width2 = xyxy2[2] - xyxy2[0]
                height2 = xyxy2[3] - xyxy2[1]
                y_distance = min(abs(xyxy1[1] - xyxy2[3]), abs(xyxy1[3] - xyxy2[1]))
                local_y_threshold = min(y_distance_threshold, max(height1, height2) * 0.1)
                x_overlap = max(min(xyxy1[2], xyxy2[2]) - max(xyxy1[0], xyxy2[0]), 0.0)
                x_overlap_ratio = x_overlap / max(min(width1, width2), 1.0)
                size_ratio = min(area1, area2) / max(max(area1, area2), 1.0)
                if (
                    y_distance < local_y_threshold
                    and x_overlap_ratio > self.merge_iou_threshold
                    and size_ratio > 0.3
                    and abs(xyxy1[0] - xyxy2[0]) < 0.5 * max(width1, width2)
                    and abs(xyxy1[2] - xyxy2[2]) < 0.5 * max(width1, width2)
                ):
                    merged = merge_boxes(xyxy1, xyxy2)
                    if box_area(merged) > 3.0 * max(area1, area2):
                        nxt += 1
                        continue
                    boxes[index] = ScoredBox(merged, score)
                    del boxes[nxt]
                else:
                    nxt += 1
            index += 1
        return boxes

    def process_slices_for_detection(
        self,
        image: Image.Image,
        detect: Callable[[Image.Image], tuple[list[ScoredBox], list[ScoredBox]]],
    ) -> tuple[list[ScoredBox], list[ScoredBox]]:
        """Returns (bubble boxes, text boxes) in full-page coordinates.

        Koharu computes ``calculate_slice_params`` but takes its slice count from ``div_ceil``
        directly, so the one-slice reduction is not applied to the loop. Mirrored here.
        """
        if not self.should_slice(image):
            return detect(image)

        _, slice_height, effective_slice_height, _ = self.calculate_slice_params(image)
        num_slices = _div_ceil(image.height, effective_slice_height)
        first_slice, _, _ = self.get_slice(image, 0, effective_slice_height, slice_height)
        bubble_boxes, text_boxes = detect(first_slice)
        bubble_boxes = list(bubble_boxes)
        text_boxes = list(text_boxes)
        for slice_number in range(1, num_slices):
            slice_image, start_y, _ = self.get_slice(image, slice_number, effective_slice_height, slice_height)
            slice_bubbles, slice_texts = detect(slice_image)
            bubble_boxes.extend(self.adjust_box_coordinates(slice_bubbles, start_y))
            text_boxes.extend(self.adjust_box_coordinates(slice_texts, start_y))

        return (
            self.merge_overlapping_boxes(bubble_boxes, image.height),
            self.merge_overlapping_boxes(text_boxes, image.height),
        )


def create_text_blocks(
    image_width: int,
    image_height: int,
    text_boxes: Sequence[ScoredBox],
    bubble_boxes: Sequence[ScoredBox],
) -> list[TextBlock]:
    """Pairs each merged text box with the first bubble that fits it (``text_bubble``) or none (``text_free``)."""
    text_boxes = filter_and_fix_bboxes(text_boxes, image_width, image_height)
    bubble_boxes = filter_and_fix_bboxes(bubble_boxes, image_width, image_height)
    text_boxes = merge_overlapping_boxes(text_boxes)

    blocks: list[TextBlock] = []
    for text in text_boxes:
        bubble = next(
            (
                candidate
                for candidate in bubble_boxes
                if does_rectangle_fit(candidate.xyxy, text.xyxy) or calculate_iou(candidate.xyxy, text.xyxy) >= 0.2
            ),
            None,
        )
        blocks.append(
            TextBlock(
                xyxy=_whole(text.xyxy),
                bubble_xyxy=_whole(bubble.xyxy) if bubble is not None else None,
                text_class="text_bubble" if bubble is not None else "text_free",
                score=text.score,
            )
        )
    return blocks


def filter_and_fix_bboxes(boxes: Sequence[ScoredBox], image_width: int, image_height: int) -> list[ScoredBox]:
    """Clamps to the image and drops boxes not larger than 5 px in both directions."""
    fixed: list[ScoredBox] = []
    for (x0, y0, x1, y1), score in boxes:
        x0 = min(max(x0, 0.0), float(image_width))
        x1 = min(max(x1, 0.0), float(image_width))
        y0 = min(max(y0, 0.0), float(image_height))
        y1 = min(max(y1, 0.0), float(image_height))
        if (x1 - x0) > 5.0 and (y1 - y0) > 5.0:
            fixed.append(ScoredBox((x0, y0, x1, y1), score))
    return fixed


def merge_overlapping_boxes(boxes: Sequence[ScoredBox]) -> list[ScoredBox]:
    """Text-box merge: grows each box by any box it mostly contains or is mostly contained by, then de-duplicates."""
    accepted: list[ScoredBox] = []
    for index, own in enumerate(boxes):
        merged = own.xyxy
        score = own.score
        for other_index, other in enumerate(boxes):
            if index != other_index and (is_mostly_contained(merged, other.xyxy, 0.3) or is_mostly_contained(other.xyxy, merged, 0.3)):
                merged = merge_boxes(merged, other.xyxy)
                score = max(score, other.score)
        duplicate = next(
            (position for position, kept in enumerate(accepted) if kept.xyxy == merged or calculate_iou(kept.xyxy, merged) >= 0.5),
            None,
        )
        if duplicate is not None:
            kept = accepted[duplicate]
            accepted[duplicate] = ScoredBox(kept.xyxy, max(kept.score, score))
            continue
        # Koharu also retains only boxes below IoU 0.5 here; after the check above that is a no-op.
        accepted.append(ScoredBox(merged, score))
    return accepted


def is_mostly_contained(outer: Box, inner: Box, threshold: float) -> bool:
    outer_area = box_area(outer)
    inner_area = box_area(inner)
    return outer_area >= inner_area and inner_area != 0.0 and intersection_area(outer, inner) / inner_area >= threshold


def merge_boxes(box1: Box, box2: Box) -> Box:
    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    )


def box_area(bbox: Box) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def intersection_area(box1: Box, box2: Box) -> float:
    return max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0.0) * max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0.0)


def calculate_iou(box1: Box, box2: Box) -> float:
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if union == 0.0:
        return 0.0
    return intersection / union


def does_rectangle_fit(bigger: Box, smaller: Box) -> bool:
    left1, right1 = min(bigger[0], bigger[2]), max(bigger[0], bigger[2])
    top1, bottom1 = min(bigger[1], bigger[3]), max(bigger[1], bigger[3])
    left2, right2 = min(smaller[0], smaller[2]), max(smaller[0], smaller[2])
    top2, bottom2 = min(smaller[1], smaller[3]), max(smaller[1], smaller[3])
    return left1 <= left2 and right1 >= right2 and top1 <= top2 and bottom1 >= bottom2


def _whole(box: Box) -> IntBox:
    """``value as i32``: truncation toward zero."""
    return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
