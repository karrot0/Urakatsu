"""Bubble-aware layout bounds derived from explicit scene relations.

Port of ``bubble.rs``.

Divergence: Rust computes these in ``f32`` (with some ``f64`` intermediates). Python uses
``float`` (f64) throughout, so results can differ in the last bit; the geometric decisions
(tolerances, orientation tests) are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from ..scene.components import Geometry

MAX_CONTOUR_POINTS = 1_024

Point = tuple[float, float]
Polygon = list[Point]
IndexedPolygon = tuple[int, Polygon]


@dataclass(frozen=True)
class LayoutBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class GeometryFrame:
    bounds: LayoutBox
    angle_degrees: float


def contour(geometry: Geometry, frame: GeometryFrame) -> list[Point]:
    """``contour``: geometry points in the frame's local (top-left origin, unrotated) axes."""
    if len(geometry.points) > MAX_CONTOUR_POINTS:
        return []
    bounds = frame.bounds
    center_x = bounds.x + bounds.width * 0.5
    center_y = bounds.y + bounds.height * 0.5
    radians = -math.radians(frame.angle_degrees)
    sin, cos = math.sin(radians), math.cos(radians)
    output = []
    for point in geometry.points:
        x = point.x - center_x
        y = point.y - center_y
        output.append((x * cos - y * sin + center_x - bounds.x, x * sin + y * cos + center_y - bounds.y))
    return output


def point_in_frame(frame: GeometryFrame, x: float, y: float) -> Point:
    bounds = frame.bounds
    center_x = bounds.x + bounds.width * 0.5
    center_y = bounds.y + bounds.height * 0.5
    radians = -math.radians(frame.angle_degrees)
    sin, cos = math.sin(radians), math.cos(radians)
    x = x - center_x
    y = y - center_y
    return (x * cos - y * sin + center_x - bounds.x, x * sin + y * cos + center_y - bounds.y)


def flow_cells(frame: GeometryFrame, polygon: Sequence[Point], anchors: Sequence[Point]) -> list[list[Point]]:
    """Divides a joined balloon into one non-overlapping polygon per text flow.

    Reflex-vertex diagonals recover the physical necks between lobes first. Source anchors
    assign those lobes to flows and only become the partition itself when the contour has no
    complete, non-crossing neck decomposition.
    """
    if not anchors:
        return []
    width = frame.bounds.width
    height = frame.bounds.height
    local = [point_in_frame(frame, x, y) for x, y in anchors]
    local = [(min(max(x, 0.0), width), min(max(y, 0.0), height)) for x, y in local]
    cells = _topological_flow_cells(polygon, local)
    if cells is not None:
        return cells
    return _anchor_flow_cells(width, height, polygon, local)


def _topological_flow_cells(contour_points: Sequence[Point], anchors: Sequence[Point]) -> Optional[list[list[Point]]]:
    if (
        len(contour_points) < 4
        or len(contour_points) > MAX_CONTOUR_POINTS
        or any(not (math.isfinite(x) and math.isfinite(y)) for x, y in [*contour_points, *anchors])
    ):
        return None
    bounds = _polygon_bounds(contour_points)
    if bounds is None:
        return None
    min_x, max_x, min_y, max_y = bounds
    scale = max(min(max_x - min_x, max_y - min_y), 1.0)
    tolerance = scale * 0.0075
    indexed_anchors = list(enumerate(anchors))
    cells = _decompose_lobes(list(contour_points), indexed_anchors, tolerance)
    if cells is None:
        return None
    cells.sort(key=lambda item: item[0])
    if len(cells) != len(anchors):
        return None
    return [cell for _, cell in cells]


@dataclass
class _LobeSplit:
    structural_support: float
    length_squared: float
    first: Polygon
    second: Polygon
    first_anchors: list[tuple[int, Point]]
    second_anchors: list[tuple[int, Point]]


def _decompose_lobes(
    polygon: Polygon, anchors: list[tuple[int, Point]], tolerance: float
) -> Optional[list[IndexedPolygon]]:
    if len(anchors) == 1:
        return [(anchors[0][0], polygon)]
    simplified = _simplify_closed_indices(polygon, tolerance)
    if len(simplified) < 4:
        return None
    # Rust ``f32::signum``: never zero (+0.0 maps to 1.0).
    orientation = math.copysign(1.0, _polygon_area_from_indices(polygon, simplified))
    hull = _convex_hull([polygon[index] for index in simplified])
    minimum_reflex_cross = tolerance * tolerance * 0.05
    reflex: list[tuple[int, float]] = []
    count = len(simplified)
    for index in range(count):
        previous = polygon[simplified[(index + count - 1) % count]]
        current = polygon[simplified[index]]
        nxt = polygon[simplified[(index + 1) % count]]
        cross = _turn(previous, current, nxt) * orientation
        if cross >= -minimum_reflex_cross:
            continue
        edge_product = math.sqrt(_distance_squared(previous, current) * _distance_squared(current, nxt))
        if edge_product > 1.1920929e-07:
            corner_strength = -cross / edge_product
            recession = _distance_to_polygon_boundary(current, hull)
            reflex.append((simplified[index], corner_strength * recession))

    candidates: list[_LobeSplit] = []
    for first_index in range(len(reflex)):
        for second_index in range(first_index + 1, len(reflex)):
            first_vertex, first_support = reflex[first_index]
            second_vertex, second_support = reflex[second_index]
            if not _valid_diagonal(polygon, first_vertex, second_vertex, tolerance):
                continue
            first, second = _split_polygon(polygon, first_vertex, second_vertex)
            if (
                len(first) < 3
                or len(second) < 3
                or abs(_polygon_area(first)) <= tolerance * tolerance
                or abs(_polygon_area(second)) <= tolerance * tolerance
            ):
                continue
            first_anchors: list[tuple[int, Point]] = []
            second_anchors: list[tuple[int, Point]] = []
            assigns_cleanly = True
            cross_epsilon = max(tolerance * tolerance * 0.001, 1.1920929e-07)
            for index, anchor in anchors:
                in_first = _point_in_polygon(first, anchor, cross_epsilon)
                in_second = _point_in_polygon(second, anchor, cross_epsilon)
                if in_first and not in_second:
                    first_anchors.append((index, anchor))
                elif not in_first and in_second:
                    second_anchors.append((index, anchor))
                else:
                    assigns_cleanly = False
                    break
            if not assigns_cleanly or not first_anchors or not second_anchors:
                continue
            length_squared = _distance_squared(polygon[first_vertex], polygon[second_vertex])
            if length_squared <= 1.1920929e-07:
                continue
            candidates.append(
                _LobeSplit(
                    structural_support=min(first_support, second_support) / math.sqrt(length_squared),
                    length_squared=length_squared,
                    first=first,
                    second=second,
                    first_anchors=first_anchors,
                    second_anchors=second_anchors,
                )
            )
    # A structural neck connects sharp corners that are recessed from the bubble's convex
    # envelope. Recession makes outer-wall raster kinks contribute almost no support without
    # imposing a pixel- or scale-specific rejection threshold.
    candidates.sort(key=lambda c: (-c.structural_support, c.length_squared))
    for candidate in candidates:
        first_cells = _decompose_lobes(candidate.first, candidate.first_anchors, tolerance)
        if first_cells is None:
            continue
        second_cells = _decompose_lobes(candidate.second, candidate.second_anchors, tolerance)
        if second_cells is None:
            continue
        return [*first_cells, *second_cells]
    return None


def _convex_hull(points: Sequence[Point]) -> list[Point]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique
    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and _turn(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _turn(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    lower.pop()
    upper.pop()
    return lower + upper


def _distance_to_polygon_boundary(point: Point, polygon: Sequence[Point]) -> float:
    if len(polygon) < 2:
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _simplify_closed_indices(polygon: Sequence[Point], tolerance: float) -> list[int]:
    # Rust max_by: the last of equal maxima wins.
    split = 0
    best = None
    for index in range(1, len(polygon)):
        distance = _distance_squared(polygon[0], polygon[index])
        if best is None or distance >= best:
            best = distance
            split = index
    if split == 0:
        return list(range(len(polygon)))
    first_chain = list(range(0, split + 1))
    second_chain = [*range(split, len(polygon)), 0]
    first = _simplify_chain_indices(polygon, first_chain, tolerance)
    second = _simplify_chain_indices(polygon, second_chain, tolerance)
    first.pop()
    second.pop()
    first.extend(second)
    if len(first) >= 3:
        return first
    return list(range(len(polygon)))


def _simplify_chain_indices(polygon: Sequence[Point], chain: list[int], tolerance: float) -> list[int]:
    if len(chain) <= 2:
        return list(chain)
    start = polygon[chain[0]]
    end = polygon[chain[-1]]
    farthest: Optional[tuple[int, float]] = None
    for position in range(1, len(chain) - 1):
        distance = _point_segment_distance(polygon[chain[position]], start, end)
        if farthest is None or distance > farthest[1]:
            farthest = (position, distance)
    if farthest is None:
        return [chain[0], chain[-1]]
    position, distance = farthest
    if distance <= tolerance:
        return [chain[0], chain[-1]]
    first = _simplify_chain_indices(polygon, chain[: position + 1], tolerance)
    second = _simplify_chain_indices(polygon, chain[position:], tolerance)
    first.pop()
    first.extend(second)
    return first


def _valid_diagonal(polygon: Sequence[Point], first: int, second: int, tolerance: float) -> bool:
    length = len(polygon)
    if first == second or (first + 1) % length == second or (second + 1) % length == first:
        return False
    start = polygon[first]
    end = polygon[second]
    epsilon = max(tolerance * tolerance * 0.001, 1.1920929e-07)
    for edge in range(length):
        nxt = (edge + 1) % length
        if edge == first or edge == second or nxt == first or nxt == second:
            continue
        if _segments_intersect(start, end, polygon[edge], polygon[nxt], epsilon):
            return False
    return all(
        _point_in_polygon(
            polygon,
            (start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction),
            epsilon,
        )
        for fraction in (0.2, 0.5, 0.8)
    )


def _split_polygon(polygon: Sequence[Point], first: int, second: int) -> tuple[Polygon, Polygon]:
    if first > second:
        first, second = second, first
    first_part = list(polygon[first : second + 1])
    second_part = [*polygon[second:], *polygon[: first + 1]]
    return first_part, second_part


def _anchor_flow_cells(width: float, height: float, contour_points: Sequence[Point], anchors: Sequence[Point]) -> list[list[Point]]:
    scale = max(min(width, height), 1.0)
    coincidence_distance_squared = (scale * 0.0025) ** 2
    clusters: list[list] = []  # [site_x, site_y, indices]
    for index, (ax, ay) in enumerate(anchors):
        match = None
        for cluster in clusters:
            dx = ax - cluster[0]
            dy = ay - cluster[1]
            if dx * dx + dy * dy <= coincidence_distance_squared:
                match = cluster
                break
        if match is not None:
            count = float(len(match[2]))
            match[0] = (match[0] * count + ax) / (count + 1.0)
            match[1] = (match[1] * count + ay) / (count + 1.0)
            match[2].append(index)
        else:
            clusters.append([ax, ay, [index]])

    cells: list[list[Point]] = [[] for _ in anchors]
    for cluster_index, (x, y, indices) in enumerate(clusters):
        # The cell is the editable balloon shape, so fallback bisectors must clip the physical
        # contour rather than just its bounding rectangle.
        if len(contour_points) >= 3:
            cell = list(contour_points)
        else:
            cell = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        for other_index, (other_x, other_y, _) in enumerate(clusters):
            if cluster_index == other_index:
                continue
            normal = (other_x - x, other_y - y)
            offset = (other_x * other_x + other_y * other_y - x * x - y * y) * 0.5
            cell = _clip_half_plane(cell, normal, offset)
            if len(cell) < 3:
                break
        if len(indices) == 1:
            cells[indices[0]] = cell
            continue
        if not cell:
            for index in indices:
                cells[index] = []
            continue
        min_x = min(p[0] for p in cell)
        max_x = max(p[0] for p in cell)
        min_y = min(p[1] for p in cell)
        max_y = max(p[1] for p in cell)
        horizontal = max_x - min_x >= max_y - min_y
        minimum, maximum = (min_x, max_x) if horizontal else (min_y, max_y)
        step = (maximum - minimum) / len(indices)
        for order, index in enumerate(indices):
            lower = minimum + step * order
            upper = maximum if order + 1 == len(indices) else lower + step
            if horizontal:
                lower_normal, upper_normal = (-1.0, 0.0), (1.0, 0.0)
            else:
                lower_normal, upper_normal = (0.0, -1.0), (0.0, 1.0)
            strip = _clip_half_plane(cell, lower_normal, -lower)
            cells[index] = _clip_half_plane(strip, upper_normal, upper)
    return cells


def _polygon_bounds(polygon: Sequence[Point]) -> Optional[tuple[float, float, float, float]]:
    if not polygon:
        return None
    first_x, first_y = polygon[0]
    min_x = max_x = first_x
    min_y = max_y = first_y
    for x, y in polygon[1:]:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    return min_x, max_x, min_y, max_y


def _polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        total += first[0] * second[1] - second[0] * first[1]
    return total * 0.5


def _polygon_area_from_indices(polygon: Sequence[Point], indices: Sequence[int]) -> float:
    total = 0.0
    for index in range(len(indices)):
        first = polygon[indices[index]]
        second = polygon[indices[(index + 1) % len(indices)]]
        total += first[0] * second[1] - second[0] * first[1]
    return total * 0.5


def _point_in_polygon(polygon: Sequence[Point], point: Point, epsilon: float) -> bool:
    inside = False
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(first, second, point, epsilon):
            return True
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing = (second[0] - first[0]) * (point[1] - first[1]) / (second[1] - first[1]) + first[0]
            if point[0] < crossing:
                inside = not inside
    return inside


def _segments_intersect(first_start: Point, first_end: Point, second_start: Point, second_end: Point, epsilon: float) -> bool:
    first_side_start = _orientation(first_start, first_end, second_start)
    first_side_end = _orientation(first_start, first_end, second_end)
    second_side_start = _orientation(second_start, second_end, first_start)
    second_side_end = _orientation(second_start, second_end, first_end)
    crosses = (
        (first_side_start > epsilon and first_side_end < -epsilon)
        or (first_side_start < -epsilon and first_side_end > epsilon)
    ) and (
        (second_side_start > epsilon and second_side_end < -epsilon)
        or (second_side_start < -epsilon and second_side_end > epsilon)
    )
    return (
        crosses
        or _point_on_segment(first_start, first_end, second_start, epsilon)
        or _point_on_segment(first_start, first_end, second_end, epsilon)
        or _point_on_segment(second_start, second_end, first_start, epsilon)
        or _point_on_segment(second_start, second_end, first_end, epsilon)
    )


def _point_on_segment(start: Point, end: Point, point: Point, epsilon: float) -> bool:
    return (
        abs(_orientation(start, end, point)) <= epsilon
        and min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])


def _turn(previous: Point, current: Point, nxt: Point) -> float:
    return (current[0] - previous[0]) * (nxt[1] - current[1]) - (current[1] - previous[1]) * (nxt[0] - current[0])


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    segment = (end[0] - start[0], end[1] - start[1])
    length_squared = segment[0] * segment[0] + segment[1] * segment[1]
    if length_squared <= 1.1920929e-07:
        return math.sqrt(_distance_squared(point, start))
    fraction = ((point[0] - start[0]) * segment[0] + (point[1] - start[1]) * segment[1]) / length_squared
    fraction = min(max(fraction, 0.0), 1.0)
    return math.sqrt(
        _distance_squared(point, (start[0] + segment[0] * fraction, start[1] + segment[1] * fraction))
    )


def _distance_squared(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _clip_half_plane(polygon: Sequence[Point], normal: Point, offset: float) -> list[Point]:
    if not polygon:
        return []

    def signed(point: Point) -> float:
        return point[0] * normal[0] + point[1] * normal[1] - offset

    output: list[Point] = []
    previous = polygon[-1]
    previous_distance = signed(previous)
    for current in polygon:
        current_distance = signed(current)
        previous_inside = previous_distance <= 1.1920929e-07
        current_inside = current_distance <= 1.1920929e-07
        if previous_inside != current_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > 1.1920929e-07:
                fraction = previous_distance / denominator
                output.append(
                    (
                        previous[0] + (current[0] - previous[0]) * fraction,
                        previous[1] + (current[1] - previous[1]) * fraction,
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
    return output


def geometry_bounds(geometry: Geometry) -> Optional[LayoutBox]:
    """Axis-aligned bounds of a geometry, or None when it is degenerate or non-finite."""
    if any(not (math.isfinite(p.x) and math.isfinite(p.y)) for p in geometry.points):
        return None
    if not geometry.points:
        return None
    first = geometry.points[0]
    min_x = max_x = first.x
    min_y = max_y = first.y
    for point in geometry.points[1:]:
        min_x = min(min_x, point.x)
        min_y = min(min_y, point.y)
        max_x = max(max_x, point.x)
        max_y = max(max_y, point.y)
    x = float(min_x)
    y = float(min_y)
    width = float(max_x - min_x)
    height = float(max_y - min_y)
    values = (x, y, width, height)
    if not all(math.isfinite(v) for v in values) or width <= 0.0 or height <= 0.0:
        return None
    return LayoutBox(x=x, y=y, width=width, height=height)


def geometry_frame(geometry: Geometry, angle_degrees: Optional[float] = None) -> Optional[GeometryFrame]:
    """``geometry_frame``: rotated frame of a quad (or AABB fallback) in page coordinates."""
    if angle_degrees is not None:
        if not math.isfinite(angle_degrees) or not geometry.points:
            return None
        # Measure the contour in the authored text axes. A polygon alone cannot recover those
        # axes after a rotation, even when its bounds are square.
        radians = math.radians(angle_degrees)
        sin, cos = math.sin(radians), math.cos(radians)
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        for point in geometry.points:
            if not (math.isfinite(point.x) and math.isfinite(point.y)):
                return None
            x = point.x * cos + point.y * sin
            y = -point.x * sin + point.y * cos
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        bounds = LayoutBox(
            x=center_x * cos - center_y * sin - width * 0.5,
            y=center_x * sin + center_y * cos - height * 0.5,
            width=width,
            height=height,
        )
        values = (bounds.x, bounds.y, bounds.width, bounds.height)
        if all(math.isfinite(v) for v in values) and bounds.width > 0.0 and bounds.height > 0.0:
            return GeometryFrame(bounds=bounds, angle_degrees=angle_degrees)
        return None

    if len(geometry.points) != 4:
        bounds = geometry_bounds(geometry)
        return GeometryFrame(bounds=bounds, angle_degrees=0.0) if bounds is not None else None
    top_left, top_right, bottom_right, bottom_left = geometry.points
    top = (top_right.x - top_left.x, top_right.y - top_left.y)
    right = (bottom_right.x - top_right.x, bottom_right.y - top_right.y)
    bottom = (bottom_left.x - bottom_right.x, bottom_left.y - bottom_right.y)
    left = (top_left.x - bottom_left.x, top_left.y - bottom_left.y)
    width = math.hypot(*top)
    height = math.hypot(*right)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 2.220446049250313e-16 or height <= 2.220446049250313e-16:
        return None

    scale = max(width, height, 1.0)
    length_tolerance = scale * 1e-6
    opposite_lengths_match = abs(math.hypot(*bottom) - width) <= length_tolerance and abs(
        math.hypot(*left) - height
    ) <= length_tolerance
    perpendicular = abs(top[0] * right[0] + top[1] * right[1]) <= width * height * 1e-6
    diagonals_bisect = (
        abs((top_left.x + bottom_right.x) - (top_right.x + bottom_left.x)) <= length_tolerance
        and abs((top_left.y + bottom_right.y) - (top_right.y + bottom_left.y)) <= length_tolerance
    )
    if not opposite_lengths_match or not perpendicular or not diagonals_bisect:
        bounds = geometry_bounds(geometry)
        return GeometryFrame(bounds=bounds, angle_degrees=0.0) if bounds is not None else None

    center_x = (top_left.x + top_right.x + bottom_right.x + bottom_left.x) * 0.25
    center_y = (top_left.y + top_right.y + bottom_right.y + bottom_left.y) * 0.25
    return GeometryFrame(
        bounds=LayoutBox(x=center_x - width * 0.5, y=center_y - height * 0.5, width=width, height=height),
        angle_degrees=math.degrees(math.atan2(top[1], top[0])),
    )
