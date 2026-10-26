"""Repeated watermark removal over a chapter. Not in Koharu.

Platform logos and scanlator marks are drawn over the art with partial opacity, and the same mark
repeats from page to page. Repainting the mark's box (LaMa) invents new art in place of the art
that is still visible through it, so instead each mark's alpha matte is learned from its repeats
and inverted:

    I = a*W + (1 - a)*B   =>   B = (I - a*W) / (1 - a)

with ``a`` the mark's per-pixel opacity and ``a*W`` its premultiplied colour (``color`` below).

1. Seeds. A watermark detector proposes boxes. Each box's gradient image is matched against every
   page (normalised cross-correlation of Sobel magnitudes, flat windows excluded). A box whose
   match recurs on ``MIN_PAGES`` or more pages is a repeated mark; one that recurs on one page only
   is a texture (lace, screentone), and one that never recurs is left alone.
2. Matte. Across the repeats (one per page), a pixel belongs to the mark when every page has an
   edge there (the per-pixel minimum of edge strength); art differs from page to page and drops
   out. A first background guess ``B_i`` fills that region from its surroundings, then per pixel
   a robust least squares over the repeats solves ``I_i - B_i = a*W - a*B_i``. Alpha is pooled
   over 3x3 after centring each pixel, so only alpha (not the mark's colour) is assumed locally
   constant. ``B_i`` is then re-estimated by unblending and median filtering, a few times.
3. Placements. The matte's own gradients find every placement, including pages the detector
   missed. A placement is kept only when unblending removes most of the mark-shaped edges there
   (a false match gains edges instead). The matte is refitted once from the kept placements.
4. Removal. Each placement is unblended at the sub-pixel offset that leaves the flattest result,
   then lightly denoised (non-local means) on the mark's footprint only, which removes the matte's
   residual grain without blurring the art. Marks are found and removed one at a time, so a mark
   drawn over another (a platform logo on a scanlator banner) does not pollute its matte.

Everything here is NumPy and OpenCV on whole-page arrays; the detector is the only model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# A mark must repeat on this many pages of the chapter to be learned.
MIN_PAGES = 3
# Gradient NCC for a raw candidate crop to count as a repeat (the crop includes art).
SEED_NCC = 0.45
# Gradient NCC for the learned matte's template; verification by unblending decides the rest.
SEARCH_NCC = 0.2
# Detector boxes wider than this share of the page are usually several marks merged into one.
MAX_SEED_WIDTH = 0.6
MIN_SEED_SIDE = 12
# A placement is kept when unblending leaves at most this share of its mark-shaped edges, and it
# had at least this much (in units of one full mark) to begin with.
KEEP_RESIDUAL = 0.6
MIN_EDGES = 0.3
# Alpha above this is too opaque to invert reliably; those pixels are reported for inpainting.
OPAQUE_ALPHA = 0.85
DENOISE_STRENGTH = 4
ALPHA_POOL = 3
# Placements looked at per page and template.
MAX_PEAKS = 4
# A matte whose edges all run one way (structure-tensor coherence) is a line of the art that
# happens to recur, not a mark: unblending it would erase line art. Lettering and logos have
# edges in every direction (a platform logo measures about 0.1, a slanted scanlator URL 0.4).
MAX_COHERENCE = 0.6
# Repeats a matte is learned from (the best matches); more add time, not accuracy.
MAX_MATTE_PAGES = 8


@dataclass
class Candidate:
    page: str
    box: tuple[int, int, int, int]  # left, top, right, bottom
    score: float


@dataclass
class Placement:
    page: str
    x: int  # top-left of the matte on the page; may be negative (the mark runs off the page)
    y: int


@dataclass
class Watermark:
    alpha: np.ndarray  # (h, w) float32 in [0, 1)
    color: np.ndarray  # (h, w, 3) float32: alpha * mark colour
    placements: list[Placement] = field(default_factory=list)


@dataclass
class PageResult:
    image: np.ndarray  # (H, W, 3) uint8: the page with every mark removed
    footprint: np.ndarray  # (H, W) bool: the pixels a removal changed
    opaque: np.ndarray  # (H, W) bool: pixels too opaque to unblend (left for inpainting)


@dataclass
class ChapterResult:
    watermarks: list[Watermark]
    pages: dict[str, PageResult]  # only pages where a mark was removed


def gradient(image: np.ndarray) -> np.ndarray:
    gray = image.mean(2) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


class _Page:
    """A page under analysis: the working copy (marks found so far removed) and its gradients."""

    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image.astype(np.float32)
        self.footprint = np.zeros(image.shape[:2], bool)
        self.opaque = np.zeros(image.shape[:2], bool)
        self._grad: Optional[np.ndarray] = None

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None:
            self._grad = gradient(self.image)
        return self._grad

    def changed(self) -> None:
        self._grad = None

    def crop(self, x: int, y: int, h: int, w: int) -> np.ndarray:
        """``(h, w, 3)`` at ``(x, y)``, NaN where it falls off the page."""
        out = np.full((h, w, 3), np.nan, np.float32)
        height, width = self.image.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = self.image[y0:y1, x0:x1]
        return out


def _fast_size(n: int) -> int:
    """The smallest size >= ``n`` whose prime factors are all 2, 3, 5 or 7."""
    while True:
        m = n
        for prime in (2, 3, 5, 7):
            while m % prime == 0:
                m //= prime
        if m == 1:
            return n
        n += 1


class _Searcher:
    """Template matching over every page at once, on a torch device.

    The pages' gradient images sit in one zero-padded stack (zero beyond a page's edge, so a
    template may hang off it). Matching is ``TM_CCOEFF_NORMED``: the cross-correlation with the
    zero-mean template through one batched FFT, normalised by each window's variance from
    separable box averages.
    """

    def __init__(self, pages: list[_Page], device: str):
        import torch

        self.torch = torch
        self.device = device
        self.pages = pages
        self.sizes = [page.image.shape[:2] for page in pages]
        height, width = max(h for h, _ in self.sizes), max(w for _, w in self.sizes)
        self.stack = torch.zeros((len(pages), height, width), dtype=torch.float32, device=device)
        for index in range(len(pages)):
            self.update(index)

    def update(self, index: int) -> None:
        h, w = self.sizes[index]
        self.stack[index].zero_()
        self.stack[index, :h, :w] = self.torch.from_numpy(self.pages[index].grad).to(self.device)

    def search(self, template: np.ndarray, threshold: float) -> dict[str, list[tuple[int, int, float]]]:
        """Up to ``MAX_PEAKS`` placements per page, as ``(x, y, ncc)`` with ``(x, y)`` the
        template's top-left on the page; the template may hang off the page by half its size."""
        torch = self.torch
        fn = torch.nn.functional
        th, tw = template.shape
        py, px = th // 2, tw // 2
        grads = fn.pad(self.stack, (px, px, py, py))
        n, height, width = grads.shape
        if height < th or width < tw:
            return {}
        t = torch.from_numpy(template.astype(np.float32)).to(self.device)
        centred = t - t.mean()
        norm = float(centred.square().sum().sqrt())
        if norm < 1e-6:
            return {}
        size = (_fast_size(height), _fast_size(width))  # cuFFT is slow on sizes with large prime factors
        correlation = torch.fft.irfft2(torch.fft.rfft2(grads, s=size) * torch.conj(torch.fft.rfft2(centred, s=size)), s=size)
        correlation = correlation[:, : height - th + 1, : width - tw + 1]
        def window_mean(planes):  # separable box average over (th, tw), valid positions only
            return fn.avg_pool2d(fn.avg_pool2d(planes[:, None], (th, 1), stride=1), (1, tw), stride=1)[:, 0]

        count = th * tw
        mean = window_mean(grads)
        variance = (window_mean(grads * grads) - mean * mean).clamp_min(0)  # per pixel of the window
        ncc = correlation / ((variance * count).sqrt() * norm).clamp_min(1e-6)
        # Flat windows correlate with anything: a window needs some of the template's texture.
        flat = variance.sqrt() < 0.3 * float(t.std(unbiased=False))
        ncc = torch.where(flat, torch.full_like(ncc, -1.0), ncc)
        # Peaks: local maxima over half a template, best first.
        kernel = (max(th // 2, 1) | 1, max(tw // 2, 1) | 1)
        pooled = fn.max_pool2d(ncc[:, None], (kernel[0], 1), stride=1, padding=(kernel[0] // 2, 0))  # separable:
        pooled = fn.max_pool2d(pooled, (1, kernel[1]), stride=1, padding=(0, kernel[1] // 2))[:, 0]  # rows, then columns
        peaks = (ncc == pooled) & (ncc >= threshold)
        found: dict[str, list[tuple[int, int, float]]] = {}
        index, ys, xs = peaks.nonzero(as_tuple=True)
        scores = ncc[index, ys, xs]
        order = scores.argsort(descending=True).cpu().numpy()
        index, ys, xs, scores = index.cpu().numpy(), ys.cpu().numpy(), xs.cpu().numpy(), scores.cpu().numpy()
        for k in order:
            name = self.pages[index[k]].name
            placements = found.setdefault(name, [])
            if len(placements) < MAX_PEAKS:
                placements.append((int(xs[k]) - px, int(ys[k]) - py, float(scores[k])))
        return found


def _fill_background(crop: np.ndarray, region: np.ndarray) -> np.ndarray:
    """A first guess of the art under the mark: the region filled from its surroundings."""
    valid = ~np.isnan(crop[..., 0])
    filled = np.where(valid[..., None], crop, np.nanmedian(crop.reshape(-1, 3), 0) if valid.any() else 0)
    image = np.clip(filled, 0, 255).astype(np.uint8)
    out = cv2.inpaint(image, (region & valid).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
    out[~valid] = np.nan
    return out


def estimate_matte(crops: np.ndarray, seed: tuple[int, int, int, int], device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """``(alpha, color)`` of the mark repeated in ``crops`` (n, h, w, 3; NaN off-page), which holds
    the mark inside ``seed`` (x0, y0, x1, y1). Zero where there is no mark. The least squares
    runs on ``device`` (a torch device)."""
    n, h, w = crops.shape[:3]
    valid = ~np.isnan(crops[..., 0])
    count = valid.sum(0)
    edges = []
    for i in range(n):
        filled = np.where(valid[i, ..., None], crops[i], np.nanmedian(crops[i].reshape(-1, 3), 0))
        edges.append(np.where(valid[i], cv2.dilate(gradient(filled), np.ones((3, 3), np.uint8)), np.nan))
    with np.errstate(all="ignore"):
        strength = np.nan_to_num(np.nanmin(np.stack(edges), 0))
    strength[count < MIN_PAGES] = 0
    edge = (strength > max(6.0, 0.2 * float(np.percentile(strength, 99.5)))).astype(np.uint8)
    region = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # The mark is one connected piece through the seed box; detached pieces are leaked art.
    labels_count, labels = cv2.connectedComponents(region)
    x0, y0, x1, y1 = seed
    keep = [k for k in range(1, labels_count) if (labels[y0:y1, x0:x1] == k).any()]
    region = cv2.dilate(np.isin(labels, keep).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    region &= count >= MIN_PAGES
    alpha = np.zeros((h, w), np.float32)
    color = np.zeros((h, w, 3), np.float32)
    if not region.any():
        return alpha, color

    import torch

    background = np.stack([np.nan_to_num(_fill_background(crops[i], region)) for i in range(n)])
    images = torch.from_numpy(np.nan_to_num(crops)).to(device)
    valid_t = torch.from_numpy(valid[..., None].astype(np.float32)).to(device)
    region_t = torch.from_numpy(region).to(device)
    pool = torch.ones((1, 1, ALPHA_POOL, ALPHA_POOL), device=device)

    def pooled(plane: "torch.Tensor") -> "torch.Tensor":
        return torch.nn.functional.conv2d(plane[None, None], pool, padding=ALPHA_POOL // 2)[0, 0]

    for _ in range(4):
        b = torch.from_numpy(background).to(device)
        weight = valid_t
        for _ in range(3):
            y = images - b  # = color - alpha * b
            total = weight.sum(0).clamp_min(1e-6)
            mean_b = (weight * b).sum(0) / total
            mean_y = (weight * y).sum(0) / total
            pooled_cov = pooled((weight * (b - mean_b) * (y - mean_y)).sum((0, 3)))
            pooled_var = pooled((weight * (b - mean_b) ** 2).sum((0, 3)))
            a = (-pooled_cov / pooled_var.clamp_min(1e-3)).clamp(0, 0.95)
            col = mean_y + a[..., None] * mean_b
            residual = (y - (col - a[..., None] * b)).abs().mean(3, keepdim=True)
            # Huber weights: a page whose background guess broke at an art edge counts less there.
            weight = valid_t / (residual / 4.0).clamp_min(1.0)
        # Where the backgrounds barely differ across pages, alpha and colour cannot be told apart:
        # use the alpha of well-determined pixels and solve the colour from the data.
        determined = (pooled_var > 1500 * ALPHA_POOL * ALPHA_POOL) & region_t
        if int(determined.sum()) > 20:
            fill = a[determined].median()
            a = torch.where(determined, a, torch.where(region_t, fill, torch.zeros_like(a)))
            col = mean_y + a[..., None] * mean_b
        a = torch.where(region_t & (a >= 0.02), a, torch.zeros_like(a))
        col = torch.where(a[..., None] > 0, col, torch.zeros_like(col))
        unblended = ((images - col) / (1 - a[..., None])).clamp(0, 255).to(torch.uint8).cpu().numpy()
        alpha, color = a.cpu().numpy(), col.cpu().numpy()
        # Next background guess: unblend, then median away the leftover ringing.
        for i in range(n):
            smooth = cv2.medianBlur(unblended[i], 5).astype(np.float32)
            background[i] = np.where(region[..., None], smooth, background[i])
    return alpha.astype(np.float32), color.astype(np.float32)


def coherence(color: np.ndarray) -> float:
    """How much a matte's edges share one direction: 0 for every direction, 1 for one."""
    gray = color.mean(2).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    return float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / max(jxx + jyy, 1e-6))


def _trim(alpha: np.ndarray, color: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, int, int]]:
    ys, xs = np.nonzero(alpha)
    if len(ys) < 50 or coherence(color) > MAX_COHERENCE:
        return None
    t, l = int(ys.min()), int(xs.min())
    return alpha[t:ys.max() + 1, l:xs.max() + 1], color[t:ys.max() + 1, l:xs.max() + 1], l, t


def _edge_amount(image: np.ndarray, template: np.ndarray, valid: np.ndarray) -> float:
    """How much mark-shaped edge ``image`` holds, in units of one full mark."""
    centred = template - template.mean()
    g = np.where(valid, gradient(image), 0)
    return float(((g - g.mean()) * centred * valid).sum() / max(float((centred * centred * valid).sum()), 1e-6))


def unblend(page: _Page, watermark: Watermark, x: int, y: int, refine: bool) -> Optional[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], float, float]]:
    """The page under one placement with the mark removed, or None when the placement is not the
    mark. Returns ``(pixels, alpha, (x0, y0, x1, y1) on the page, edges before, edges after)``."""
    pad = 2
    h, w = watermark.alpha.shape
    window = page.crop(x - pad, y - pad, h + 2 * pad, w + 2 * pad)
    valid = ~np.isnan(window[..., 0])
    if valid.sum() < 0.3 * window.shape[0] * window.shape[1]:
        return None
    filled = np.nan_to_num(window)
    alpha0 = np.pad(watermark.alpha, pad)
    color0 = np.pad(watermark.color, ((pad, pad), (pad, pad), (0, 0)))
    best = None
    steps = (-0.5, 0.0, 0.5) if refine else (0.0,)
    for dy in steps:
        for dx in steps:
            shift = np.float32([[1, 0, dx], [0, 1, dy]])
            alpha = cv2.warpAffine(alpha0, shift, (alpha0.shape[1], alpha0.shape[0])) if dx or dy else alpha0
            color = cv2.warpAffine(color0, shift, (alpha0.shape[1], alpha0.shape[0])) if dx or dy else color0
            result = (filled - color) / (1 - alpha[..., None])
            inside = (alpha > 0.01) & valid
            cost = float(gradient(result)[inside].mean()) if inside.any() else np.inf
            if best is None or cost < best[0]:
                best = (cost, result, alpha)
    _, result, alpha = best
    template = gradient(color0)
    before, after = _edge_amount(filled, template, valid), _edge_amount(result, template, valid)
    if before < MIN_EDGES or after > KEEP_RESIDUAL * before:
        return None
    x0, y0 = x - pad, y - pad
    return result, alpha, (x0, y0, x0 + window.shape[1], y0 + window.shape[0]), before, after


def _apply(page: _Page, watermark: Watermark, x: int, y: int) -> bool:
    """Removes one placement from the working page (refined, denoised). False if it does not verify."""
    removed = unblend(page, watermark, x, y, refine=True)
    if removed is None:
        return False
    result, alpha, (x0, y0, x1, y1), _, _ = removed
    footprint = cv2.dilate((alpha > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8))
    clean = np.clip(np.nan_to_num(result), 0, 255).astype(np.uint8)
    if DENOISE_STRENGTH:
        denoised = cv2.fastNlMeansDenoisingColored(clean, None, DENOISE_STRENGTH, DENOISE_STRENGTH, 5, 15).astype(np.float32)
        blend = cv2.GaussianBlur(footprint.astype(np.float32), (5, 5), 0)[..., None]
        clean_f = blend * denoised + (1 - blend) * np.nan_to_num(result)
    else:
        clean_f = np.nan_to_num(result)
    height, width = page.image.shape[:2]
    px0, py0, px1, py1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    cx0, cy0 = px0 - x0, py0 - y0
    region = (slice(py0, py1), slice(px0, px1))
    local = (slice(cy0, cy0 + py1 - py0), slice(cx0, cx0 + px1 - px0))
    touched = footprint[local].astype(bool)
    page.image[region] = np.where(touched[..., None], np.clip(clean_f[local], 0, 255), page.image[region])
    page.footprint[region] |= touched
    page.opaque[region] |= alpha[local] > OPAQUE_ALPHA
    page.changed()
    return True


def _placements(searcher: _Searcher, watermark: Watermark) -> list[Placement]:
    by_name = {page.name: page for page in searcher.pages}
    kept = []
    for name, found in searcher.search(gradient(watermark.color), SEARCH_NCC).items():
        for x, y, _ in found:
            if unblend(by_name[name], watermark, x, y, refine=False) is not None:
                kept.append(Placement(name, x, y))
    return kept


def _one_per_page(placements: list[Placement]) -> list[Placement]:
    seen: dict[str, Placement] = {}
    for placement in placements:
        seen.setdefault(placement.page, placement)
    return list(seen.values())


def _learn(searcher: _Searcher, seed: Candidate) -> Optional[Watermark]:
    device = searcher.device
    by_name = {page.name: page for page in searcher.pages}
    x1, y1, x2, y2 = seed.box
    template = gradient(by_name[seed.page].image[y1:y2, x1:x2])
    # The best match per page (a texture repeating within one page counts once).
    best = {name: found[0] for name, found in searcher.search(template, SEED_NCC).items()}
    if len(best) < MIN_PAGES:
        return None
    best = dict(sorted(best.items(), key=lambda item: -item[1][2])[:MAX_MATTE_PAGES])
    # The box may hold only part of the mark (one line of a two-line banner): learn over a window
    # around it, and let the matte decide where the mark ends.
    mx, my = int(0.6 * (x2 - x1)) + 16, int(1.2 * (y2 - y1)) + 16
    h, w = (y2 - y1) + 2 * my, (x2 - x1) + 2 * mx
    crops = np.stack([by_name[name].crop(x - mx, y - my, h, w) for name, (x, y, _) in best.items()])
    trimmed = _trim(*estimate_matte(crops, (mx, my, mx + x2 - x1, my + y2 - y1), device))
    if trimmed is None:
        return None
    watermark = Watermark(trimmed[0], trimmed[1])
    placements = _placements(searcher, watermark)
    if len({p.page for p in placements}) < MIN_PAGES:
        return None
    # Refit from the verified placements, over a slightly larger window.
    grow = 16
    h, w = watermark.alpha.shape
    crops = np.stack([by_name[p.page].crop(p.x - grow, p.y - grow, h + 2 * grow, w + 2 * grow) for p in _one_per_page(placements)[:MAX_MATTE_PAGES]])
    trimmed = _trim(*estimate_matte(crops, (grow, grow, grow + w, grow + h), device))
    if trimmed is not None:
        refit = Watermark(trimmed[0], trimmed[1])
        refit.placements = _placements(searcher, refit)
        if len({p.page for p in refit.placements}) >= MIN_PAGES:
            return refit
    watermark.placements = placements
    return watermark


def _covered(placements: list[Placement], shape: tuple[int, int], candidate: Candidate) -> bool:
    """Whether most of the candidate's box is a mark already found (a box that merely holds a
    smaller mark, a platform logo on a scanlator banner, is still a seed of its own)."""
    x1, y1, x2, y2 = candidate.box
    h, w = shape
    for p in placements:
        if p.page != candidate.page:
            continue
        ix = max(0, min(x2, p.x + w) - max(x1, p.x))
        iy = max(0, min(y2, p.y + h) - max(y1, p.y))
        if ix * iy > 0.5 * (x2 - x1) * (y2 - y1):
            return True
    return False


def remove_watermarks(images: dict[str, np.ndarray], candidates: list[Candidate], device: str = "cpu") -> ChapterResult:
    """Learns the repeated marks among ``candidates`` and removes them from ``images`` (RGB uint8,
    one chapter's pages). ``device`` is the torch device the mattes are fitted on."""
    pages = [_Page(name, image) for name, image in images.items()]
    searcher = _Searcher(pages, device)
    index = {page.name: i for i, page in enumerate(pages)}
    watermarks: list[Watermark] = []
    for seed in sorted(candidates, key=lambda c: -c.score):
        x1, y1, x2, y2 = seed.box
        width = images[seed.page].shape[1]
        if x2 - x1 > MAX_SEED_WIDTH * width or min(x2 - x1, y2 - y1) < MIN_SEED_SIDE:
            continue
        if any(_covered(wm.placements, wm.alpha.shape, seed) for wm in watermarks):
            continue
        watermark = _learn(searcher, seed)
        if watermark is None:
            continue
        watermarks.append(watermark)
        log.info(
            "watermark %d (%dx%d, alpha %.2f) on %d pages",
            len(watermarks),
            watermark.alpha.shape[1],
            watermark.alpha.shape[0],
            float(np.median(watermark.alpha[watermark.alpha > 0])),
            len({p.page for p in watermark.placements}),
        )
        by_name = {page.name: page for page in pages}
        kept = []
        for placement in watermark.placements:
            if _apply(by_name[placement.page], watermark, placement.x, placement.y):
                kept.append(placement)
        for name in {p.page for p in kept}:
            searcher.update(index[name])
        watermark.placements = kept
    results = {
        page.name: PageResult(np.clip(page.image + 0.5, 0, 255).astype(np.uint8), page.footprint, page.opaque)
        for page in pages
        if page.footprint.any()
    }
    return ChapterResult(watermarks, results)
