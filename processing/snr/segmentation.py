"""Decide which pixels are beads and which are background.

The contrast-to-noise ratio needs two regions: the beads, and a signal-free patch to
measure noise in. Getting those regions wrong matters more than the choice of noise
estimator, because a background region that clips the edge of a bead inflates every
estimate at once. That is why both regions are eroded away from the boundary.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

Rect = tuple[int, int, int, int]  # row, column, height, width


@dataclass(frozen=True)
class Regions:
    """Two disjoint pixel sets plus what we learned while building them."""

    bead: BoolArray
    background: BoolArray
    threshold: float
    bead_count: int
    mean_bead_area_px: float
    boundary_fraction: float

    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.bead.sum() < 100:
            notes.append(f"only {int(self.bead.sum())} bead pixels survived erosion")
        if self.background.sum() < 1000:
            notes.append(
                f"only {int(self.background.sum())} background pixels; block_percentile and "
                "mad_haar need a few thousand to be stable"
            )
        if self.boundary_fraction > 0.4:
            notes.append(
                f"{self.boundary_fraction:.0%} of the frame is bead/background boundary, so "
                "the beads are small or densely packed relative to the erosion width"
            )
        return notes


def otsu_threshold(data: Array, bins: int = 256) -> float:
    """Otsu's threshold: the split that minimises the variance within the two classes.

    Written out rather than imported so a reviewer can see there is no hidden smoothing
    or rescaling. Nobuyuki Otsu, "A threshold selection method from gray-level
    histograms", IEEE Trans. Syst. Man Cybern. 9(1):62-66, 1979.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    counts, edges = np.histogram(finite, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0

    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    usable = (weight_low > 0) & (weight_high > 0)
    if not usable.any():
        raise ValueError("image is constant, so there is nothing to threshold")

    total = np.cumsum(counts * centres)
    mean_low = np.divide(total, weight_low, out=np.zeros_like(total), where=weight_low > 0)
    mean_high = np.divide(
        total[-1] - total, weight_high, out=np.zeros_like(total), where=weight_high > 0
    )
    # Between-class variance. Maximising it is the same as minimising within-class.
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~usable] = -np.inf
    return float(edges[int(np.argmax(between)) + 1])


def flatten_illumination(data: Array, sigma_px: float) -> Array:
    """Remove a slowly varying lighting gradient before thresholding.

    A single global threshold fails when one corner of the testbed is brighter than the
    other. Subtracting a heavily blurred copy fixes that. Only use the result to choose
    regions, never to measure noise: the blur correlates neighbouring pixels.
    """
    if sigma_px <= 0:
        return data
    background = np.asarray(ndimage.gaussian_filter(data, sigma=sigma_px), dtype=np.float64)
    return data - background + float(np.mean(data))


def _erode(mask: BoolArray, pixels: int) -> BoolArray:
    if pixels <= 0:
        return mask
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(
        mask, structure=structure, iterations=pixels, border_value=False
    )
    return np.asarray(eroded, dtype=bool)


def _drop_small(mask: BoolArray, min_area_px: int) -> tuple[BoolArray, int, float]:
    """Remove specks, then report how many objects are left and how big they are."""
    labels_any, count = ndimage.label(mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = np.asarray(labels_any, dtype=np.int64)
    if count == 0:
        return mask, 0, 0.0

    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    keep = areas >= max(min_area_px, 1)
    if not keep.any():
        return np.zeros_like(mask), 0, 0.0

    kept_labels = np.flatnonzero(keep) + 1
    cleaned = np.isin(labels, kept_labels)
    return cleaned, int(keep.sum()), float(areas[keep].mean())


def bead_regions(
    data: Array,
    polarity: str = "bright",
    erode_px: int = 2,
    min_bead_area_px: int = 4,
    flatten_sigma_px: float = 0.0,
) -> Regions:
    """Split a frame into beads and background with Otsu plus an erosion margin.

    `polarity` says whether the beads are the brighter or the darker class. There is no
    reliable way to guess this from one frame, so it is an input, not a detection.

    `erode_px` pulls both regions away from the bead edges. Those edge pixels are part
    bead and part background; leaving them in the background is the single easiest way
    to over-estimate the noise.
    """
    if polarity not in {"bright", "dark"}:
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")

    for_threshold = flatten_illumination(data, flatten_sigma_px)
    threshold = otsu_threshold(for_threshold)
    above = for_threshold > threshold
    bead_raw = above if polarity == "bright" else ~above

    bead_clean, bead_count, mean_area = _drop_small(bead_raw, min_bead_area_px)
    bead = _erode(bead_clean, erode_px)
    background = _erode(~bead_clean, erode_px)
    assigned = int(bead.sum()) + int(background.sum())

    return Regions(
        bead=bead,
        background=background,
        threshold=threshold,
        bead_count=bead_count,
        # Measured before erosion: the Rose index needs the real bead footprint, and
        # erosion would shrink it.
        mean_bead_area_px=mean_area,
        boundary_fraction=1.0 - assigned / data.size,
    )


def rect_regions(shape: tuple[int, int], signal: Rect, background: Rect) -> Regions:
    """Build regions from two rectangles you picked by hand.

    Slower to use but fully traceable: no threshold, no segmentation, nothing to argue
    about in review. This is the method to fall back on when Otsu misbehaves.
    """
    bead = np.zeros(shape, dtype=bool)
    back = np.zeros(shape, dtype=bool)
    for rect, target in ((signal, bead), (background, back)):
        row, col, height, width = rect
        if height <= 0 or width <= 0:
            raise ValueError(f"rectangle {rect} has a non-positive size")
        target[row : row + height, col : col + width] = True

    if (bead & back).any():
        raise ValueError("the signal and background rectangles overlap")

    area = float(signal[2] * signal[3])
    return Regions(
        bead=bead,
        background=back,
        threshold=float("nan"),
        bead_count=1,
        mean_bead_area_px=area,
        boundary_fraction=1.0 - (bead.sum() + back.sum()) / (shape[0] * shape[1]),
    )
