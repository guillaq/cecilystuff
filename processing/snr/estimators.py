"""Single-frame noise estimators, and the registry that lets you compare them.

The testbed gives frames where the wheel and the beads are moving, so we cannot get the
noise by repeating a static scene and looking at how each pixel varies over time. Every
method here has to infer the noise from one frame instead, which means every one of them
can mistake bead texture for noise. They fail in different ways and by different amounts,
which is exactly why we run all of them and compare.

References
    global_std / robust_mad: the region-of-interest approach, e.g. Constantinides, Atalar
        and McVeigh, "Signal-to-noise measurements in magnitude images from NMR phased
        arrays", Magn. Reson. Med. 38(5):852-857, 1997.
    immerkaer: J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image
        Understanding 64(2):300-302, 1996.
    mad_haar: D. Donoho and I. Johnstone, "Ideal spatial adaptation by wavelet shrinkage",
        Biometrika 81(3):425-455, 1994. The median-absolute-deviation of the finest
        detail band is their standard noise scale estimate.
    block_percentile: A. Amer and E. Dubois, "Fast and reliable structure-oriented video
        noise estimation", IEEE Trans. Circuits Syst. Video Technol. 15(1):113-118, 2005.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.stats import chi2

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Scale factor turning a median absolute deviation into a Gaussian standard deviation.
# It is 1 / Phi^-1(3/4).
MAD_TO_SIGMA = 1.482602218505602

# Immerkaer's 3x3 mask. It is the difference of two Laplacians, chosen so that it
# annihilates any locally linear brightness ramp and responds only to noise.
IMMERKAER_MASK: Array = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float64,
)
# sqrt(sum of squares of the mask) = 6, the factor by which it amplifies white noise.
IMMERKAER_GAIN = 6.0

EstimatorFunc = Callable[[Array, BoolArray | None], float]


@dataclass(frozen=True)
class NoiseEstimator:
    """One way of turning a frame into a noise standard deviation, plus its caveats."""

    name: str
    func: EstimatorFunc
    summary: str
    assumes: str
    fails_when: str


def _region(data: Array, mask: BoolArray | None) -> Array:
    return data.ravel() if mask is None else data[mask]


def _windows_inside(shape: tuple[int, ...], mask: BoolArray | None, size: int) -> BoolArray:
    """Pixels whose size x size neighbourhood lies entirely inside the mask.

    A high-pass estimator reads a whole neighbourhood, so a pixel one step outside the
    background would drag a bead edge into what is supposed to be a signal-free region.
    """
    base = np.ones(shape, dtype=bool) if mask is None else mask
    structure = np.ones((size, size), dtype=bool)
    eroded = ndimage.binary_erosion(base, structure=structure, border_value=False)
    return np.asarray(eroded, dtype=bool)


def _require_samples(count: int, name: str, minimum: int) -> None:
    if count < minimum:
        raise ValueError(
            f"{name} needs at least {minimum} usable samples but the region has {count}; "
            "the mask is too small or too fragmented"
        )


def global_std(data: Array, mask: BoolArray | None = None) -> float:
    """Plain standard deviation of the region.

    Only measures noise if the region really is uniform. On a whole frame it measures
    scene contrast instead, which is why it is here mostly as a baseline to beat.
    """
    values = _region(data, mask)
    _require_samples(values.size, "global_std", 2)
    return float(np.std(values, ddof=1))


def robust_mad(data: Array, mask: BoolArray | None = None) -> float:
    """Standard deviation from the median absolute deviation of the pixel values.

    Same idea as global_std but it shrugs off a few stray bright pixels, so it is the
    better choice when the background mask has caught the edge of a bead.
    """
    values = _region(data, mask)
    _require_samples(values.size, "robust_mad", 2)
    deviation = np.median(np.abs(values - np.median(values)))
    return float(MAD_TO_SIGMA * deviation)


def immerkaer(data: Array, mask: BoolArray | None = None) -> float:
    """Immerkaer's Laplacian estimator.

    Convolving with a Laplacian cancels smooth brightness gradients, so what is left is
    dominated by noise. The mean absolute response is converted to a standard deviation
    with the sqrt(pi/2) factor that relates the two for a Gaussian.
    """
    response = np.asarray(ndimage.convolve(data, IMMERKAER_MASK, mode="constant", cval=0.0))
    valid = _windows_inside(data.shape, mask, 3)
    count = int(valid.sum())
    _require_samples(count, "immerkaer", 1)
    mean_abs = float(np.abs(response[valid]).sum() / count)
    return float(np.sqrt(np.pi / 2.0) * mean_abs / IMMERKAER_GAIN)


def mad_haar(data: Array, mask: BoolArray | None = None) -> float:
    """Median absolute deviation of the diagonal detail band of one Haar step.

    The diagonal band (a - b - c + d) / 2 over each 2x2 block has exactly the same
    standard deviation as the noise, and only responds to structure at the finest scale.
    Taking the median rather than the mean is what makes it survive a textured image.
    """
    height, width = data.shape
    even_h, even_w = height - height % 2, width - width % 2
    if even_h < 2 or even_w < 2:
        raise ValueError("mad_haar needs an image of at least 2x2 pixels")

    top_left = data[0:even_h:2, 0:even_w:2]
    top_right = data[0:even_h:2, 1:even_w:2]
    bottom_left = data[1:even_h:2, 0:even_w:2]
    bottom_right = data[1:even_h:2, 1:even_w:2]
    diagonal = (top_left - top_right - bottom_left + bottom_right) / 2.0

    if mask is None:
        usable = diagonal.ravel()
    else:
        block_ok = (
            mask[0:even_h:2, 0:even_w:2]
            & mask[0:even_h:2, 1:even_w:2]
            & mask[1:even_h:2, 0:even_w:2]
            & mask[1:even_h:2, 1:even_w:2]
        )
        usable = diagonal[block_ok]

    _require_samples(usable.size, "mad_haar", 2)
    return float(MAD_TO_SIGMA * np.median(np.abs(usable)))


def _block_stds(data: Array, mask: BoolArray | None, block: int) -> Array:
    """Standard deviation of every block that lies entirely inside the mask."""
    height, width = data.shape
    rows, cols = height // block, width // block
    if rows == 0 or cols == 0:
        raise ValueError(f"image {height}x{width} is smaller than one {block}x{block} block")

    tiles = data[: rows * block, : cols * block].reshape(rows, block, cols, block)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(rows * cols, block * block)
    stds = np.std(tiles, axis=1, ddof=1)

    if mask is not None:
        flags = mask[: rows * block, : cols * block].reshape(rows, block, cols, block)
        keep = flags.transpose(0, 2, 1, 3).reshape(rows * cols, block * block).all(axis=1)
        stds = stds[keep]
    return np.asarray(stds, dtype=np.float64)


def make_block_percentile(block: int = 8, percentile: float = 10.0) -> EstimatorFunc:
    """Build the homogeneous-block estimator with a given block size and percentile.

    The idea: cut the frame into small blocks, and assume the quietest blocks contain no
    structure, only noise. Taking a low percentile of the block standard deviations picks
    those blocks out without having to detect edges.

    A low percentile of a noise-only distribution sits below the true sigma, so we divide
    by the corresponding quantile of the chi distribution to put it back. That correction
    is exact when every block is noise-only and slightly over-corrects when they are not,
    which is measured in the benchmark.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")

    samples = block * block
    dof = samples - 1
    quantile = float(np.sqrt(chi2.ppf(percentile / 100.0, dof) / dof))

    def estimator(data: Array, mask: BoolArray | None = None) -> float:
        stds = _block_stds(data, mask, block)
        _require_samples(stds.size, "block_percentile", 1)
        return float(np.percentile(stds, percentile) / quantile)

    return estimator


block_percentile = make_block_percentile()


ESTIMATORS: dict[str, NoiseEstimator] = {
    est.name: est
    for est in (
        NoiseEstimator(
            name="global_std",
            func=global_std,
            summary="standard deviation of the pixels in the region",
            assumes="the region is uniformly lit and contains no structure",
            fails_when="run on a whole frame, where it measures scene contrast, not noise",
        ),
        NoiseEstimator(
            name="robust_mad",
            func=robust_mad,
            summary="standard deviation from the median absolute deviation of the pixels",
            assumes="the region is uniform apart from a minority of outlier pixels",
            fails_when="more than about half the region is structure rather than background",
        ),
        NoiseEstimator(
            name="immerkaer",
            func=immerkaer,
            summary="mean absolute response to a 3x3 Laplacian, rescaled to a sigma",
            assumes="noise is additive, Gaussian, and the same everywhere in the region",
            fails_when="the region has fine texture or sharp edges; it uses a mean, so a few "
            "strong edges pull the estimate up a lot",
        ),
        NoiseEstimator(
            name="mad_haar",
            func=mad_haar,
            summary="median absolute deviation of the finest diagonal wavelet detail band",
            assumes="fewer than half the 2x2 blocks in the region straddle an edge",
            fails_when="beads are so densely packed that most 2x2 blocks contain an edge",
        ),
        NoiseEstimator(
            name="block_percentile",
            func=block_percentile,
            summary="bias-corrected 10th percentile of 8x8 block standard deviations",
            assumes="at least 10 percent of the blocks in the region are free of structure",
            fails_when="no block is structure-free, or the region is too small to hold many "
            "blocks; it then reports the quietest textured block instead",
        ),
    )
}


def estimate_all(
    data: Array,
    mask: BoolArray | None = None,
    methods: list[str] | None = None,
) -> dict[str, float]:
    """Run several estimators on the same region and return sigma in DN for each.

    A method that raises (usually because the region is too small) is reported as NaN
    rather than killing the whole run, so one bad frame does not stop a batch.
    """
    names = methods if methods is not None else list(ESTIMATORS)
    unknown = [name for name in names if name not in ESTIMATORS]
    if unknown:
        raise KeyError(f"unknown estimator(s) {unknown}; available: {sorted(ESTIMATORS)}")

    results: dict[str, float] = {}
    for name in names:
        try:
            results[name] = ESTIMATORS[name].func(data, mask)
        except ValueError:
            results[name] = float("nan")
    return results
