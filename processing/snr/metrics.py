"""Turn a noise estimate and a pair of regions into numbers you can act on.

Three different ratios get called "SNR" in the literature and they are not
interchangeable:

    snr        bead brightness divided by the noise. Inflated by any black-level offset,
               because a constant added to every pixel raises the signal and not the
               noise. Reported for comparison with camera datasheets, not for decisions.
    cnr        the bead-to-background contrast divided by the noise, per pixel. This is
               the one that says whether a bead stands out from what surrounds it.
    rose_index the same contrast integrated over the area of one bead. A tracker looks at
               a whole bead, not one pixel, so this is what governs whether the beads can
               actually be found and followed.

The Rose criterion says an object needs an integrated signal-to-noise around 3 to 5 to be
reliably detected. See A. Rose, "Vision: Human and Electronic", Plenum Press, 1973, and
the summary in Bushberg et al., "The Essential Physics of Medical Imaging".

Detection is not the same question as displacement measurement, and for beads a few tens
of pixels across it is almost always the easier one. So we also report the best possible
precision for locating a bead, which is what actually limits a wheel displacement
measurement. See `displacement_precision_px`.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from processing.snr import estimators
from processing.snr.images import Frame
from processing.snr.segmentation import Regions

Array = npt.NDArray[np.float64]

ROSE_DETECTABLE = 5.0
ROSE_MARGINAL = 3.0

VERDICT_DETECTABLE = "detectable"
VERDICT_MARGINAL = "marginal"
VERDICT_NOT_DETECTABLE = "not detectable"
VERDICT_UNKNOWN = "unknown"


def rose_verdict(rose_index: float) -> str:
    """Plain-language reading of the integrated detectability index."""
    if not math.isfinite(rose_index):
        return VERDICT_UNKNOWN
    if rose_index >= ROSE_DETECTABLE:
        return VERDICT_DETECTABLE
    if rose_index >= ROSE_MARGINAL:
        return VERDICT_MARGINAL
    return VERDICT_NOT_DETECTABLE


@dataclass(frozen=True)
class MethodResult:
    """What one noise estimator, measured over one region, says about one frame."""

    image: str
    method: str
    noise_region: str
    sigma_dn: float
    bead_mean_dn: float
    background_mean_dn: float
    contrast_dn: float
    bead_area_px: float
    snr: float
    snr_db: float
    cnr: float
    rose_index: float
    verdict: str
    displacement_precision_px: float


@dataclass(frozen=True)
class ImageAnalysis:
    """Every method's answer for one frame, plus the caveats that apply to all of them."""

    image: str
    shape: tuple[int, int]
    bead_count: int
    mean_bead_area_px: float
    threshold_dn: float
    results: list[MethodResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def spread(self) -> float:
        """Ratio of the largest to the smallest sigma across methods, on one region.

        A ratio near 1 means the methods agree and the choice does not matter. A large
        ratio means the frame violates at least one method's assumptions, and you have to
        decide which one to believe rather than averaging them.
        """
        sigmas = [r.sigma_dn for r in self.results if math.isfinite(r.sigma_dn) and r.sigma_dn > 0]
        if len(sigmas) < 2:
            return float("nan")
        return max(sigmas) / min(sigmas)


def displacement_precision_px(
    data: Array,
    support: npt.NDArray[np.bool_],
    sigma_dn: float,
    bead_count: int,
) -> float:
    """Best achievable precision for locating one bead, in pixels. Smaller is better.

    Detecting a bead and measuring how far it moved are different problems. A 12-pixel
    bead is trivially detectable long after its displacement has become too noisy to
    measure, so this is the number that limits a wheel displacement experiment.

    The bound is the Cramer-Rao limit for estimating a translation under additive noise:
    the variance of any unbiased estimate of a shift is at least sigma^2 divided by the
    summed squared image gradient over the region being matched. Steep bead edges carry
    the information, flat interiors carry none. This is the standard result behind
    digital image correlation, see Sutton, Orteu and Schreier, "Image Correlation for
    Shape, Motion and Deformation Measurements", Springer 2009, chapter 5.

    Two things to keep in mind. It is a lower bound, so real correlation software will do
    worse, typically by a factor of two or more once interpolation, bead rotation and
    partial occlusion are included. And the measured gradient contains noise of its own,
    which would make the bound look better than it is, so the noise contribution is
    subtracted before the bound is formed.
    """
    if bead_count <= 0 or sigma_dn <= 0:
        return float("nan")

    # np.gradient uses a central difference inside the array and a one-sided difference on
    # the outer row and column. The one-sided version carries four times the noise, so the
    # border is dropped rather than modelled.
    interior = np.zeros(data.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    usable = support & interior
    pixels = int(usable.sum())
    if pixels == 0:
        return float("nan")

    gradient_x = np.gradient(data, axis=1)
    gradient_y = np.gradient(data, axis=0)

    # A central difference over independent pixels has variance sigma^2 / 2, so that much
    # gradient energy per pixel is noise rather than bead edge. Subtracting the expected
    # amount is not enough on its own: the subtraction has its own sampling spread, and
    # without a significance check pure noise yields a confident and completely wrong
    # bound. Below three standard deviations of that spread we report nothing.
    noise_per_pixel = sigma_dn**2 / 2.0
    expected_noise = pixels * noise_per_pixel
    noise_spread = math.sqrt(2.0 * pixels) * noise_per_pixel

    energies = [float(np.sum(g[usable] ** 2)) - expected_noise for g in (gradient_x, gradient_y)]
    signal_energy = min(energies)
    if signal_energy < 3.0 * noise_spread:
        return float("inf")

    return sigma_dn / math.sqrt(signal_energy / bead_count)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("inf")


def _decibels(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return 20.0 * math.log10(ratio)


def measure(
    image: Frame,
    regions: Regions,
    methods: list[str] | None = None,
    noise_regions: tuple[str, ...] = ("background", "frame"),
) -> ImageAnalysis:
    """Run the requested estimators over the requested regions and assemble the results.

    `noise_regions` picks where the noise is measured. "background" is the defensible
    choice: it is signal-free by construction. "frame" runs the blind estimators over
    everything, which is how they are normally published, and the gap between the two
    tells you how much bead texture each method is absorbing.
    """
    data = image.data
    bead_values = data[regions.bead]
    background_values = data[regions.background]
    if bead_values.size == 0 or background_values.size == 0:
        raise ValueError(
            f"{image.name}: one of the regions is empty, so there is no contrast to measure"
        )

    bead_mean = float(np.mean(bead_values))
    background_mean = float(np.mean(background_values))
    contrast = abs(bead_mean - background_mean)
    area = regions.mean_bead_area_px

    results: list[MethodResult] = []
    for region_name in noise_regions:
        if region_name == "background":
            mask = regions.background
        elif region_name == "frame":
            mask = None
        else:
            raise ValueError(f"noise region must be 'background' or 'frame', got {region_name!r}")

        for method, sigma in estimators.estimate_all(data, mask, methods).items():
            cnr = _ratio(contrast, sigma)
            rose = _ratio(contrast * math.sqrt(max(area, 0.0)), sigma)
            snr = _ratio(bead_mean, sigma)
            # Beads plus their edges: the edges are where the gradient information is, and
            # the eroded bead mask has had them removed on purpose.
            support = ~regions.background
            results.append(
                MethodResult(
                    image=image.name,
                    method=method,
                    noise_region=region_name,
                    sigma_dn=sigma,
                    bead_mean_dn=bead_mean,
                    background_mean_dn=background_mean,
                    contrast_dn=contrast,
                    bead_area_px=area,
                    snr=snr,
                    snr_db=_decibels(snr),
                    cnr=cnr,
                    rose_index=rose,
                    verdict=rose_verdict(rose),
                    displacement_precision_px=displacement_precision_px(
                        data, support, sigma, regions.bead_count
                    ),
                )
            )

    return ImageAnalysis(
        image=image.name,
        shape=image.shape,
        bead_count=regions.bead_count,
        mean_bead_area_px=area,
        threshold_dn=regions.threshold,
        results=results,
        warnings=image.quality_warnings() + regions.warnings(),
    )
