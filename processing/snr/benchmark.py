"""Measure the estimators against images whose noise we chose ourselves.

There is no way to validate a noise estimator on testbed data, because on a real frame
nobody knows the right answer. So we build synthetic bead fields, add a known amount of
noise, and check what each method reports back.

The scenes matter as much as the estimators. On a perfectly flat background every method
gets the right answer and the comparison is worthless. Real regolith has surface texture
and the testbed has a lighting gradient, and it is those two that separate the methods, so
they are part of the test rather than an afterthought.

Three things come out of a run:

    accuracy      does the method recover the sigma we put in?
    scene bias    how much worse does it get with packed beads, texture and a gradient?
    texture bias  what does it report on a scene with structure and no noise at all?

The last one is the decisive test. Every method here infers noise from a single frame, so
every one of them can read surface structure as noise. The texture suite puts a number on
that for each method, which is what lets us pick one for the testbed.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from processing.snr import estimators
from processing.snr.segmentation import bead_regions

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# A plausible testbed frame: 16-bit camera, beads a few hundred DN above the surface.
DEFAULT_BACKGROUND_DN = 2000.0
DEFAULT_BEAD_DN = 2600.0
DEFAULT_RADIUS_PX = 6.0
DEFAULT_SHAPE = (512, 512)

SWEEP_SIGMAS_DN = (2.0, 5.0, 10.0, 20.0, 50.0)
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Scene:
    """A synthetic testbed scene, from the easy case to the one we expect to have.

    `texture_scale_px` is the correlation length of the surface structure. Coarse texture
    is a bumpy surface and is easy for a high-pass estimator to ignore. Fine texture sits
    at the same scale as the noise itself, and nothing can separate the two, which is a
    limit of single-frame estimation rather than of any particular method.
    """

    name: str
    bead_count: int
    min_gap_px: float
    surface_texture_dn: float
    texture_scale_px: float
    gradient_dn: float
    note: str


SCENES: tuple[Scene, ...] = (
    Scene("flat_sparse", 40, 2.0, 0.0, 8.0, 0.0, "few beads, flat surface, even lighting"),
    Scene("flat_dense", 600, 1.0, 0.0, 8.0, 0.0, "packed beads, flat surface"),
    Scene(
        "textured_dense",
        600,
        1.0,
        60.0,
        8.0,
        200.0,
        "packed beads, bumpy surface, lit from one side",
    ),
    Scene(
        "fine_texture_dense",
        600,
        1.0,
        60.0,
        2.0,
        200.0,
        "as above but the surface grain is pixel-scale",
    ),
)


@dataclass(frozen=True)
class SyntheticField:
    """A noise-free scene and the ground truth that goes with it."""

    clean: Array
    bead_mask: BoolArray
    background_mask: BoolArray
    bead_dn: float
    background_dn: float
    radius_px: float
    bead_count: int

    @property
    def contrast_dn(self) -> float:
        return abs(self.bead_dn - self.background_dn)

    @property
    def coverage(self) -> float:
        return float(self.bead_mask.mean())


def _correlated_texture(shape: tuple[int, int], scale_px: float, rng: np.random.Generator) -> Array:
    """Smoothly varying random field with unit standard deviation.

    Blurring white noise gives structure with a chosen correlation length. The result is
    rescaled to unit variance so the caller sets the amplitude in DN directly.
    """
    white = rng.normal(0.0, 1.0, size=shape)
    field = np.asarray(ndimage.gaussian_filter(white, sigma=scale_px), dtype=np.float64)
    spread = float(np.std(field))
    return field / spread if spread > 0 else field


def synthetic_field(
    scene: Scene | None = None,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    radius_px: float = DEFAULT_RADIUS_PX,
    bead_dn: float = DEFAULT_BEAD_DN,
    background_dn: float = DEFAULT_BACKGROUND_DN,
    edge_blur_px: float = 0.8,
    seed: int = 0,
) -> SyntheticField:
    """Build a scene: textured, unevenly lit surface with non-overlapping beads on it.

    Beads are placed by rejection sampling so they do not overlap, which keeps the true
    bead area exactly pi r^2 and lets the Rose index be checked against arithmetic. The
    edge blur stands in for the optics; with perfectly sharp edges the estimators look
    worse than they will in practice.
    """
    scene = scene if scene is not None else SCENES[0]
    rng = np.random.default_rng(seed)
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    rows, cols = np.ogrid[:height, :width]

    centres: list[tuple[float, float]] = []
    minimum_separation = 2.0 * radius_px + scene.min_gap_px
    margin = radius_px + 2.0
    attempts = 0
    while len(centres) < scene.bead_count and attempts < scene.bead_count * 200:
        attempts += 1
        row = float(rng.uniform(margin, height - margin))
        col = float(rng.uniform(margin, width - margin))
        if any(math.hypot(row - r, col - c) < minimum_separation for r, c in centres):
            continue
        centres.append((row, col))
        mask |= (rows - row) ** 2 + (cols - col) ** 2 <= radius_px**2

    surface = np.full(shape, background_dn, dtype=np.float64)
    if scene.gradient_dn != 0.0:
        ramp = np.linspace(-0.5, 0.5, width, dtype=np.float64)
        surface = surface + scene.gradient_dn * ramp[np.newaxis, :]
    if scene.surface_texture_dn > 0:
        surface = surface + scene.surface_texture_dn * _correlated_texture(
            shape, scene.texture_scale_px, rng
        )

    clean = np.where(mask, surface + (bead_dn - background_dn), surface)
    if edge_blur_px > 0:
        clean = np.asarray(ndimage.gaussian_filter(clean, sigma=edge_blur_px), dtype=np.float64)

    # The background used for measurement is pulled well clear of the bead edges, the
    # same way segmentation.bead_regions does it on real frames.
    structure = np.ones((3, 3), dtype=bool)
    dilated = np.asarray(
        ndimage.binary_dilation(mask, structure=structure, iterations=3), dtype=bool
    )

    return SyntheticField(
        clean=np.asarray(clean, dtype=np.float64),
        bead_mask=mask,
        background_mask=~dilated,
        bead_dn=bead_dn,
        background_dn=background_dn,
        radius_px=radius_px,
        bead_count=len(centres),
    )


def add_gaussian_noise(clean: Array, sigma_dn: float, seed: int = 0) -> Array:
    """Additive Gaussian noise, the model every estimator here assumes."""
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, sigma_dn, size=clean.shape)


def add_shot_noise(clean: Array, gain_e_per_dn: float, seed: int = 0) -> Array:
    """Poisson photon noise, the model a real sensor actually follows.

    Noise is no longer constant across the frame: it grows as the square root of the
    signal, so beads are noisier than background. The estimators do not know that, and
    the sweep measures how much it costs them.
    """
    if gain_e_per_dn <= 0:
        raise ValueError("gain must be positive")
    rng = np.random.default_rng(seed)
    electrons = np.clip(clean, 0.0, None) * gain_e_per_dn
    return np.asarray(rng.poisson(electrons), dtype=np.float64) / gain_e_per_dn


@dataclass(frozen=True)
class SweepRow:
    """One estimator's answer on one synthetic frame."""

    suite: str
    scene: str
    noise_model: str
    true_sigma_dn: float
    method: str
    region: str
    estimated_sigma_dn: float
    relative_error: float
    seed: int


def _row(
    suite: str,
    scene: str,
    noise_model: str,
    true_sigma: float,
    method: str,
    region: str,
    estimated: float,
    seed: int,
) -> SweepRow:
    error = (estimated - true_sigma) / true_sigma if true_sigma > 0 else float("nan")
    return SweepRow(
        suite=suite,
        scene=scene,
        noise_model=noise_model,
        true_sigma_dn=true_sigma,
        method=method,
        region=region,
        estimated_sigma_dn=estimated,
        relative_error=error,
        seed=seed,
    )


def _both_regions(field: SyntheticField) -> dict[str, BoolArray | None]:
    return {"background": field.background_mask, "frame": None}


def gaussian_sweep(
    sigmas: tuple[float, ...] = SWEEP_SIGMAS_DN,
    seeds: tuple[int, ...] = SEEDS,
    scenes: tuple[Scene, ...] = SCENES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> list[SweepRow]:
    """Accuracy against known additive Gaussian noise, across all scenes."""
    rows: list[SweepRow] = []
    for scene in scenes:
        for seed in seeds:
            field = synthetic_field(scene, shape=shape, seed=seed)
            for sigma in sigmas:
                noisy = add_gaussian_noise(field.clean, sigma, seed=seed + 1000)
                for region, mask in _both_regions(field).items():
                    for method, value in estimators.estimate_all(noisy, mask).items():
                        rows.append(
                            _row(
                                "gaussian",
                                scene.name,
                                "gaussian",
                                sigma,
                                method,
                                region,
                                value,
                                seed,
                            )
                        )
    return rows


def shot_noise_sweep(
    gains_e_per_dn: tuple[float, ...] = (0.5, 2.0, 8.0),
    seeds: tuple[int, ...] = SEEDS,
    scenes: tuple[Scene, ...] = SCENES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> list[SweepRow]:
    """Accuracy against Poisson noise, which is what a real sensor produces.

    The reference is the true noise in the background region, sqrt(background electrons)
    converted back to DN. Whole-frame measurements are compared against that same
    reference on purpose: over-reporting there is bead and texture structure leaking in,
    and that is the thing worth seeing.
    """
    rows: list[SweepRow] = []
    for scene in scenes:
        for seed in seeds:
            field = synthetic_field(scene, shape=shape, seed=seed)
            for gain in gains_e_per_dn:
                noisy = add_shot_noise(field.clean, gain, seed=seed + 2000)
                true_sigma = math.sqrt(field.background_dn / gain)
                for region, mask in _both_regions(field).items():
                    for method, value in estimators.estimate_all(noisy, mask).items():
                        rows.append(
                            _row(
                                "shot",
                                scene.name,
                                "poisson",
                                true_sigma,
                                method,
                                region,
                                value,
                                seed,
                            )
                        )
    return rows


@dataclass(frozen=True)
class TextureRow:
    """What a method reports on a scene that has structure but no noise at all."""

    scene: str
    edge_blur_px: float
    method: str
    region: str
    spurious_sigma_dn: float
    fraction_of_contrast: float
    seed: int


def texture_suite(
    edge_blurs: tuple[float, ...] = (0.0, 0.8, 2.0),
    seeds: tuple[int, ...] = SEEDS,
    scenes: tuple[Scene, ...] = SCENES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> list[TextureRow]:
    """The honest answer to "does this method confuse the scene with noise".

    The correct output is zero. Anything else is structure the method could not tell
    apart from noise, expressed as a fraction of the bead contrast so it can be compared
    against the contrast-to-noise ratio it will later corrupt.
    """
    rows: list[TextureRow] = []
    for scene in scenes:
        for seed in seeds:
            for blur in edge_blurs:
                field = synthetic_field(scene, shape=shape, edge_blur_px=blur, seed=seed)
                for region, mask in _both_regions(field).items():
                    for method, value in estimators.estimate_all(field.clean, mask).items():
                        rows.append(
                            TextureRow(
                                scene=scene.name,
                                edge_blur_px=blur,
                                method=method,
                                region=region,
                                spurious_sigma_dn=value,
                                fraction_of_contrast=value / field.contrast_dn,
                                seed=seed,
                            )
                        )
    return rows


@dataclass(frozen=True)
class SegmentationRow:
    """How well Otsu recovers the beads we drew, at a given noise level."""

    scene: str
    sigma_dn: float
    flatten_sigma_px: float
    intersection_over_union: float
    area_error: float
    seed: int


def segmentation_suite(
    sigmas: tuple[float, ...] = SWEEP_SIGMAS_DN,
    seeds: tuple[int, ...] = SEEDS,
    scenes: tuple[Scene, ...] = SCENES,
    flatten_options: tuple[float, ...] = (0.0, 25.0),
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> list[SegmentationRow]:
    """Check the region finding, not just the noise estimation.

    Segmentation errors move the contrast and the bead area, and both feed the Rose
    index, so a good noise estimate on top of bad regions still gives a wrong answer.
    The two flatten settings show what removing the lighting gradient is worth.
    """
    rows: list[SegmentationRow] = []
    for scene in scenes:
        for seed in seeds:
            field = synthetic_field(scene, shape=shape, seed=seed)
            true_area = math.pi * field.radius_px**2
            for sigma in sigmas:
                noisy = add_gaussian_noise(field.clean, sigma, seed=seed + 3000)
                for flatten in flatten_options:
                    regions = bead_regions(
                        noisy, erode_px=0, min_bead_area_px=4, flatten_sigma_px=flatten
                    )
                    intersection = float((regions.bead & field.bead_mask).sum())
                    union = float((regions.bead | field.bead_mask).sum())
                    rows.append(
                        SegmentationRow(
                            scene=scene.name,
                            sigma_dn=sigma,
                            flatten_sigma_px=flatten,
                            intersection_over_union=(
                                intersection / union if union > 0 else float("nan")
                            ),
                            area_error=(regions.mean_bead_area_px - true_area) / true_area,
                            seed=seed,
                        )
                    )
    return rows


@dataclass(frozen=True)
class MethodScore:
    """Aggregate verdict for one method on one region, in one scene."""

    suite: str
    scene: str
    method: str
    region: str
    median_relative_error: float
    worst_relative_error: float


def summarise(rows: list[SweepRow]) -> list[MethodScore]:
    """Median and worst-case relative error per method, region and scene."""
    scores: list[MethodScore] = []
    keys = sorted({(r.suite, r.scene, r.method, r.region) for r in rows})
    for suite, scene, method, region in keys:
        errors = [
            abs(r.relative_error)
            for r in rows
            if (r.suite, r.scene, r.method, r.region) == (suite, scene, method, region)
            and math.isfinite(r.relative_error)
        ]
        if not errors:
            continue
        scores.append(
            MethodScore(
                suite=suite,
                scene=scene,
                method=method,
                region=region,
                median_relative_error=float(np.median(errors)),
                worst_relative_error=float(np.max(errors)),
            )
        )
    return scores


@dataclass(frozen=True)
class BenchmarkReport:
    """Everything the validation run produced, ready to be written out."""

    sweeps: list[SweepRow]
    textures: list[TextureRow]
    segmentations: list[SegmentationRow]

    @property
    def scores(self) -> list[MethodScore]:
        return summarise(self.sweeps)


def run_benchmark() -> BenchmarkReport:
    """Run every suite. Takes a few minutes; it is the validation, not a unit test."""
    return BenchmarkReport(
        sweeps=gaussian_sweep() + shot_noise_sweep(),
        textures=texture_suite(),
        segmentations=segmentation_suite(),
    )
