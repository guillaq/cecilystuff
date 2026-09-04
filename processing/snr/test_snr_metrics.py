"""The arithmetic that turns sigma into a decision, checked by hand."""

import math

import numpy as np
import pytest

from processing.snr.images import Frame
from processing.snr.metrics import (
    ROSE_DETECTABLE,
    ROSE_MARGINAL,
    VERDICT_DETECTABLE,
    VERDICT_MARGINAL,
    VERDICT_NOT_DETECTABLE,
    measure,
    rose_verdict,
)
from processing.snr.segmentation import rect_regions


def frame_with(bead_dn: float, background_dn: float, sigma: float, seed: int = 0) -> Frame:
    """Left half is bead at a known level, right half is background, plus known noise."""
    rng = np.random.default_rng(seed)
    data = np.full((128, 128), background_dn, dtype=np.float64)
    data[:, :64] = bead_dn
    data = data + rng.normal(0.0, sigma, size=data.shape)
    return Frame(name="synthetic", data=data, full_scale=65535.0, source_dtype="float64")


HALVES = ((0, 0, 128, 60), (0, 68, 128, 60))


def test_contrast_and_cnr_match_hand_arithmetic() -> None:
    """Contrast 400 DN over sigma 10 DN is a CNR of 40, whichever method measured it."""
    image = frame_with(bead_dn=1400.0, background_dn=1000.0, sigma=10.0)
    analysis = measure(image, rect_regions(image.shape, *HALVES), noise_regions=("background",))

    for result in analysis.results:
        assert result.contrast_dn == pytest.approx(400.0, rel=0.02)
        assert result.cnr == pytest.approx(result.contrast_dn / result.sigma_dn, rel=1e-9)
        assert result.cnr == pytest.approx(40.0, rel=0.1)


def test_snr_is_inflated_by_a_black_level_but_cnr_is_not() -> None:
    """The reason the README tells people to read CNR and ignore SNR."""
    low = frame_with(1400.0, 1000.0, 10.0, seed=1)
    high = Frame(name="offset", data=low.data + 20000.0, full_scale=65535.0, source_dtype="float64")
    regions = rect_regions(low.shape, *HALVES)

    plain = measure(low, regions, methods=["immerkaer"], noise_regions=("background",)).results[0]
    offset = measure(high, regions, methods=["immerkaer"], noise_regions=("background",)).results[0]

    assert offset.snr > 10 * plain.snr
    assert offset.cnr == pytest.approx(plain.cnr, rel=1e-6)


def test_rose_index_scales_with_the_square_root_of_bead_area() -> None:
    """A tracker averages over a whole bead, so a bigger bead is easier to find."""
    image = frame_with(1400.0, 1000.0, 10.0, seed=2)
    small = rect_regions(image.shape, (0, 0, 10, 10), (0, 68, 128, 60))
    large = rect_regions(image.shape, (0, 0, 40, 40), (0, 68, 128, 60))

    a = measure(image, small, methods=["immerkaer"], noise_regions=("background",)).results[0]
    b = measure(image, large, methods=["immerkaer"], noise_regions=("background",)).results[0]

    assert b.rose_index / a.rose_index == pytest.approx(math.sqrt(1600 / 100), rel=0.05)


def test_rose_index_is_cnr_times_root_area() -> None:
    image = frame_with(1400.0, 1000.0, 10.0, seed=3)
    result = measure(
        image,
        rect_regions(image.shape, *HALVES),
        methods=["immerkaer"],
        noise_regions=("background",),
    ).results[0]

    assert result.rose_index == pytest.approx(result.cnr * math.sqrt(result.bead_area_px), rel=1e-9)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (ROSE_DETECTABLE + 0.1, VERDICT_DETECTABLE),
        (ROSE_DETECTABLE, VERDICT_DETECTABLE),
        (ROSE_MARGINAL, VERDICT_MARGINAL),
        (ROSE_MARGINAL - 0.1, VERDICT_NOT_DETECTABLE),
        (0.0, VERDICT_NOT_DETECTABLE),
    ],
)
def test_verdict_thresholds(index: float, expected: str) -> None:
    assert rose_verdict(index) == expected


def test_verdict_is_unknown_when_the_estimate_failed() -> None:
    assert rose_verdict(float("nan")) == "unknown"


def test_snr_db_is_twenty_log_ten() -> None:
    image = frame_with(1400.0, 1000.0, 10.0, seed=4)
    result = measure(
        image,
        rect_regions(image.shape, *HALVES),
        methods=["immerkaer"],
        noise_regions=("background",),
    ).results[0]

    assert result.snr_db == pytest.approx(20.0 * math.log10(result.snr), rel=1e-9)


def test_spread_is_one_when_the_methods_agree() -> None:
    """On flat noise every method is right, so the spread has to be about 1."""
    image = frame_with(1400.0, 1000.0, 10.0, seed=5)
    analysis = measure(image, rect_regions(image.shape, *HALVES), noise_regions=("background",))

    assert analysis.spread() == pytest.approx(1.0, abs=0.15)


def test_measuring_on_the_frame_and_the_background_gives_different_answers() -> None:
    """The gap between the two is the amount of scene structure a method absorbed."""
    image = frame_with(1400.0, 1000.0, 10.0, seed=6)
    analysis = measure(image, rect_regions(image.shape, *HALVES), methods=["global_std"])

    frame = next(r for r in analysis.results if r.noise_region == "frame")
    background = next(r for r in analysis.results if r.noise_region == "background")
    assert frame.sigma_dn > 10 * background.sigma_dn


def test_empty_region_is_an_error_not_a_silent_nan() -> None:
    image = frame_with(1400.0, 1000.0, 10.0, seed=7)
    regions = rect_regions(image.shape, *HALVES)
    empty = type(regions)(
        bead=np.zeros_like(regions.bead),
        background=regions.background,
        threshold=0.0,
        bead_count=0,
        mean_bead_area_px=0.0,
        boundary_fraction=0.0,
    )

    with pytest.raises(ValueError, match="regions is empty"):
        measure(image, empty)


def test_unknown_noise_region_is_rejected() -> None:
    image = frame_with(1400.0, 1000.0, 10.0, seed=8)
    with pytest.raises(ValueError, match="noise region"):
        measure(image, rect_regions(image.shape, *HALVES), noise_regions=("elsewhere",))


def disk_frame(contrast: float, sigma: float, radius: int = 12, seed: int = 0) -> Frame:
    """One disk of known contrast on a flat background, plus known noise."""
    rng = np.random.default_rng(seed)
    data = np.full((128, 128), 1000.0)
    rows, cols = np.ogrid[:128, :128]
    data[(rows - 64) ** 2 + (cols - 64) ** 2 <= radius**2] += contrast
    return Frame(
        name="disk",
        data=data + rng.normal(0.0, sigma, size=data.shape),
        full_scale=65535.0,
        source_dtype="float64",
    )


def precision_of(contrast: float, sigma: float, seed: int = 0) -> float:
    from processing.snr.segmentation import bead_regions

    image = disk_frame(contrast, sigma, seed=seed)
    regions = bead_regions(image.data, erode_px=2, min_bead_area_px=20)
    analysis = measure(image, regions, methods=["immerkaer"], noise_regions=("background",))
    return analysis.results[0].displacement_precision_px


def test_displacement_precision_worsens_in_proportion_to_the_noise() -> None:
    """The bound is sigma / sqrt(gradient energy), so it is linear in sigma."""
    quiet = precision_of(contrast=600.0, sigma=5.0)
    loud = precision_of(contrast=600.0, sigma=20.0)

    assert loud / quiet == pytest.approx(4.0, rel=0.2)


def test_displacement_precision_improves_in_proportion_to_the_contrast() -> None:
    faint = precision_of(contrast=150.0, sigma=10.0)
    strong = precision_of(contrast=600.0, sigma=10.0)

    assert faint / strong == pytest.approx(4.0, rel=0.25)


def test_a_well_exposed_bead_can_be_located_to_a_small_fraction_of_a_pixel() -> None:
    """Sanity anchor: a 24-pixel bead at CNR 60 should be good to well under 0.1 px."""
    assert precision_of(contrast=600.0, sigma=10.0) < 0.1


def test_precision_is_infinite_when_there_is_no_signal_left() -> None:
    """Subtracting the noise contribution stops pure noise looking like bead edges."""
    from processing.snr.metrics import displacement_precision_px

    rng = np.random.default_rng(1)
    data = rng.normal(1000.0, 25.0, size=(128, 128))
    support = np.ones((128, 128), dtype=bool)

    assert math.isinf(displacement_precision_px(data, support, 25.0, bead_count=1))


def test_precision_is_not_reported_without_beads() -> None:
    from processing.snr.metrics import displacement_precision_px

    data = np.zeros((32, 32))
    support = np.ones((32, 32), dtype=bool)

    assert math.isnan(displacement_precision_px(data, support, 5.0, bead_count=0))
    assert math.isnan(displacement_precision_px(data, support, 0.0, bead_count=4))
