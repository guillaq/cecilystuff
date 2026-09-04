"""The regions matter more than the estimator, so they get their own tests."""

import math

import numpy as np
import pytest
from scipy import ndimage

from processing.snr.segmentation import bead_regions, otsu_threshold, rect_regions


def two_level_image(low: float, high: float, shape=(64, 64)) -> np.ndarray:
    data = np.full(shape, low, dtype=np.float64)
    data[: shape[0] // 2] = high
    return data


def test_otsu_lands_between_two_clear_populations() -> None:
    threshold = otsu_threshold(two_level_image(10.0, 100.0))
    assert 10.0 < threshold < 100.0


def test_otsu_refuses_a_constant_image() -> None:
    with pytest.raises(ValueError, match="constant"):
        otsu_threshold(np.full((16, 16), 5.0))


def test_regions_recover_a_known_disk() -> None:
    """One disk of known radius, no noise. Area and count must come back exactly."""
    radius = 10.0
    data = np.full((128, 128), 100.0)
    rows, cols = np.ogrid[:128, :128]
    disk = (rows - 64) ** 2 + (cols - 64) ** 2 <= radius**2
    data[disk] = 900.0

    regions = bead_regions(data, erode_px=0, min_bead_area_px=4)

    assert regions.bead_count == 1
    assert regions.mean_bead_area_px == pytest.approx(math.pi * radius**2, rel=0.03)
    assert not (regions.bead & regions.background).any()


def test_erosion_separates_the_regions_and_is_reported() -> None:
    """Edge pixels are part bead and part background; they must land in neither."""
    data = np.full((128, 128), 100.0)
    rows, cols = np.ogrid[:128, :128]
    data[(rows - 64) ** 2 + (cols - 64) ** 2 <= 100] = 900.0

    tight = bead_regions(data, erode_px=0)
    eroded = bead_regions(data, erode_px=3)

    assert eroded.bead.sum() < tight.bead.sum()
    assert eroded.boundary_fraction > tight.boundary_fraction


def test_dark_polarity_swaps_the_regions() -> None:
    data = two_level_image(10.0, 100.0)
    bright = bead_regions(data, erode_px=0, min_bead_area_px=1)
    dark = bead_regions(data, polarity="dark", erode_px=0, min_bead_area_px=1)

    assert bright.bead.sum() == pytest.approx(dark.background.sum(), rel=0.05)


def test_specks_are_dropped() -> None:
    data = np.full((64, 64), 100.0)
    data[10, 10] = 900.0  # single hot pixel
    data[30:40, 30:40] = 900.0  # a real object

    regions = bead_regions(data, erode_px=0, min_bead_area_px=4)
    assert regions.bead_count == 1


def test_flattening_rescues_a_gradient_that_defeats_a_global_threshold() -> None:
    """With a strong ramp, one global threshold cuts the image in half instead of
    finding the beads. Removing the ramp first fixes it."""
    rows, cols = np.ogrid[:128, :128]
    data = 100.0 + np.broadcast_to(np.linspace(0, 800, 128)[np.newaxis, :], (128, 128)).copy()
    disk = (rows - 64) ** 2 + (cols - 20) ** 2 <= 100
    data = data + np.where(disk, 400.0, 0.0)

    naive = bead_regions(data, erode_px=0, min_bead_area_px=4)
    flattened = bead_regions(data, erode_px=0, min_bead_area_px=4, flatten_sigma_px=20.0)
    truth = float(disk.sum())

    assert naive.bead.sum() > 10 * truth
    assert flattened.bead.sum() == pytest.approx(truth, rel=0.4)


def test_flattening_is_only_used_to_choose_regions() -> None:
    """The returned masks index the original data, never the blurred copy."""
    rng = np.random.default_rng(0)
    data = rng.normal(500.0, 5.0, size=(64, 64))
    regions = bead_regions(data, erode_px=0, min_bead_area_px=1, flatten_sigma_px=10.0)

    assert regions.bead.shape == data.shape


def test_rect_regions_reject_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        rect_regions((64, 64), (0, 0, 20, 20), (10, 10, 20, 20))


def test_rect_regions_report_the_rectangle_area() -> None:
    regions = rect_regions((64, 64), (0, 0, 10, 8), (30, 30, 20, 20))
    assert regions.mean_bead_area_px == 80.0
    assert math.isnan(regions.threshold)


def test_small_regions_are_flagged() -> None:
    regions = rect_regions((64, 64), (0, 0, 4, 4), (30, 30, 5, 5))
    notes = " ".join(regions.warnings())
    assert "bead pixels" in notes and "background pixels" in notes


def test_background_mask_excludes_the_bead_neighbourhood() -> None:
    """A background pixel touching a bead would carry part of the edge into the noise."""
    data = np.full((128, 128), 100.0)
    rows, cols = np.ogrid[:128, :128]
    disk = (rows - 64) ** 2 + (cols - 64) ** 2 <= 100
    data[disk] = 900.0

    regions = bead_regions(data, erode_px=2, min_bead_area_px=4)
    grown = ndimage.binary_dilation(disk, structure=np.ones((3, 3), bool), iterations=1)

    assert not (regions.background & np.asarray(grown, dtype=bool)).any()
