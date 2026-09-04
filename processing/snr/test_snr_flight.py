"""The flight prediction is arithmetic on a published noise model, so check the arithmetic.

The MTF test is the one that matters: it verifies the blur width numerically rather than
trusting the algebra that produced it.
"""

import math

import numpy as np
import pytest
from scipy import ndimage

from processing.snr.flight import (
    WHEELCAM,
    electrons_from_testbed,
    mtf_blur_sigma_px,
    predict,
    quantise,
    resample_to_flight_scale,
)


def measured_mtf_at_nyquist(sigma_px: float, length: int = 512) -> float:
    """Blur an impulse, transform it, and read the contrast left at Nyquist.

    This does not reuse the formula under test: it applies the actual blur and measures
    the result, so a wrong constant in mtf_blur_sigma_px would show up here.
    """
    impulse = np.zeros(length)
    impulse[length // 2] = 1.0
    blurred = ndimage.gaussian_filter1d(impulse, sigma=sigma_px, mode="wrap")
    spectrum = np.abs(np.fft.rfft(blurred))
    return float(spectrum[length // 2] / spectrum[0])


@pytest.mark.parametrize("target", [0.1, 0.2, 0.5, 0.8])
def test_blur_width_reproduces_the_requested_mtf(target: float) -> None:
    assert measured_mtf_at_nyquist(mtf_blur_sigma_px(target)) == pytest.approx(target, rel=0.02)


def test_wheelcam_blur_is_about_two_thirds_of_a_pixel() -> None:
    """Sanity anchor for the published MTF > 0.2 at Nyquist."""
    assert mtf_blur_sigma_px(WHEELCAM.mtf_at_nyquist) == pytest.approx(0.683, abs=0.005)


def test_the_continuous_gaussian_formula_would_have_been_wrong() -> None:
    """Records why the width is solved numerically instead of taken from the textbook.

    exp(-2 pi^2 sigma^2 f^2) at f = 0.5 is the continuous Gaussian MTF. Using it here
    would leave twice the contrast at Nyquist that we asked for, because the blur is
    narrower than one pixel and the sampled kernel is not the continuous one.
    """
    continuous = math.sqrt(-2.0 * math.log(0.2) / math.pi**2)
    assert measured_mtf_at_nyquist(continuous) == pytest.approx(0.4, rel=0.05)
    assert mtf_blur_sigma_px(0.2) > continuous


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_impossible_mtf_values_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="MTF at Nyquist"):
        mtf_blur_sigma_px(bad)


def test_pixel_scale_follows_similar_triangles() -> None:
    """Twice the range, twice the ground sampling distance."""
    near = WHEELCAM.pixel_scale_um(distance_cm=30.0)
    far = WHEELCAM.pixel_scale_um(distance_cm=60.0)

    assert far == pytest.approx(2.0 * near, rel=1e-9)
    # The paper quotes 100 um at 30 cm; flat-field geometry gives a little under that.
    assert 85.0 < near < 100.0


def test_bright_scenes_are_shot_noise_limited() -> None:
    result = predict(signal_e=1e6, contrast_e=1e5, bead_diameter_mm=3.0, integration_time_s=0.001)

    assert result.total_noise_e == pytest.approx(math.sqrt(1e6), rel=0.01)
    assert "shot noise" in result.dominant_noise()


def test_dark_scenes_are_read_noise_limited() -> None:
    result = predict(signal_e=1.0, contrast_e=0.5, bead_diameter_mm=3.0, integration_time_s=0.0)

    assert result.total_noise_e == pytest.approx(WHEELCAM.read_noise_e, rel=0.01)
    assert "read noise" in result.dominant_noise()


def test_long_exposures_become_dark_current_limited() -> None:
    """560 e-/s means a one second frame collects 560 electrons of dark signal alone."""
    result = predict(signal_e=10.0, contrast_e=5.0, bead_diameter_mm=3.0, integration_time_s=1.0)

    assert result.dark_noise_e == pytest.approx(math.sqrt(560.0), rel=0.01)
    assert "dark current" in result.dominant_noise()


def test_noise_terms_add_in_quadrature() -> None:
    result = predict(
        signal_e=5000.0, contrast_e=800.0, bead_diameter_mm=2.0, integration_time_s=0.05
    )
    quadrature = math.sqrt(result.shot_noise_e**2 + result.dark_noise_e**2 + result.read_noise_e**2)

    assert result.total_noise_e == pytest.approx(quadrature, rel=1e-9)


def test_cnr_is_proportional_to_contrast() -> None:
    single = predict(
        signal_e=5000.0, contrast_e=500.0, bead_diameter_mm=3.0, integration_time_s=0.01
    )
    double = predict(
        signal_e=5000.0, contrast_e=1000.0, bead_diameter_mm=3.0, integration_time_s=0.01
    )

    assert double.cnr == pytest.approx(2.0 * single.cnr, rel=1e-9)


def test_rose_index_is_cnr_times_root_bead_area() -> None:
    result = predict(
        signal_e=5000.0, contrast_e=500.0, bead_diameter_mm=3.0, integration_time_s=0.01
    )
    assert result.rose_index == pytest.approx(result.cnr * math.sqrt(result.bead_area_px), rel=1e-9)


def test_bead_area_matches_a_disk_of_the_stated_diameter() -> None:
    result = predict(signal_e=1.0, contrast_e=1.0, bead_diameter_mm=3.0, integration_time_s=0.0)
    diameter_px = 3000.0 / WHEELCAM.pixel_scale_um()

    assert result.bead_diameter_px == pytest.approx(diameter_px, rel=1e-9)
    assert result.bead_area_px == pytest.approx(math.pi / 4 * diameter_px**2, rel=1e-9)


def test_a_bright_high_contrast_bead_is_declared_detectable() -> None:
    result = predict(
        signal_e=20000.0, contrast_e=6000.0, bead_diameter_mm=3.0, integration_time_s=0.01
    )
    assert result.verdict == "detectable"


def test_a_bead_smaller_than_a_pixel_is_not_detectable() -> None:
    result = predict(signal_e=200.0, contrast_e=8.0, bead_diameter_mm=0.05, integration_time_s=0.01)
    assert result.verdict == "not detectable"


def test_negative_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        predict(signal_e=-1.0, contrast_e=1.0, bead_diameter_mm=1.0, integration_time_s=0.0)


def test_electrons_from_testbed_scales_both_terms() -> None:
    signal, contrast = electrons_from_testbed(
        1000.0, 200.0, gain_e_per_dn=4.0, illumination_ratio=0.5
    )

    assert signal == pytest.approx(2000.0)
    assert contrast == pytest.approx(400.0)


@pytest.mark.parametrize(("gain", "ratio"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_electrons_from_testbed_rejects_impossible_scaling(gain: float, ratio: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        electrons_from_testbed(1000.0, 200.0, gain, ratio)


def test_resampling_shrinks_a_frame_taken_at_a_finer_scale() -> None:
    """A testbed at 20 um per pixel is five times finer than the flight camera."""
    rng = np.random.default_rng(0)
    data = rng.normal(1000.0, 10.0, size=(200, 200))
    resampled = resample_to_flight_scale(data, testbed_pixel_scale_um=20.0)

    expected = 200 * 20.0 / WHEELCAM.pixel_scale_um()
    assert resampled.shape[0] == pytest.approx(expected, abs=2)


def test_resampling_rejects_a_nonsense_pixel_scale() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        resample_to_flight_scale(np.zeros((10, 10)), testbed_pixel_scale_um=0.0)


def test_quantising_to_ten_bits_leaves_1024_levels() -> None:
    data = np.linspace(0.0, 65535.0, 4096).reshape(64, 64)
    reduced = quantise(data, full_scale=65535.0)

    assert len(np.unique(reduced)) <= 2**WHEELCAM.adc_bits
    assert reduced.max() == pytest.approx(65535.0, rel=1e-6)


def test_quantising_destroys_a_low_contrast_bead() -> None:
    """The check worth running before trusting a faint feature: 10 bits is coarse.

    One 10-bit step is 65535/1023 = 64 DN. A 20 DN bead cannot survive it: depending on
    where it falls it either vanishes or gets stretched to a whole step.
    """
    data = np.full((32, 32), 30000.0)
    data[10:20, 10:20] += 20.0
    reduced = quantise(data, full_scale=65535.0)
    surviving_contrast = float(reduced[10:20, 10:20].mean() - reduced[0, 0])

    assert len(np.unique(reduced)) <= 2
    assert abs(surviving_contrast - 20.0) > 10.0
