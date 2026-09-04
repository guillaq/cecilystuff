"""Validate the photon transfer code against a simulated sensor we fully control.

This is the strongest test in the package. We build frames from a sensor whose gain,
read noise and offset we chose, then check that the EMVA 1288 fit recovers them. If the
implementation had the gain upside down or the factor of two wrong, this would catch it.
"""

import numpy as np
import pytest

from processing.snr.camera import FramePair, photon_transfer, theoretical_snr

TRUE_GAIN_DN_PER_E = 0.5
TRUE_READ_NOISE_E = 10.0
TRUE_OFFSET_DN = 100.0
SHAPE = (128, 128)
LEVELS_E = (50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 60000.0)


def simulate(
    mean_electrons: float, seed: int, fixed_pattern: np.ndarray | None = None
) -> np.ndarray:
    """One frame from a sensor with known gain, read noise and offset."""
    rng = np.random.default_rng(seed)
    signal = rng.poisson(mean_electrons, size=SHAPE).astype(np.float64)
    if fixed_pattern is not None:
        signal = signal * fixed_pattern
    read = rng.normal(0.0, TRUE_READ_NOISE_E, size=SHAPE)
    return TRUE_OFFSET_DN + TRUE_GAIN_DN_PER_E * (signal + read)


def build_series(seed: int = 0, fixed_pattern: np.ndarray | None = None):
    dark = FramePair(frame_a=simulate(0.0, seed + 1), frame_b=simulate(0.0, seed + 2))
    flats = [
        FramePair(
            frame_a=simulate(level, seed + 10 + index * 2, fixed_pattern),
            frame_b=simulate(level, seed + 11 + index * 2, fixed_pattern),
        )
        for index, level in enumerate(LEVELS_E)
    ]
    return dark, flats


def test_recovers_the_gain_we_simulated() -> None:
    result = photon_transfer(*build_series(), saturation_dn=65535.0)

    assert result.gain_dn_per_e == pytest.approx(TRUE_GAIN_DN_PER_E, rel=0.03)
    assert result.system_gain_e_per_dn == pytest.approx(1.0 / TRUE_GAIN_DN_PER_E, rel=0.03)


def test_recovers_the_read_noise_from_the_dark_frames() -> None:
    """The direct route: var(darkA - darkB) / 2, which is independent of the fit."""
    result = photon_transfer(*build_series(), saturation_dn=65535.0)

    assert result.read_noise_dn == pytest.approx(TRUE_GAIN_DN_PER_E * TRUE_READ_NOISE_E, rel=0.05)
    assert result.read_noise_e == pytest.approx(TRUE_READ_NOISE_E, rel=0.05)


def test_the_fit_intercept_agrees_with_the_dark_frames_to_within_a_factor() -> None:
    """The cross-check passes, but only loosely, which is why it is not the primary route.

    Extrapolating the photon transfer line back to zero signal is a poor way to measure a
    small intercept: the fit is dominated by the bright end. On this simulated sensor the
    two routes land within about 30% of each other, and that is the good case.
    """
    result = photon_transfer(*build_series(), saturation_dn=65535.0)

    assert result.read_noise_fit_dn == pytest.approx(result.read_noise_dn, rel=0.5)
    assert result.warnings() == []


def test_recovers_the_dark_offset() -> None:
    result = photon_transfer(*build_series(), saturation_dn=65535.0)
    assert result.dark_offset_dn == pytest.approx(TRUE_OFFSET_DN, rel=0.02)


def test_the_fit_is_linear_for_a_linear_sensor() -> None:
    result = photon_transfer(*build_series(), saturation_dn=65535.0)
    assert result.r_squared > 0.999


def test_saturation_capacity_and_dynamic_range_follow_the_gain() -> None:
    result = photon_transfer(*build_series(), saturation_dn=65535.0)
    expected_capacity = (65535.0 - TRUE_OFFSET_DN) / TRUE_GAIN_DN_PER_E

    assert result.saturation_capacity_e == pytest.approx(expected_capacity, rel=0.05)
    assert result.max_snr == pytest.approx(np.sqrt(expected_capacity), rel=0.05)


def test_a_uniform_sensor_has_almost_no_fixed_pattern_noise() -> None:
    result = photon_transfer(*build_series(), saturation_dn=65535.0)

    assert result.dsnu_e < 2.0
    assert result.prnu_percent < 1.0


def test_prnu_picks_up_a_pixel_gain_variation() -> None:
    """Give 5% of pixel-to-pixel response variation and PRNU should report about 5%."""
    rng = np.random.default_rng(99)
    pattern = rng.normal(1.0, 0.05, size=SHAPE)
    result = photon_transfer(*build_series(fixed_pattern=pattern), saturation_dn=65535.0)

    assert result.prnu_percent == pytest.approx(5.0, rel=0.25)


def compress(frame: np.ndarray, strength: float) -> np.ndarray:
    """Gentle saturating non-linearity, the kind a real sensor shows near full well."""
    above = np.clip(frame - TRUE_OFFSET_DN, 0.0, None)
    span = 30000.0
    return TRUE_OFFSET_DN + above * (1.0 - strength * above / span)


def test_a_non_linear_sensor_is_reported_rather_than_fitted_quietly() -> None:
    """A sensor that compresses by 20% at the top of its range must not pass silently."""
    dark, flats = build_series()
    squashed = [
        FramePair(frame_a=compress(pair.frame_a, 0.2), frame_b=compress(pair.frame_b, 0.2))
        for pair in flats
    ]
    result = photon_transfer(dark, squashed, saturation_dn=65535.0)

    assert result.r_squared < 0.99
    assert any("linearly" in note for note in result.warnings())


def test_gamma_corrected_frames_fail_loudly() -> None:
    """Square-root compression makes noise shrink as signal grows, which is impossible.

    This is what a frame that has had a display gamma applied looks like, and it is the
    single most likely way for someone to feed the wrong files in.
    """
    dark, flats = build_series()
    gamma = [
        FramePair(
            frame_a=TRUE_OFFSET_DN + 40.0 * np.sqrt(np.clip(p.frame_a - TRUE_OFFSET_DN, 0, None)),
            frame_b=TRUE_OFFSET_DN + 40.0 * np.sqrt(np.clip(p.frame_b - TRUE_OFFSET_DN, 0, None)),
        )
        for p in flats
    ]
    with pytest.raises(ValueError, match="not positive"):
        photon_transfer(dark, gamma, saturation_dn=65535.0)


def test_a_series_that_does_not_span_the_range_is_rejected() -> None:
    dark, flats = build_series()
    with pytest.raises(ValueError, match="does not span"):
        photon_transfer(dark, flats[:1], saturation_dn=65535.0)


def test_no_flats_is_an_error() -> None:
    dark, _ = build_series()
    with pytest.raises(ValueError, match="at least one"):
        photon_transfer(dark, [], saturation_dn=65535.0)


def test_theoretical_snr_is_shot_limited_when_bright_and_read_limited_when_dark() -> None:
    bright = float(theoretical_snr(1_000_000.0, read_noise_e=10.0))
    dark = float(theoretical_snr(1.0, read_noise_e=10.0))

    assert bright == pytest.approx(np.sqrt(1_000_000.0), rel=0.01)
    assert dark == pytest.approx(1.0 / 10.0, rel=0.05)
