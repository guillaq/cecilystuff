"""Loading has to be boring and honest: no rescaling, and loud about broken inputs."""

import numpy as np
import pytest
import tifffile

from processing.snr.images import Frame, check_suffix, load_image, load_raw


def write_tiff(tmp_path, data: np.ndarray, name: str = "frame.tif"):
    path = tmp_path / name
    tifffile.imwrite(path, data)
    return path


def test_pixel_values_are_not_rescaled(tmp_path) -> None:
    """Noise is reported in DN, so a loader that normalises would silently break it."""
    data = np.array([[0, 1000], [30000, 65535]], dtype=np.uint16)
    image = load_image(write_tiff(tmp_path, data))

    assert image.data.dtype == np.float64
    np.testing.assert_array_equal(image.data, data.astype(np.float64))
    assert image.full_scale == 65535.0


def test_full_scale_can_be_overridden_for_a_12_bit_sensor(tmp_path) -> None:
    data = np.full((16, 16), 4095, dtype=np.uint16)
    image = load_image(write_tiff(tmp_path, data), full_scale=4095.0)

    assert image.saturated_fraction == 1.0


def test_saturation_is_reported(tmp_path) -> None:
    data = np.full((100, 100), 1000, dtype=np.uint16)
    data[:5, :] = 65535
    warnings = load_image(write_tiff(tmp_path, data)).quality_warnings()

    assert any("full scale" in note for note in warnings)


def test_low_peak_value_hints_at_the_wrong_full_scale(tmp_path) -> None:
    data = np.full((32, 32), 2000, dtype=np.uint16)
    warnings = load_image(write_tiff(tmp_path, data)).quality_warnings()

    assert any("far below the assumed full scale" in note for note in warnings)


def test_a_clean_frame_produces_no_warnings(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(30000, 200, size=(64, 64)).astype(np.uint16)

    assert load_image(write_tiff(tmp_path, data)).quality_warnings() == []


def test_colour_images_must_name_a_channel(tmp_path) -> None:
    """Converting to luma would change the noise, so we refuse to guess."""
    data = np.zeros((8, 8, 3), dtype=np.uint16)
    path = write_tiff(tmp_path, data, "colour.tif")

    with pytest.raises(ValueError, match="Pick one with channel"):
        load_image(path)
    assert load_image(path, channel=1).shape == (8, 8)


def test_raw_files_check_the_declared_shape(tmp_path) -> None:
    path = tmp_path / "frame.raw"
    np.arange(64, dtype=np.uint16).tofile(path)

    assert load_raw(path, (8, 8)).data.sum() == float(np.arange(64).sum())
    with pytest.raises(ValueError, match="holds 64 samples"):
        load_raw(path, (16, 16))


def test_lossy_formats_are_called_out() -> None:
    assert check_suffix("run/frame.jpg") != []
    assert check_suffix("run/frame.tif") == []


def test_floor_fraction_flags_a_clipped_black_level() -> None:
    data = np.zeros((10, 10), dtype=np.float64)
    data[5:, :] = 500.0
    image = Frame(name="x", data=data, full_scale=65535.0, source_dtype="float64")

    assert image.floor_fraction == pytest.approx(0.5)
    assert any("at zero" in note for note in image.quality_warnings())
