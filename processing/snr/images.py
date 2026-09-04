"""Load testbed images and flag the conditions that invalidate a noise estimate.

Every estimator in this package assumes the pixel values are a linear function of the
number of photons that hit the sensor. Clipped, gamma-corrected or lossily compressed
frames break that assumption silently, so loading is also where we check for it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import tifffile

Array = npt.NDArray[np.float64]

# Fractions of the frame above which a problem is worth reporting to the user.
SATURATION_WARN_FRACTION = 1e-3
FLOOR_WARN_FRACTION = 1e-3

# Extensions we know carry linear, unquantised-by-a-codec data.
LOSSLESS_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".npy"})
LOSSY_SUFFIXES = frozenset({".jpg", ".jpeg", ".webp"})


@dataclass(frozen=True)
class Frame:
    """One frame, held as float64 digital numbers (DN) on its original scale.

    We deliberately do not normalise to 0..1: noise levels are reported in DN so they
    can be compared against the camera's read noise and gain.
    """

    name: str
    data: Array
    full_scale: float
    source_dtype: str

    @property
    def shape(self) -> tuple[int, int]:
        height, width = self.data.shape
        return int(height), int(width)

    @property
    def saturated_fraction(self) -> float:
        return float(np.mean(self.data >= self.full_scale))

    @property
    def floor_fraction(self) -> float:
        return float(np.mean(self.data <= 0.0))

    def quality_warnings(self) -> list[str]:
        """Conditions that make the noise estimate untrustworthy, in plain words."""
        notes: list[str] = []
        if self.saturated_fraction > SATURATION_WARN_FRACTION:
            notes.append(
                f"{self.saturated_fraction:.2%} of pixels are at full scale; clipped pixels "
                "have no noise, so sigma is under-estimated"
            )
        if self.floor_fraction > FLOOR_WARN_FRACTION:
            notes.append(
                f"{self.floor_fraction:.2%} of pixels are at zero; a clipped black level "
                "also hides noise"
            )
        if self.data.max() > 0 and self.data.max() <= self.full_scale / 8:
            notes.append(
                f"peak value {self.data.max():.0f} is far below the assumed full scale "
                f"{self.full_scale:.0f}; if the ADC is 10- or 12-bit inside a 16-bit file, "
                "pass the real full scale or saturation will never be detected"
            )
        return notes


def _full_scale_for(dtype: np.dtype[Any]) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype.type).max)
    # Float files carry no declared range. 1.0 is the usual convention for normalised
    # exports; the caller can override when it is not.
    return 1.0


def _to_single_channel(raw: npt.NDArray[np.generic], channel: int | None, name: str) -> Array:
    if raw.ndim == 2:
        if channel is not None:
            raise ValueError(f"{name} is single-channel but channel={channel} was requested")
        return raw.astype(np.float64)
    if raw.ndim == 3:
        if channel is None:
            raise ValueError(
                f"{name} has {raw.shape[2]} channels. Pick one with channel=... . We do not "
                "convert to luma automatically because the weighted sum changes the noise "
                "statistics the estimators are trying to measure."
            )
        return raw[:, :, channel].astype(np.float64)
    raise ValueError(f"{name} has {raw.ndim} dimensions, expected 2 or 3")


def load_image(
    path: str | Path,
    channel: int | None = None,
    full_scale: float | None = None,
) -> Frame:
    """Read a frame from disk without rescaling it.

    `full_scale` is the DN value that means "saturated". It defaults to the maximum of
    the file's integer type, which is wrong for a 12-bit sensor written into a 16-bit
    TIFF, so pass it explicitly when you know it.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        raw = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        raw = np.asarray(tifffile.imread(path))
    else:
        raw = np.asarray(iio.imread(path))

    data = _to_single_channel(raw, channel, path.name)
    scale = full_scale if full_scale is not None else _full_scale_for(raw.dtype)
    return Frame(
        name=path.name,
        data=data,
        full_scale=float(scale),
        source_dtype=str(raw.dtype),
    )


def load_raw(
    path: str | Path,
    shape: tuple[int, int],
    dtype: str = "uint16",
    full_scale: float | None = None,
) -> Frame:
    """Read a headerless binary dump, which is what most frame grabbers write.

    You have to supply the shape and dtype because the file does not record them. This
    does not read camera raw formats such as DNG or CR2; convert those first.
    """
    path = Path(path)
    np_dtype = np.dtype(dtype)
    raw = np.fromfile(path, dtype=np_dtype)
    expected = shape[0] * shape[1]
    if raw.size != expected:
        raise ValueError(
            f"{path.name} holds {raw.size} samples but shape {shape} needs {expected}; "
            "check the dtype and whether the file has a header"
        )
    scale = full_scale if full_scale is not None else _full_scale_for(np_dtype)
    return Frame(
        name=path.name,
        data=raw.reshape(shape).astype(np.float64),
        full_scale=float(scale),
        source_dtype=str(np_dtype),
    )


def check_suffix(path: str | Path) -> list[str]:
    """Warn about file formats that destroy the noise we are trying to measure."""
    suffix = Path(path).suffix.lower()
    if suffix in LOSSY_SUFFIXES:
        return [
            f"{suffix} is lossy: the codec smooths flat areas and adds block edges, which "
            "makes every blind noise estimate meaningless. Re-export as TIFF."
        ]
    if suffix not in LOSSLESS_SUFFIXES:
        return [f"unrecognised extension {suffix}; assuming it decodes to linear data"]
    return []
