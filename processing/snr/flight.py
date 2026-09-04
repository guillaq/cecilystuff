"""Translate a testbed measurement into what the IDEFIX WheelCams would see on Phobos.

Two separate things live here and it matters that they stay separate.

`predict` is arithmetic on a published noise model. Given how many electrons a bead and
its background produce, it says what contrast-to-noise the flight camera would deliver.
Nothing is invented: shot, dark and read noise add in quadrature, and the bead footprint
comes from the camera geometry.

`resample_to_flight_scale` is a picture, not a measurement. It resamples a testbed frame
to the flight pixel scale and applies the blur implied by the published MTF, so a human
can look at a bead the size the flight camera would render it. It deliberately does not
synthesise flight noise, because the frame already carries the testbed camera's own noise
and adding more would double count it.

WheelCam numbers are from Murdoch et al., "The WheelCams on the IDEFIX rover", Progress
in Earth and Planetary Science, 2025.
https://link.springer.com/article/10.1186/s40645-025-00725-3
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from processing.snr.metrics import rose_verdict

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class WheelCam:
    """Published IDEFIX WheelCam parameters.

    Every field except `adc_bits` is quoted directly from the instrument paper.
    `adc_bits` is inferred: the paper gives 41.9 Mbit per uncompressed unbinned image
    over a 2048 x 2048 array, which is 9.99 bits per pixel, so 10.
    """

    array_px: int = 2048
    pixel_pitch_um: float = 5.5
    focal_length_mm: float = 18.0
    field_of_view_deg: float = 32.5
    best_focus_cm: float = 30.35
    read_noise_e: float = 13.9
    dark_current_e_per_s: float = 560.0
    adc_bits: int = 10
    mtf_at_nyquist: float = 0.2

    def pixel_scale_um(self, distance_cm: float | None = None) -> float:
        """Ground sampling distance at a given range, from similar triangles.

        The paper quotes 100 um at 30 cm. Plain geometry gives 91.7 um, the difference
        being the tilt of the focal plane, so treat this as the flat-field approximation.
        """
        distance = self.best_focus_cm if distance_cm is None else distance_cm
        return self.pixel_pitch_um * (distance * 10.0) / self.focal_length_mm


WHEELCAM = WheelCam()


@dataclass(frozen=True)
class FlightPrediction:
    """Predicted per-pixel and per-bead performance under flight conditions."""

    signal_e: float
    contrast_e: float
    shot_noise_e: float
    dark_noise_e: float
    read_noise_e: float
    total_noise_e: float
    pixel_scale_um: float
    bead_diameter_px: float
    bead_area_px: float
    cnr: float
    rose_index: float
    verdict: str

    def dominant_noise(self) -> str:
        """Which noise term to attack first if the prediction is not good enough."""
        terms = {
            "photon shot noise (add light or exposure)": self.shot_noise_e,
            "dark current (shorten the exposure or cool the sensor)": self.dark_noise_e,
            "read noise (fixed by the sensor)": self.read_noise_e,
        }
        return max(terms, key=lambda key: terms[key])


def discrete_mtf_at_nyquist(sigma_px: float) -> float:
    """Contrast left at Nyquist by the sampled Gaussian kernel scipy actually applies.

    A signal at Nyquist alternates sign from one sample to the next, so a symmetric
    kernel w passes sum(w[k] * (-1)^k) / sum(w[k]) of it. The kernel is built the same
    way `scipy.ndimage.gaussian_filter` builds it, truncated at four standard deviations.
    """
    if sigma_px <= 0:
        return 1.0
    radius = max(int(4.0 * sigma_px + 0.5), 1)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-(offsets.astype(np.float64) ** 2) / (2.0 * sigma_px**2))
    return float(np.sum(weights * (-1.0) ** offsets) / np.sum(weights))


def mtf_blur_sigma_px(mtf_at_nyquist: float) -> float:
    """Gaussian blur width that leaves a given contrast at the Nyquist frequency.

    The textbook answer is sqrt(-2 ln m / pi^2), from the continuous Gaussian whose MTF
    is exp(-2 pi^2 sigma^2 f^2). That formula is wrong for what we are doing here. The
    blur we need is under one pixel wide, and a sub-pixel Gaussian sampled onto a pixel
    grid keeps far more contrast at Nyquist than the continuous one does: at a target of
    0.2 the continuous formula lands near 0.4, a factor of two out. So we solve for the
    width against the discrete kernel that will really be applied.

    A real lens is not Gaussian, so this reproduces the contrast loss at the finest scale
    and not the exact shape of the blur.
    """
    if not 0.0 < mtf_at_nyquist < 1.0:
        raise ValueError(f"MTF at Nyquist must be in (0, 1), got {mtf_at_nyquist}")

    # The MTF falls monotonically as the blur widens, so plain bisection is enough and is
    # easier to check by eye than pulling in a root finder.
    low, high = 1e-3, 20.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if discrete_mtf_at_nyquist(middle) > mtf_at_nyquist:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def predict(
    signal_e: float,
    contrast_e: float,
    bead_diameter_mm: float,
    integration_time_s: float,
    camera: WheelCam = WHEELCAM,
    distance_cm: float | None = None,
) -> FlightPrediction:
    """Predicted contrast-to-noise for beads seen by the flight camera.

    `signal_e` and `contrast_e` are per-pixel electron counts under flight illumination.
    They do not come out of a testbed image on their own: converting requires the sensor
    gain, from `camera.photon_transfer`, and a radiometric budget relating testbed
    lighting to the LED illumination on Phobos. `electrons_from_testbed` does the
    arithmetic once you have both.
    """
    if signal_e < 0 or integration_time_s < 0:
        raise ValueError("signal and integration time cannot be negative")

    dark_e = camera.dark_current_e_per_s * integration_time_s
    shot = math.sqrt(signal_e)
    dark_shot = math.sqrt(dark_e)
    total = math.sqrt(signal_e + dark_e + camera.read_noise_e**2)

    scale_um = camera.pixel_scale_um(distance_cm)
    diameter_px = bead_diameter_mm * 1000.0 / scale_um
    area_px = math.pi / 4.0 * diameter_px**2

    cnr = contrast_e / total if total > 0 else float("inf")
    rose = cnr * math.sqrt(area_px)

    return FlightPrediction(
        signal_e=signal_e,
        contrast_e=contrast_e,
        shot_noise_e=shot,
        dark_noise_e=dark_shot,
        read_noise_e=camera.read_noise_e,
        total_noise_e=total,
        pixel_scale_um=scale_um,
        bead_diameter_px=diameter_px,
        bead_area_px=area_px,
        cnr=cnr,
        rose_index=rose,
        verdict=rose_verdict(rose),
    )


def electrons_from_testbed(
    signal_dn: float,
    contrast_dn: float,
    gain_e_per_dn: float,
    illumination_ratio: float = 1.0,
) -> tuple[float, float]:
    """Convert measured DN to the electron counts the flight camera would collect.

    `gain_e_per_dn` is the testbed camera's system gain, which you get from a photon
    transfer series. `illumination_ratio` is flight electrons per testbed electron: it
    covers the LED output, the exposure time, the lens aperture and the surface albedo,
    and it has to come from a radiometric budget. Leaving it at 1.0 answers the narrower
    question "what if the flight camera saw exactly this scene at this brightness".
    """
    if gain_e_per_dn <= 0:
        raise ValueError("gain must be positive")
    if illumination_ratio <= 0:
        raise ValueError("illumination ratio must be positive")
    return (
        signal_dn * gain_e_per_dn * illumination_ratio,
        contrast_dn * gain_e_per_dn * illumination_ratio,
    )


def resample_to_flight_scale(
    data: Array,
    testbed_pixel_scale_um: float,
    camera: WheelCam = WHEELCAM,
    distance_cm: float | None = None,
) -> Array:
    """Show a testbed frame at the flight pixel scale, with the flight MTF applied.

    For looking at, not for measuring. Both the resampling and the blur correlate
    neighbouring pixels, which breaks the independent-noise assumption every estimator in
    this package relies on. Running an estimator on the output will report a noise level
    that is too low.
    """
    if testbed_pixel_scale_um <= 0:
        raise ValueError("testbed pixel scale must be positive")

    zoom = testbed_pixel_scale_um / camera.pixel_scale_um(distance_cm)
    # Blur before downsampling, the way the optics do, so we are not aliasing detail that
    # the flight camera would never have resolved in the first place.
    sigma_in_testbed_px = mtf_blur_sigma_px(camera.mtf_at_nyquist) / zoom if zoom > 0 else 0.0
    blurred = np.asarray(ndimage.gaussian_filter(data, sigma=sigma_in_testbed_px), dtype=np.float64)
    resampled = ndimage.zoom(blurred, zoom, order=1)
    return np.asarray(resampled, dtype=np.float64)


def quantise(data: Array, full_scale: float, camera: WheelCam = WHEELCAM) -> Array:
    """Requantise to the flight ADC depth, to see whether 10 bits loses the contrast.

    Worth checking when the testbed camera is 16-bit and the beads are low contrast: a
    difference of a few DN out of 65535 can disappear entirely at 10 bits.
    """
    if full_scale <= 0:
        raise ValueError("full scale must be positive")
    levels = 2**camera.adc_bits - 1
    normalised = np.clip(data / full_scale, 0.0, 1.0)
    return np.round(normalised * levels) / levels * full_scale
