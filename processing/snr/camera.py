"""Camera characterisation from a photon transfer series, following EMVA 1288.

This is the only module here that measures noise rather than inferring it, and it is the
only one that gives you numbers in electrons instead of digital numbers. In exchange it
needs data the testbed run images cannot provide.

What you have to capture, once, with the lens capped or a uniform light source:

    1. Two dark frames, same exposure as the series below, lens capped.
    2. Two frames at each of about 10 to 15 illumination levels, evenly spaced from
       nearly dark to just past saturation. Keep the light and the exposure stable inside
       each pair; the pair is what separates temporal noise from fixed pattern noise.
    3. A flat, defocused, uniformly lit field. Non-uniformity biases the spatial terms.

Taking two frames per level and using their difference is what removes fixed pattern
noise from the temporal variance, and is the method the standard prescribes.

Reference: EMVA Standard 1288, "Standard for Characterization of Image Sensors and
Cameras", release 3.0, https://www.emva.org/wp-content/uploads/EMVA1288-3.0.pdf
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

# EMVA 1288 fits the photon transfer curve only up to 70% of saturation, because sensors
# stop being linear before they visibly clip.
LINEAR_RANGE_FRACTION = 0.7


@dataclass(frozen=True)
class FramePair:
    """Two frames taken back to back under identical conditions."""

    frame_a: Array
    frame_b: Array

    def mean_dn(self) -> float:
        return float((np.mean(self.frame_a) + np.mean(self.frame_b)) / 2.0)

    def temporal_variance_dn2(self) -> float:
        """Temporal variance of a single frame, with fixed pattern noise removed.

        var(A - B) = 2 * var(single frame) when the two frames differ only by noise, so
        halving the difference variance leaves the per-frame temporal variance. Anything
        fixed in the sensor cancels in the subtraction.
        """
        return float(np.var(self.frame_a - self.frame_b, ddof=1) / 2.0)

    def spatial_variance_dn2(self) -> float:
        """Variance across pixels of the averaged pair, corrected for temporal noise.

        Averaging two frames leaves half the temporal variance in the mean image, which
        would otherwise be counted as fixed pattern non-uniformity.
        """
        mean_image = (self.frame_a + self.frame_b) / 2.0
        raw = float(np.var(mean_image, ddof=1))
        return max(raw - self.temporal_variance_dn2() / 2.0, 0.0)


@dataclass(frozen=True)
class PhotonTransfer:
    """Sensor parameters recovered from the photon transfer curve."""

    gain_dn_per_e: float
    system_gain_e_per_dn: float
    read_noise_e: float
    read_noise_dn: float
    read_noise_fit_dn: float
    dark_offset_dn: float
    saturation_dn: float
    saturation_capacity_e: float
    dynamic_range_db: float
    max_snr: float
    max_snr_db: float
    dsnu_e: float
    prnu_percent: float
    r_squared: float
    points_used: int

    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.points_used < 5:
            notes.append(
                f"only {self.points_used} exposure levels fell in the linear range; the "
                "standard expects at least 10 to 15 across the full range"
            )
        if self.r_squared < 0.99:
            notes.append(
                f"the photon transfer fit has r^2 = {self.r_squared:.4f}. Below about 0.99 "
                "the sensor is not behaving linearly and the gain is not trustworthy"
            )
        # Two independent routes to the same quantity: the dark frames, which is what we
        # report, and the intercept of the fit. They should agree. When they do not, the
        # series has a problem the gain alone will not reveal. Expect the intercept to be
        # the noisier of the two: it is an extrapolation to zero signal from data that
        # mostly sits at high signal, so it inherits every bit of curvature in the curve.
        if self.read_noise_dn > 0 and self.read_noise_fit_dn > 0:
            ratio = self.read_noise_fit_dn / self.read_noise_dn
            if not 0.5 <= ratio <= 2.0:
                notes.append(
                    f"read noise from the dark frames ({self.read_noise_dn:.2f} DN) and from "
                    f"the fit intercept ({self.read_noise_fit_dn:.2f} DN) differ by a factor "
                    f"of {max(ratio, 1 / ratio):.1f}. Usually the illumination drifted between "
                    "the two frames of a pair, or the dark frames used a different exposure"
                )
        return notes


def _fit_line(x: Array, y: Array) -> tuple[float, float, float]:
    """Least squares slope, intercept and r^2. Written out so the r^2 is auditable."""
    if x.size < 2:
        raise ValueError("need at least two points to fit the photon transfer curve")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return float(slope), float(intercept), r_squared


def photon_transfer(
    dark: FramePair,
    flats: list[FramePair],
    saturation_dn: float,
    linear_range_fraction: float = LINEAR_RANGE_FRACTION,
) -> PhotonTransfer:
    """Fit noise variance against mean signal to recover gain, read noise and range.

    The physics: each pixel counts photons, and photon counts are Poisson, so their
    variance equals their mean in electrons. Converting to digital numbers multiplies the
    mean by the gain K and the variance by K squared, which leaves

        variance_in_DN = K * mean_in_DN + read_noise_variance

    a straight line whose slope is the gain and whose intercept is everything that does
    not scale with light.
    """
    if not flats:
        raise ValueError("photon_transfer needs at least one illuminated frame pair")

    dark_offset = dark.mean_dn()
    dark_temporal_var = dark.temporal_variance_dn2()

    means = np.array([pair.mean_dn() - dark_offset for pair in flats], dtype=np.float64)
    variances = np.array([pair.temporal_variance_dn2() for pair in flats], dtype=np.float64)

    limit = linear_range_fraction * (saturation_dn - dark_offset)
    usable = (means > 0) & (means <= limit)
    if usable.sum() < 2:
        raise ValueError(
            f"only {int(usable.sum())} exposure levels sit between the dark offset and "
            f"{linear_range_fraction:.0%} of saturation; the series does not span the "
            "linear range"
        )

    gain, intercept, r_squared = _fit_line(means[usable], variances[usable])
    if gain <= 0:
        raise ValueError(
            "the fitted gain is not positive, so noise does not grow with signal. Check "
            "that the frames are linear and that the pairs are not identical files"
        )

    # EMVA 1288 takes the temporal dark noise straight from the dark frame pair. The fit
    # intercept measures the same thing but extrapolates to zero signal from data that is
    # dominated by the bright end, so it is kept only as a consistency check.
    read_noise_dn = float(np.sqrt(dark_temporal_var))
    read_noise_e = read_noise_dn / gain
    read_noise_fit_dn = float(np.sqrt(max(intercept, 0.0)))
    saturation_capacity_e = (saturation_dn - dark_offset) / gain

    # Spatial non-uniformity. DSNU is the dark fixed pattern; PRNU is how much the
    # response varies between pixels, taken at the brightest usable level.
    dsnu_e = float(np.sqrt(dark.spatial_variance_dn2())) / gain
    brightest = int(np.flatnonzero(usable)[int(np.argmax(means[usable]))])
    bright_pair = flats[brightest]
    prnu_var = max(bright_pair.spatial_variance_dn2() - dark.spatial_variance_dn2(), 0.0)
    prnu_percent = 100.0 * float(np.sqrt(prnu_var)) / means[brightest]

    max_snr = float(np.sqrt(saturation_capacity_e)) if saturation_capacity_e > 0 else float("nan")
    dynamic_range = saturation_capacity_e / read_noise_e if read_noise_e > 0 else float("inf")

    return PhotonTransfer(
        gain_dn_per_e=gain,
        system_gain_e_per_dn=1.0 / gain,
        read_noise_e=read_noise_e,
        read_noise_dn=read_noise_dn,
        read_noise_fit_dn=read_noise_fit_dn,
        dark_offset_dn=dark_offset,
        saturation_dn=saturation_dn,
        saturation_capacity_e=saturation_capacity_e,
        dynamic_range_db=20.0 * float(np.log10(dynamic_range))
        if dynamic_range > 0
        else float("nan"),
        max_snr=max_snr,
        max_snr_db=20.0 * float(np.log10(max_snr)) if max_snr > 0 else float("nan"),
        dsnu_e=dsnu_e,
        prnu_percent=prnu_percent,
        r_squared=r_squared,
        points_used=int(usable.sum()),
    )


def theoretical_snr(signal_e: Array | float, read_noise_e: float, dark_e: float = 0.0) -> Array:
    """SNR of an ideal sensor at a given signal level, in electrons.

    Shot noise, dark shot noise and read noise add in quadrature. Plotting a measured SNR
    against this curve is the quickest way to see whether a camera is behaving.
    """
    signal = np.asarray(signal_e, dtype=np.float64)
    noise = np.sqrt(signal + dark_e + read_noise_e**2)
    return np.asarray(np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0))
