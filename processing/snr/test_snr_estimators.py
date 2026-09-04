"""Each estimator has to recover a noise level we chose, on data with no structure.

These are unit tests, deliberately small and fast. The real validation is in
test_snr_benchmark.py and in the full benchmark run, which use realistic scenes.
"""

import numpy as np
import pytest

from processing.snr import estimators

SHAPE = (256, 256)
TOLERANCE = 0.05  # 256x256 gives a sampling error well under 1%, so 5% is generous.


def pure_noise(sigma: float, seed: int = 0, offset: float = 1000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return offset + rng.normal(0.0, sigma, size=SHAPE)


@pytest.mark.parametrize("name", sorted(estimators.ESTIMATORS))
@pytest.mark.parametrize("sigma", [1.0, 10.0, 100.0])
def test_recovers_known_sigma_on_flat_noise(name: str, sigma: float) -> None:
    estimate = estimators.ESTIMATORS[name].func(pure_noise(sigma), None)
    assert estimate == pytest.approx(sigma, rel=TOLERANCE)


@pytest.mark.parametrize("name", sorted(estimators.ESTIMATORS))
def test_scales_linearly_with_sigma(name: str) -> None:
    """Doubling the noise has to double the estimate, whatever the absolute accuracy."""
    func = estimators.ESTIMATORS[name].func
    single = func(pure_noise(4.0, seed=1), None)
    double = func(pure_noise(8.0, seed=1), None)
    assert double / single == pytest.approx(2.0, rel=TOLERANCE)


@pytest.mark.parametrize("name", sorted(estimators.ESTIMATORS))
def test_offset_does_not_change_the_estimate(name: str) -> None:
    """A black level shifts every pixel equally and must not look like noise."""
    func = estimators.ESTIMATORS[name].func
    base = pure_noise(5.0, seed=2, offset=0.0)
    assert func(base + 5000.0, None) == pytest.approx(func(base, None), rel=1e-9)


def ramp(total_dn: float) -> np.ndarray:
    """A left-to-right lighting gradient, the classic testbed illumination problem."""
    return np.linspace(0.0, total_dn, SHAPE[1])[np.newaxis, :] * np.ones((SHAPE[0], 1))


@pytest.mark.parametrize("name", ["immerkaer", "mad_haar"])
def test_laplacian_and_wavelet_are_immune_to_a_ramp(name: str) -> None:
    """Both operators cancel any locally linear brightness change, exactly.

    That is not an approximation: the Laplacian of a plane is zero, and so is the
    diagonal Haar detail of a plane. It is the reason these two are the safe default.
    """
    func = estimators.ESTIMATORS[name].func
    assert func(pure_noise(5.0, seed=3) + ramp(500.0), None) == pytest.approx(5.0, rel=TOLERANCE)


def test_block_percentile_is_biased_by_a_steep_ramp() -> None:
    """A known limit, measured rather than assumed.

    The block estimator has no notion of gradients: it reads whatever spread a block
    contains. A ramp of slope s adds s * sqrt((n^2 - 1) / 12) to every 8-pixel block, in
    quadrature with the noise. At 1.95 DN per pixel against a sigma of 5 that is a 30%
    over-estimate. Ten times gentler and it disappears. The practical rule: stay under
    about a third of sigma of gradient across one block, or flatten the illumination.
    """
    steep = estimators.block_percentile(pure_noise(5.0, seed=3) + ramp(500.0), None)
    gentle = estimators.block_percentile(pure_noise(5.0, seed=3) + ramp(50.0), None)

    assert steep > 1.2 * 5.0
    assert gentle == pytest.approx(5.0, rel=0.1)


def test_global_std_is_fooled_by_a_ramp() -> None:
    """The counter-example that justifies having the other four methods at all."""
    assert estimators.global_std(pure_noise(5.0, seed=3) + ramp(500.0)) > 100.0


@pytest.mark.parametrize("name", sorted(estimators.ESTIMATORS))
def test_mask_restricts_the_measurement(name: str) -> None:
    """Noise inside the mask is 3 DN; outside it is 300. The mask has to win."""
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 300.0, size=SHAPE)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[32:224, 32:224] = True
    data[mask] = rng.normal(0.0, 3.0, size=int(mask.sum()))

    assert estimators.ESTIMATORS[name].func(data, mask) == pytest.approx(3.0, rel=0.15)


def test_immerkaer_matches_its_published_formula() -> None:
    """Check the constant, not just the behaviour, on a hand-computed example."""
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 7.0, size=(64, 64))
    mask = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])

    total = 0.0
    for row in range(1, 63):
        for col in range(1, 63):
            total += abs(float(np.sum(data[row - 1 : row + 2, col - 1 : col + 2] * mask)))
    expected = np.sqrt(np.pi / 2) * total / (6 * 62 * 62)

    assert estimators.immerkaer(data) == pytest.approx(expected, rel=1e-9)


def test_block_percentile_bias_correction_matters() -> None:
    """Without the chi-quantile correction the 10th percentile sits well below sigma."""
    data = pure_noise(20.0, seed=6)
    corrected = estimators.make_block_percentile(block=8, percentile=10.0)(data, None)
    raw_stds = estimators._block_stds(data, None, 8)
    uncorrected = float(np.percentile(raw_stds, 10.0))

    assert corrected == pytest.approx(20.0, rel=TOLERANCE)
    assert uncorrected < 0.92 * 20.0
    assert abs(corrected - 20.0) < abs(uncorrected - 20.0)


def test_estimate_all_rejects_unknown_methods() -> None:
    with pytest.raises(KeyError, match="unknown estimator"):
        estimators.estimate_all(pure_noise(1.0), None, ["not_a_method"])


def test_estimate_all_returns_nan_rather_than_failing() -> None:
    """One unusable frame in a batch must not stop the batch."""
    tiny_mask = np.zeros(SHAPE, dtype=bool)
    tiny_mask[0, 0] = True
    results = estimators.estimate_all(pure_noise(5.0), tiny_mask)

    assert set(results) == set(estimators.ESTIMATORS)
    assert np.isnan(results["mad_haar"])


def test_registry_documents_every_method() -> None:
    """A method with no stated failure mode is a trap for whoever reads the output."""
    for name, estimator in estimators.ESTIMATORS.items():
        assert estimator.name == name
        assert estimator.summary and estimator.assumes and estimator.fails_when
