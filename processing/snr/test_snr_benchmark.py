"""Check that the validation harness itself is sound before trusting what it says.

The scenes have to contain what they claim to contain, and the noise we add has to be the
noise we think we added. Small and fast here; the full sweep is a command, not a test.
"""

import math

import numpy as np
import pytest

from processing.snr import benchmark as bm

SMALL = (192, 192)
FLAT = bm.Scene("flat", 20, 2.0, 0.0, 8.0, 0.0, "test scene")
TEXTURED = bm.Scene("textured", 20, 2.0, 60.0, 2.0, 200.0, "test scene")


def test_beads_do_not_overlap() -> None:
    """The Rose index uses pi r^2 as the bead area, so overlap would silently break it."""
    field = bm.synthetic_field(FLAT, shape=SMALL, edge_blur_px=0.0, seed=0)
    expected = field.bead_count * math.pi * field.radius_px**2

    assert field.bead_mask.sum() == pytest.approx(expected, rel=0.05)


def test_scene_contrast_is_what_was_asked_for() -> None:
    field = bm.synthetic_field(FLAT, shape=SMALL, bead_dn=2600.0, background_dn=2000.0, seed=0)
    assert field.contrast_dn == 600.0


def test_a_flat_scene_really_is_flat() -> None:
    """If the easy scene were not flat, every accuracy number would be wrong."""
    field = bm.synthetic_field(FLAT, shape=SMALL, seed=0)
    assert float(np.std(field.clean[field.background_mask])) < 1e-6


def test_the_textured_scene_carries_the_structure_it_promises() -> None:
    field = bm.synthetic_field(TEXTURED, shape=SMALL, seed=0)
    background = field.clean[field.background_mask]

    # A 60 DN texture plus a 200 DN ramp, so tens of DN of spread and no more.
    assert 20.0 < float(np.std(background)) < 150.0


def test_the_background_mask_keeps_clear_of_the_beads() -> None:
    field = bm.synthetic_field(FLAT, shape=SMALL, seed=0)
    assert not (field.bead_mask & field.background_mask).any()


def test_gaussian_noise_has_the_standard_deviation_we_asked_for() -> None:
    field = bm.synthetic_field(FLAT, shape=SMALL, seed=0)
    noisy = bm.add_gaussian_noise(field.clean, 12.0, seed=1)
    added = noisy - field.clean

    assert float(np.std(added)) == pytest.approx(12.0, rel=0.02)


def test_shot_noise_variance_equals_the_mean_in_electrons() -> None:
    """The defining property of Poisson noise, checked rather than assumed."""
    flat = np.full((256, 256), 1000.0)
    gain = 4.0
    noisy = bm.add_shot_noise(flat, gain_e_per_dn=gain, seed=2)
    electrons = noisy * gain

    assert float(np.var(electrons)) == pytest.approx(1000.0 * gain, rel=0.05)


def test_shot_noise_rejects_a_nonsense_gain() -> None:
    with pytest.raises(ValueError, match="gain must be positive"):
        bm.add_shot_noise(np.ones((8, 8)), gain_e_per_dn=0.0)


def test_gaussian_sweep_produces_one_row_per_combination() -> None:
    rows = bm.gaussian_sweep(sigmas=(5.0, 20.0), seeds=(0,), scenes=(FLAT,), shape=SMALL)
    expected = 2 * 1 * 1 * 2 * len(bm.estimators.ESTIMATORS)  # sigmas x seeds x scenes x regions

    assert len(rows) == expected
    assert {row.region for row in rows} == {"background", "frame"}


def worst_error(rows: list[bm.SweepRow], method: str) -> float:
    return max(
        abs(r.relative_error) for r in rows if r.region == "background" and r.method == method
    )


def test_immerkaer_survives_the_hardest_scene() -> None:
    """The result the recommendation rests on: a few percent even with pixel-scale texture."""
    rows = bm.gaussian_sweep(sigmas=(5.0, 20.0), seeds=(0, 1), scenes=(TEXTURED,), shape=SMALL)
    assert worst_error(rows, "immerkaer") < 0.05


def test_mad_haar_starts_to_slip_when_the_texture_is_pixel_scale() -> None:
    """Its detail band sits at the same scale as the grain, so some of it reads as noise.

    Still usable, and useful precisely because it fails differently from immerkaer: when
    the two agree on a real frame, that agreement means something.
    """
    rows = bm.gaussian_sweep(sigmas=(5.0, 20.0), seeds=(0, 1), scenes=(TEXTURED,), shape=SMALL)
    assert 0.05 < worst_error(rows, "mad_haar") < 0.30


def test_block_percentile_fails_on_pixel_scale_texture() -> None:
    """No 8x8 block is free of grain, so the quietest block is still full of structure."""
    rows = bm.gaussian_sweep(sigmas=(5.0,), seeds=(0,), scenes=(TEXTURED,), shape=SMALL)
    assert worst_error(rows, "block_percentile") > 1.0


def test_global_std_is_wrecked_by_surface_texture() -> None:
    """The result that decides which method the testbed should use."""
    rows = bm.gaussian_sweep(sigmas=(5.0,), seeds=(0,), scenes=(TEXTURED,), shape=SMALL)
    naive = next(r for r in rows if r.region == "background" and r.method == "global_std")

    assert naive.relative_error > 2.0


def test_texture_suite_reports_zero_on_a_flat_scene_background() -> None:
    """No structure and no noise in the background, so the only honest answer is zero."""
    rows = bm.texture_suite(edge_blurs=(0.8,), seeds=(0,), scenes=(FLAT,), shape=SMALL)
    background = [r for r in rows if r.region == "background"]

    assert background
    assert max(r.spurious_sigma_dn for r in background) < 1e-6


def test_texture_suite_catches_methods_that_invent_noise() -> None:
    rows = bm.texture_suite(edge_blurs=(0.8,), seeds=(0,), scenes=(TEXTURED,), shape=SMALL)
    invented = {r.method: r.spurious_sigma_dn for r in rows if r.region == "background"}

    assert invented["immerkaer"] < 1.0
    assert invented["global_std"] > 20.0


def test_shot_noise_sweep_uses_the_background_level_as_truth() -> None:
    rows = bm.shot_noise_sweep(gains_e_per_dn=(2.0,), seeds=(0,), scenes=(FLAT,), shape=SMALL)
    expected = math.sqrt(bm.DEFAULT_BACKGROUND_DN / 2.0)

    assert all(r.true_sigma_dn == pytest.approx(expected) for r in rows)
    good = [r for r in rows if r.region == "background" and r.method == "immerkaer"]
    assert max(abs(r.relative_error) for r in good) < 0.1


def test_segmentation_suite_measures_overlap_with_the_truth() -> None:
    rows = bm.segmentation_suite(
        sigmas=(5.0,), seeds=(0,), scenes=(FLAT,), flatten_options=(0.0,), shape=SMALL
    )

    assert len(rows) == 1
    assert rows[0].intersection_over_union > 0.8


def test_flattening_rescues_segmentation_on_a_lit_gradient() -> None:
    """Quantifies what the flatten option is worth, rather than asserting it helps."""
    rows = bm.segmentation_suite(
        sigmas=(5.0,), seeds=(0, 1), scenes=(TEXTURED,), flatten_options=(0.0, 25.0), shape=SMALL
    )
    naive = np.mean([r.intersection_over_union for r in rows if r.flatten_sigma_px == 0.0])
    flattened = np.mean([r.intersection_over_union for r in rows if r.flatten_sigma_px == 25.0])

    assert flattened > naive


def test_summarise_aggregates_by_method_and_region() -> None:
    rows = bm.gaussian_sweep(sigmas=(5.0, 20.0), seeds=(0, 1), scenes=(FLAT,), shape=SMALL)
    scores = bm.summarise(rows)

    assert len(scores) == len(bm.estimators.ESTIMATORS) * 2
    assert all(score.worst_relative_error >= score.median_relative_error for score in scores)


def test_every_shipped_scene_is_documented() -> None:
    for scene in bm.SCENES:
        assert scene.note
