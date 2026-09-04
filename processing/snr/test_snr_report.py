"""The output layer only has to be correct and boring, but it is what people read."""

import csv

import numpy as np
import pytest

from processing.snr import benchmark as bm
from processing.snr import report
from processing.snr.images import Frame
from processing.snr.metrics import measure
from processing.snr.segmentation import rect_regions

SMALL = (128, 128)
FLAT = bm.Scene("flat", 12, 2.0, 0.0, 8.0, 0.0, "test scene")


def sample_analysis(name: str, sigma: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = np.full(SMALL, 1000.0)
    data[:, :64] = 1500.0
    data = data + rng.normal(0.0, sigma, size=SMALL)
    image = Frame(name=name, data=data, full_scale=65535.0, source_dtype="float64")
    regions = rect_regions(SMALL, (0, 0, 128, 60), (0, 68, 128, 60))
    return measure(image, regions, noise_regions=("background",))


def test_csv_has_one_row_per_result_and_named_columns(tmp_path) -> None:
    analyses = [sample_analysis("a.tif", 5.0), sample_analysis("b.tif", 20.0, seed=1)]
    path = report.analysis_to_csv(analyses, tmp_path / "results.csv")

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == sum(len(a.results) for a in analyses)
    assert {"image", "method", "sigma_dn", "cnr", "rose_index", "verdict"} <= set(rows[0])
    assert {row["image"] for row in rows} == {"a.tif", "b.tif"}


def test_empty_input_writes_an_empty_file_rather_than_crashing(tmp_path) -> None:
    path = report.write_csv([], tmp_path / "nothing.csv")
    assert path.read_text() == ""


def test_write_csv_rejects_plain_dictionaries(tmp_path) -> None:
    with pytest.raises(TypeError, match="dataclass"):
        report.write_csv([{"a": 1}], tmp_path / "x.csv")


def test_notes_carry_the_warnings_and_the_spread(tmp_path) -> None:
    text = report.write_analysis_notes(
        [sample_analysis("a.tif", 5.0)], tmp_path / "notes.md"
    ).read_text()

    assert "## a.tif" in text
    assert "spread between methods" in text


def test_notes_call_out_a_large_disagreement(tmp_path) -> None:
    """A 2x spread means an assumption is broken, and the note has to say so."""
    rng = np.random.default_rng(3)
    ramp = np.linspace(0.0, 2000.0, 128)[np.newaxis, :] * np.ones((128, 1))
    data = 1000.0 + ramp + rng.normal(0.0, 5.0, size=SMALL)
    data[:, :60] += 500.0
    image = Frame(name="ramped.tif", data=data, full_scale=65535.0, source_dtype="float64")
    analysis = measure(
        image,
        rect_regions(SMALL, (0, 0, 128, 60), (0, 68, 128, 60)),
        noise_regions=("background",),
    )

    text = report.write_analysis_notes([analysis], tmp_path / "notes.md").read_text()
    assert analysis.spread() > 2.0
    assert "disagree by more than 2x" in text


def test_markdown_table_renders_the_requested_columns() -> None:
    analysis = sample_analysis("a.tif", 5.0)
    table = report.markdown_table(analysis.results, ["method", "sigma_dn", "cnr"], digits=2)
    lines = table.splitlines()

    assert lines[0] == "| method | sigma_dn | cnr |"
    assert len(lines) == len(analysis.results) + 2


def test_markdown_table_says_so_when_there_is_nothing_to_show() -> None:
    assert report.markdown_table([], ["method"]) == "_no rows_"


def test_plots_are_written(tmp_path) -> None:
    analyses = [sample_analysis("a.tif", 5.0), sample_analysis("b.tif", 20.0, seed=1)]
    path = report.plot_method_comparison(analyses, tmp_path / "methods.png")

    assert path.exists() and path.stat().st_size > 1000


def test_benchmark_output_lands_in_one_directory(tmp_path) -> None:
    sweeps = bm.gaussian_sweep(sigmas=(5.0,), seeds=(0,), scenes=(FLAT,), shape=SMALL)
    textures = bm.texture_suite(edge_blurs=(0.8,), seeds=(0,), scenes=(FLAT,), shape=SMALL)
    segments = bm.segmentation_suite(
        sigmas=(5.0,), seeds=(0,), scenes=(FLAT,), flatten_options=(0.0,), shape=SMALL
    )
    written = report.write_benchmark(
        bm.BenchmarkReport(sweeps=sweeps, textures=textures, segmentations=segments), tmp_path
    )

    assert set(written) == {
        "sweeps",
        "scores",
        "textures",
        "segmentation",
        "sweep_plot",
        "texture_plot",
    }
    assert all(path.exists() for path in written.values())


def test_methods_are_ranked_by_their_worst_case() -> None:
    """Ranking on the median would let a method that fails badly once come out on top."""
    scenes = (FLAT, bm.Scene("textured", 12, 2.0, 60.0, 2.0, 200.0, "test scene"))
    rows = bm.gaussian_sweep(sigmas=(5.0,), seeds=(0,), scenes=scenes, shape=SMALL)
    ranked = report.best_methods(bm.summarise(rows))

    assert ranked[0].method == "immerkaer"
    assert ranked[-1].method in {"global_std", "robust_mad"}
