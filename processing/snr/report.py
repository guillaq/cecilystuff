"""Write results out as CSV, Markdown and plots.

Nothing here computes anything. It exists so the analysis modules never have to think
about file formats, and so every table in the README can be regenerated from a command
rather than typed by hand.
"""

import csv
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display on CI or on a headless testbed machine.
import matplotlib.pyplot as plt

from processing.snr.benchmark import (
    BenchmarkReport,
    MethodScore,
    SweepRow,
    TextureRow,
)
from processing.snr.metrics import ImageAnalysis


def _as_rows(records: list[Any]) -> list[dict[str, Any]]:
    if not records:
        return []
    if not is_dataclass(records[0]):
        raise TypeError("write_csv expects a list of dataclass instances")
    return [asdict(record) for record in records]


def write_csv(records: list[Any], path: str | Path) -> Path:
    """Write a list of dataclasses to CSV, one row each, columns in field order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _as_rows(records)
    if not rows:
        path.write_text("")
        return path

    columns = [f.name for f in fields(records[0])]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_table(records: list[Any], columns: list[str], digits: int = 3) -> str:
    """Render dataclasses as a Markdown table, for pasting into a README."""
    if not records:
        return "_no rows_"

    def render(value: Any) -> str:
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(render(getattr(record, column)) for column in columns) + " |"
        for record in records
    ]
    return "\n".join([header, divider, *body])


def analysis_to_csv(analyses: list[ImageAnalysis], path: str | Path) -> Path:
    """Flatten per-image results into one row per (image, method, region)."""
    rows = [result for analysis in analyses for result in analysis.results]
    return write_csv(rows, path)


def write_analysis_notes(analyses: list[ImageAnalysis], path: str | Path) -> Path:
    """Save the per-image warnings, which are the part people skip and should not."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for analysis in analyses:
        lines.append(f"## {analysis.image}")
        lines.append(
            f"- {analysis.shape[0]}x{analysis.shape[1]} px, {analysis.bead_count} beads, "
            f"mean bead area {analysis.mean_bead_area_px:.1f} px"
        )
        spread = analysis.spread()
        lines.append(f"- spread between methods: {spread:.2f}x")
        if spread > 2.0:
            lines.append(
                "  - the methods disagree by more than 2x, so at least one assumption is "
                "broken on this frame. Do not average them, work out which one to trust"
            )
        for warning in analysis.warnings:
            lines.append(f"- WARNING: {warning}")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def plot_method_comparison(analyses: list[ImageAnalysis], path: str | Path) -> Path:
    """One grouped bar per image showing what each method thinks the noise is."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    background = [r for a in analyses for r in a.results if r.noise_region == "background"]
    methods = sorted({r.method for r in background})
    images = [a.image for a in analyses]
    if not methods or not images:
        raise ValueError("nothing to plot")

    width = 0.8 / len(methods)
    figure, axes = plt.subplots(figsize=(max(6.0, 1.5 * len(images)), 4.5))
    for index, method in enumerate(methods):
        values = [
            next(
                (r.sigma_dn for r in background if r.image == image and r.method == method),
                float("nan"),
            )
            for image in images
        ]
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(images))]
        axes.bar(offsets, values, width=width, label=method)

    axes.set_xticks(range(len(images)))
    axes.set_xticklabels(images, rotation=30, ha="right")
    axes.set_ylabel("estimated noise sigma (DN)")
    axes.set_title("Noise estimate per method, measured on the background region")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_sweep(rows: list[SweepRow], path: str | Path, region: str = "background") -> Path:
    """Estimated against true sigma, one panel per scene. The diagonal is correct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subset = [r for r in rows if r.region == region and r.suite == "gaussian"]
    scenes = sorted({r.scene for r in subset})
    methods = sorted({r.method for r in subset})
    if not scenes:
        raise ValueError("no gaussian sweep rows to plot")

    figure, axes_row = plt.subplots(1, len(scenes), figsize=(4.0 * len(scenes), 4.0), squeeze=False)
    for column, scene in enumerate(scenes):
        axes = axes_row[0][column]
        for method in methods:
            points = sorted(
                (
                    (r.true_sigma_dn, r.estimated_sigma_dn)
                    for r in subset
                    if r.scene == scene and r.method == method
                ),
                key=lambda pair: pair[0],
            )
            axes.plot(
                [p[0] for p in points], [p[1] for p in points], "o", markersize=3, label=method
            )
        ticks = sorted({r.true_sigma_dn for r in subset})
        axes.plot([ticks[0], ticks[-1]], [ticks[0], ticks[-1]], "k--", linewidth=1, label="truth")
        axes.set_xscale("log")
        axes.set_yscale("log")
        # Default log ticks collide at these values, so label only the levels we tested.
        axes.set_xticks(ticks)
        axes.set_xticklabels([f"{value:g}" for value in ticks])
        axes.minorticks_off()
        axes.set_xlabel("true sigma (DN)")
        if column == 0:
            axes.set_ylabel("estimated sigma (DN)")
            axes.legend(fontsize=7)
        axes.set_title(scene, fontsize=9)

    figure.suptitle(f"Recovery of a known noise level, measured on the {region}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_texture_bias(rows: list[TextureRow], path: str | Path, region: str = "background") -> Path:
    """How much noise each method invents on a noise-free scene. Zero is correct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subset = [r for r in rows if r.region == region and r.edge_blur_px == 0.8]
    scenes = sorted({r.scene for r in subset})
    methods = sorted({r.method for r in subset})
    if not scenes:
        raise ValueError("no texture rows to plot")

    width = 0.8 / len(methods)
    tallest = 0.0
    figure, axes = plt.subplots(figsize=(2.2 * len(scenes) + 3, 4.5))
    for index, method in enumerate(methods):
        values = [
            float(
                sum(r.spurious_sigma_dn for r in subset if r.scene == scene and r.method == method)
                / max(sum(1 for r in subset if r.scene == scene and r.method == method), 1)
            )
            for scene in scenes
        ]
        tallest = max(tallest, *values)
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(scenes))]
        axes.bar(offsets, values, width=width, label=method)

    axes.set_xticks(range(len(scenes)))
    axes.set_xticklabels(scenes, rotation=20, ha="right")
    # Log would hide the exact zeros, which are the whole point on the easy scenes.
    axes.set_yscale("symlog", linthresh=0.1)
    axes.set_ylim(0.0, tallest * 2.0)
    axes.set_ylabel("noise reported on a noise-free scene (DN)")
    axes.set_title(f"Structure mistaken for noise, measured on the {region}. Zero is correct")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def best_methods(scores: list[MethodScore], region: str = "background") -> list[MethodScore]:
    """Rank methods by their worst-case error across scenes, worst case first.

    Ranking on the worst case rather than the median is deliberate: a method that is
    usually right and occasionally off by a factor of eight is not usable for a go/no-go
    decision on a frame nobody has looked at.
    """
    subset = [s for s in scores if s.region == region]
    worst: dict[str, float] = {}
    for score in subset:
        worst[score.method] = max(worst.get(score.method, 0.0), score.worst_relative_error)
    order = sorted(worst, key=lambda method: worst[method])
    return [s for method in order for s in subset if s.method == method]


def write_benchmark(report: BenchmarkReport, directory: str | Path) -> dict[str, Path]:
    """Write every benchmark table and figure into one directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "sweeps": write_csv(report.sweeps, directory / "sweeps.csv"),
        "scores": write_csv(report.scores, directory / "scores.csv"),
        "textures": write_csv(report.textures, directory / "texture_bias.csv"),
        "segmentation": write_csv(report.segmentations, directory / "segmentation.csv"),
        "sweep_plot": plot_sweep(report.sweeps, directory / "sweep.png"),
        "texture_plot": plot_texture_bias(report.textures, directory / "texture_bias.png"),
    }
