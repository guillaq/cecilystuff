"""Command line entry point.

    uv run python -m processing.snr.run analyse frames/*.tif --out results
    uv run python -m processing.snr.run benchmark --out validation
    uv run python -m processing.snr.run camera --manifest series.csv --saturation 65535
    uv run python -m processing.snr.run flight --signal-e 8000 --contrast-e 2000 \
        --bead-diameter-mm 3 --integration-time-s 0.01

Every subcommand prints a readable summary and, when given --out, writes the same numbers
to CSV so they can go into a report without retyping.
"""

import argparse
import csv
import sys
from pathlib import Path

from processing.snr import benchmark as benchmark_module
from processing.snr import report
from processing.snr.camera import FramePair, photon_transfer
from processing.snr.estimators import ESTIMATORS
from processing.snr.flight import WHEELCAM, electrons_from_testbed, predict
from processing.snr.images import check_suffix, load_image
from processing.snr.metrics import ImageAnalysis, measure
from processing.snr.segmentation import Rect, bead_regions, rect_regions


def _parse_rect(text: str) -> Rect:
    parts = [int(piece) for piece in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected row,col,height,width but got {text!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _print_analysis(analysis: ImageAnalysis) -> None:
    print(f"\n{analysis.image}  ({analysis.shape[0]}x{analysis.shape[1]})")
    print(
        f"  {analysis.bead_count} beads, mean area {analysis.mean_bead_area_px:.0f} px, "
        f"threshold {analysis.threshold_dn:.0f} DN"
    )
    for warning in analysis.warnings:
        print(f"  WARNING: {warning}")

    print(
        f"  {'region':<11} {'method':<17} {'sigma DN':>9} {'CNR':>7} {'Rose':>8} "
        f"{'px precision':>13}  verdict"
    )
    for result in analysis.results:
        print(
            f"  {result.noise_region:<11} {result.method:<17} {result.sigma_dn:>9.2f} "
            f"{result.cnr:>7.1f} {result.rose_index:>8.1f} "
            f"{result.displacement_precision_px:>13.3f}  {result.verdict}"
        )
    spread = analysis.spread()
    print(f"  spread between methods: {spread:.2f}x")
    if spread > 2.0:
        print("  the methods disagree by more than 2x, so at least one assumption is broken here")


def command_analyse(args: argparse.Namespace) -> int:
    rect_pair = (args.rect_signal, args.rect_background)
    if any(rect_pair) and not all(rect_pair):
        print("--rect-signal and --rect-background must be given together", file=sys.stderr)
        return 2

    analyses: list[ImageAnalysis] = []
    for path in args.images:
        for note in check_suffix(path):
            print(f"{Path(path).name}: WARNING: {note}", file=sys.stderr)
        image = load_image(path, channel=args.channel, full_scale=args.full_scale)

        if all(rect_pair):
            regions = rect_regions(image.shape, args.rect_signal, args.rect_background)
        else:
            regions = bead_regions(
                image.data,
                polarity=args.polarity,
                erode_px=args.erode_px,
                min_bead_area_px=args.min_bead_area_px,
                flatten_sigma_px=args.flatten_sigma_px,
            )

        analysis = measure(image, regions, methods=args.methods)
        analyses.append(analysis)
        _print_analysis(analysis)

    if args.out:
        directory = Path(args.out)
        report.analysis_to_csv(analyses, directory / "results.csv")
        report.write_analysis_notes(analyses, directory / "notes.md")
        report.plot_method_comparison(analyses, directory / "methods.png")
        print(f"\nwrote results.csv, notes.md and methods.png to {directory}")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    print("running the validation suites, this takes a few minutes...")
    result = benchmark_module.run_benchmark()
    ranked = report.best_methods(result.scores)

    print("\nWorst-case relative error per method, noise measured on the background region.")
    print("Ranked by worst case across all scenes, best first.\n")
    print(f"  {'method':<17} {'noise':<9} {'scene':<20} {'median err':>11} {'worst err':>10}")
    for score in ranked:
        print(
            f"  {score.method:<17} {score.suite:<9} {score.scene:<20} "
            f"{score.median_relative_error:>10.1%} {score.worst_relative_error:>10.1%}"
        )

    if args.out:
        written = report.write_benchmark(result, args.out)
        print(f"\nwrote {len(written)} files to {args.out}")
    return 0


def _load_manifest(path: Path, channel: int | None, full_scale: float | None) -> list[FramePair]:
    """Read a CSV of `frame_a,frame_b` paths, relative to the manifest's own directory."""
    pairs: list[FramePair] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "frame_a" not in row or "frame_b" not in row:
                raise ValueError(f"{path} needs columns frame_a and frame_b")
            first = load_image(path.parent / row["frame_a"], channel, full_scale)
            second = load_image(path.parent / row["frame_b"], channel, full_scale)
            pairs.append(FramePair(frame_a=first.data, frame_b=second.data))
    if not pairs:
        raise ValueError(f"{path} lists no frame pairs")
    return pairs


def command_camera(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    pairs = _load_manifest(manifest, args.channel, args.full_scale)
    dark = FramePair(
        frame_a=load_image(args.dark_a, args.channel, args.full_scale).data,
        frame_b=load_image(args.dark_b, args.channel, args.full_scale).data,
    )
    result = photon_transfer(dark, pairs, saturation_dn=args.saturation)

    print("\nPhoton transfer, EMVA 1288 method")
    print(f"  system gain          {result.system_gain_e_per_dn:.3f} e-/DN")
    print(f"  read noise           {result.read_noise_e:.2f} e-  ({result.read_noise_dn:.2f} DN)")
    print(f"  read noise, fit       {result.read_noise_fit_dn:.2f} DN  (cross-check)")
    print(f"  dark offset          {result.dark_offset_dn:.1f} DN")
    print(f"  saturation capacity  {result.saturation_capacity_e:.0f} e-")
    print(f"  dynamic range        {result.dynamic_range_db:.1f} dB")
    print(f"  best possible SNR    {result.max_snr:.0f}  ({result.max_snr_db:.1f} dB)")
    print(f"  DSNU                 {result.dsnu_e:.2f} e-")
    print(f"  PRNU                 {result.prnu_percent:.2f} %")
    print(f"  linearity            r^2 = {result.r_squared:.5f} over {result.points_used} levels")
    for warning in result.warnings():
        print(f"  WARNING: {warning}")
    return 0


def command_flight(args: argparse.Namespace) -> int:
    signal_e, contrast_e = args.signal_e, args.contrast_e
    if signal_e is None or contrast_e is None:
        if args.gain_e_per_dn is None or args.signal_dn is None or args.contrast_dn is None:
            print(
                "give either --signal-e and --contrast-e, or --signal-dn, --contrast-dn and "
                "--gain-e-per-dn to convert a testbed measurement",
                file=sys.stderr,
            )
            return 2
        signal_e, contrast_e = electrons_from_testbed(
            args.signal_dn, args.contrast_dn, args.gain_e_per_dn, args.illumination_ratio
        )

    result = predict(
        signal_e=signal_e,
        contrast_e=contrast_e,
        bead_diameter_mm=args.bead_diameter_mm,
        integration_time_s=args.integration_time_s,
        distance_cm=args.distance_cm,
    )

    print("\nPredicted IDEFIX WheelCam performance")
    print(f"  pixel scale          {result.pixel_scale_um:.1f} um")
    print(
        f"  bead size            {result.bead_diameter_px:.1f} px across, {result.bead_area_px:.0f} px area"
    )
    print(f"  signal               {result.signal_e:.0f} e-, contrast {result.contrast_e:.0f} e-")
    print(
        f"  noise                {result.total_noise_e:.1f} e-  "
        f"(shot {result.shot_noise_e:.1f}, dark {result.dark_noise_e:.1f}, "
        f"read {result.read_noise_e:.1f})"
    )
    print(f"  per-pixel CNR        {result.cnr:.2f}")
    print(f"  Rose index           {result.rose_index:.1f}  -> {result.verdict}")
    print(f"  limited by           {result.dominant_noise()}")
    print(f"\n  camera model: {WHEELCAM}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snr", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="measure SNR and CNR on testbed frames")
    analyse.add_argument("images", nargs="+", help="image files")
    analyse.add_argument("--methods", nargs="+", choices=sorted(ESTIMATORS), default=None)
    analyse.add_argument("--polarity", choices=["bright", "dark"], default="bright")
    analyse.add_argument("--erode-px", type=int, default=2)
    analyse.add_argument("--min-bead-area-px", type=int, default=4)
    analyse.add_argument(
        "--flatten-sigma-px",
        type=float,
        default=0.0,
        help="blur width used to remove a lighting gradient before thresholding only",
    )
    analyse.add_argument("--channel", type=int, default=None, help="channel of a colour image")
    analyse.add_argument("--full-scale", type=float, default=None, help="saturation level in DN")
    analyse.add_argument("--rect-signal", type=_parse_rect, default=None, help="row,col,h,w")
    analyse.add_argument("--rect-background", type=_parse_rect, default=None, help="row,col,h,w")
    analyse.add_argument("--out", default=None, help="directory for CSV and plots")
    analyse.set_defaults(func=command_analyse)

    bench = subparsers.add_parser("benchmark", help="validate the estimators on synthetic scenes")
    bench.add_argument("--out", default=None, help="directory for CSV and plots")
    bench.set_defaults(func=command_benchmark)

    camera = subparsers.add_parser("camera", help="EMVA 1288 photon transfer from a frame series")
    camera.add_argument("--manifest", required=True, help="CSV with frame_a,frame_b columns")
    camera.add_argument("--dark-a", required=True)
    camera.add_argument("--dark-b", required=True)
    camera.add_argument("--saturation", type=float, required=True, help="full scale in DN")
    camera.add_argument("--channel", type=int, default=None)
    camera.add_argument("--full-scale", type=float, default=None)
    camera.set_defaults(func=command_camera)

    flight = subparsers.add_parser("flight", help="predict WheelCam performance on Phobos")
    flight.add_argument("--signal-e", type=float, default=None)
    flight.add_argument("--contrast-e", type=float, default=None)
    flight.add_argument("--signal-dn", type=float, default=None)
    flight.add_argument("--contrast-dn", type=float, default=None)
    flight.add_argument("--gain-e-per-dn", type=float, default=None)
    flight.add_argument("--illumination-ratio", type=float, default=1.0)
    flight.add_argument("--bead-diameter-mm", type=float, required=True)
    flight.add_argument("--integration-time-s", type=float, required=True)
    flight.add_argument("--distance-cm", type=float, default=None)
    flight.set_defaults(func=command_flight)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
