"""Smoke tests for the command line, so a broken argument never reaches a user."""

import csv

import numpy as np
import pytest
import tifffile

from processing.snr import benchmark as bm
from processing.snr.run import main


def write_frame(path, sigma: float = 8.0, seed: int = 0):
    """A small bead field written as a 16-bit TIFF, the format the testbed produces."""
    field = bm.synthetic_field(
        bm.Scene("small", 12, 2.0, 0.0, 8.0, 0.0, "cli test"), shape=(128, 128), seed=seed
    )
    noisy = bm.add_gaussian_noise(field.clean, sigma, seed=seed + 1)
    tifffile.imwrite(path, np.clip(noisy, 0, 65535).astype(np.uint16))
    return path


def test_analyse_prints_a_result_for_each_frame(tmp_path, capsys) -> None:
    first = write_frame(tmp_path / "a.tif", seed=0)
    second = write_frame(tmp_path / "b.tif", seed=5)

    assert main(["analyse", str(first), str(second)]) == 0
    printed = capsys.readouterr().out

    assert "a.tif" in printed and "b.tif" in printed
    assert "immerkaer" in printed
    assert "spread between methods" in printed


def test_analyse_writes_the_files_it_promises(tmp_path) -> None:
    frame = write_frame(tmp_path / "a.tif")
    out = tmp_path / "out"

    assert main(["analyse", str(frame), "--out", str(out)]) == 0
    assert (out / "results.csv").exists()
    assert (out / "notes.md").exists()
    assert (out / "methods.png").exists()

    with (out / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[0]["image"] == "a.tif"


def test_analyse_can_be_limited_to_one_method(tmp_path) -> None:
    frame = write_frame(tmp_path / "a.tif")
    out = tmp_path / "out"
    main(["analyse", str(frame), "--methods", "immerkaer", "--out", str(out)])

    with (out / "results.csv").open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert methods == {"immerkaer"}


def test_analyse_accepts_hand_picked_rectangles(tmp_path, capsys) -> None:
    frame = write_frame(tmp_path / "a.tif")
    code = main(
        ["analyse", str(frame), "--rect-signal", "0,0,40,40", "--rect-background", "60,60,50,50"]
    )

    assert code == 0
    assert "immerkaer" in capsys.readouterr().out


def test_one_rectangle_without_the_other_is_refused(tmp_path, capsys) -> None:
    frame = write_frame(tmp_path / "a.tif")
    assert main(["analyse", str(frame), "--rect-signal", "0,0,40,40"]) == 2
    assert "must be given together" in capsys.readouterr().err


def test_a_malformed_rectangle_is_reported_by_argparse(tmp_path) -> None:
    frame = write_frame(tmp_path / "a.tif")
    with pytest.raises(SystemExit):
        main(["analyse", str(frame), "--rect-signal", "0,0", "--rect-background", "1,1,2,2"])


def test_flight_reports_a_verdict(capsys) -> None:
    code = main(
        [
            "flight",
            "--signal-e",
            "20000",
            "--contrast-e",
            "6000",
            "--bead-diameter-mm",
            "3",
            "--integration-time-s",
            "0.01",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "Rose index" in printed and "detectable" in printed


def test_flight_converts_from_testbed_digital_numbers(capsys) -> None:
    code = main(
        [
            "flight",
            "--signal-dn",
            "5000",
            "--contrast-dn",
            "1500",
            "--gain-e-per-dn",
            "4",
            "--bead-diameter-mm",
            "3",
            "--integration-time-s",
            "0.01",
        ]
    )

    assert code == 0
    assert "20000 e-" in capsys.readouterr().out


def test_flight_needs_enough_information_to_convert(capsys) -> None:
    assert (
        main(
            [
                "flight",
                "--signal-dn",
                "5000",
                "--bead-diameter-mm",
                "3",
                "--integration-time-s",
                "0.01",
            ]
        )
        == 2
    )
    assert "give either" in capsys.readouterr().err


def test_camera_runs_a_photon_transfer_series(tmp_path, capsys) -> None:
    """Build a small EMVA-style series on disk and drive it through the CLI."""
    gain, offset, read_e = 0.5, 100.0, 10.0

    def sensor(mean_e: float, seed: int):
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(mean_e, size=(96, 96)) + rng.normal(0.0, read_e, size=(96, 96))
        return np.clip(offset + gain * electrons, 0, 65535).astype(np.uint16)

    for name, level, seed in (("dark_a", 0.0, 1), ("dark_b", 0.0, 2)):
        tifffile.imwrite(tmp_path / f"{name}.tif", sensor(level, seed))

    manifest = tmp_path / "series.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_a", "frame_b"])
        for index, level in enumerate((100.0, 500.0, 2000.0, 8000.0, 25000.0, 60000.0)):
            tifffile.imwrite(tmp_path / f"f{index}a.tif", sensor(level, 10 + index * 2))
            tifffile.imwrite(tmp_path / f"f{index}b.tif", sensor(level, 11 + index * 2))
            writer.writerow([f"f{index}a.tif", f"f{index}b.tif"])

    code = main(
        [
            "camera",
            "--manifest",
            str(manifest),
            "--dark-a",
            str(tmp_path / "dark_a.tif"),
            "--dark-b",
            str(tmp_path / "dark_b.tif"),
            "--saturation",
            "65535",
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "system gain" in printed
    gain_line = next(line for line in printed.splitlines() if "system gain" in line)
    assert float(gain_line.split()[2]) == pytest.approx(1.0 / gain, rel=0.05)


def test_camera_rejects_a_manifest_without_the_right_columns(tmp_path) -> None:
    manifest = tmp_path / "bad.csv"
    manifest.write_text("a,b\n1,2\n")
    tifffile.imwrite(tmp_path / "d.tif", np.zeros((8, 8), dtype=np.uint16))

    with pytest.raises(ValueError, match="frame_a and frame_b"):
        main(
            [
                "camera",
                "--manifest",
                str(manifest),
                "--dark-a",
                str(tmp_path / "d.tif"),
                "--dark-b",
                str(tmp_path / "d.tif"),
                "--saturation",
                "65535",
            ]
        )


def test_a_lossy_input_is_flagged_on_stderr(tmp_path, capsys) -> None:
    frame = write_frame(tmp_path / "a.tif")
    renamed = tmp_path / "a.jpg"
    frame.rename(renamed)
    # imageio will still decode the TIFF bytes; the point is that the name is called out.
    main(["analyse", str(renamed)])

    assert "lossy" in capsys.readouterr().err
