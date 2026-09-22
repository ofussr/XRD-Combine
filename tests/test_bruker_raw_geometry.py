from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xrd_workbench.bruker_raw import (
    BrukerRawFile,
    MAGIC_V3,
    RawRange,
    _infer_v4_scan_path,
    export_csv,
    raw_metadata_report,
    read_bruker_raw,
)
from xrd_workbench.io.bruker import read_raw_scans


def make_v3_range(
    *,
    axis_code: int,
    theta: float,
    two_theta: float,
    chi: float = 0.0,
    phi: float = 0.0,
    step: float = 0.1,
    values: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> bytes:
    header = bytearray(304)
    struct.pack_into("<II", header, 0, 304, len(values))
    for offset, value in (
        (8, theta),
        (16, two_theta),
        (24, chi),
        (32, phi),
        (40, 0.0),
        (48, 0.0),
        (56, 0.0),
    ):
        struct.pack_into("<d", header, offset, value)
    struct.pack_into("<I", header, 96, 5)
    struct.pack_into("<ffff", header, 100, 9999.0, 9999.0, 9999.0, 9999.0)
    struct.pack_into("<d", header, 176, step)
    struct.pack_into("<f", header, 192, 1.0)
    struct.pack_into("<I", header, 196, axis_code)
    struct.pack_into("<f", header, 208, 0.0)
    struct.pack_into("<II", header, 224, 40, 40)
    struct.pack_into("<d", header, 240, 1.5406)
    struct.pack_into("<II", header, 252, 4, 40)

    supplementary = bytearray(40)
    struct.pack_into("<II", supplementary, 0, 110, 40)
    struct.pack_into("<d", supplementary, 8, two_theta)
    data = np.asarray(values, dtype="<f4").tobytes()
    return bytes(header + supplementary + data)


def write_v3(path: Path, ranges: list[bytes]) -> None:
    header = bytearray(712)
    header[:8] = MAGIC_V3
    struct.pack_into("<II", header, 8, 1, len(ranges))
    header[16:26] = b"09/21/26\x00\x00"
    header[26:36] = b"12:00:00\x00\x00"
    struct.pack_into("<IIII", header, 548, 21, 13, 0, 0)
    struct.pack_into("<ffff", header, 564, 300.0, 0.6, 0.0, 2.0)
    header[608:612] = b"Cu\x00\x00"
    for offset, value in (
        (616, 1.54184),
        (624, 1.5406),
        (632, 1.54439),
        (640, 1.39222),
        (648, 0.5),
    ):
        struct.pack_into("<d", header, offset, value)
    header[656:660] = b"A\x00\x00\x00"
    path.write_bytes(bytes(header) + b"".join(ranges))


class BrukerRawGeometryTests(unittest.TestCase):
    def test_confirmed_v3_codes_select_the_real_moving_axis(self) -> None:
        cases = (
            (1, "2Theta", "two_theta_scan", 10.0, 10.2),
            (3, "Theta", "theta_scan", 5.0, 5.2),
            (5, "Phi", "phi_scan", -10.0, -9.8),
        )
        with tempfile.TemporaryDirectory() as folder:
            for code, axis, measurement_type, start, end in cases:
                with self.subTest(code=code):
                    path = Path(folder) / f"axis_{code}.raw"
                    write_v3(path, [make_v3_range(
                        axis_code=code,
                        theta=5.0,
                        two_theta=10.0,
                        phi=-10.0,
                    )])
                    raw = read_bruker_raw(path)
                    scan = raw.ranges[0]
                    self.assertEqual(scan.axis_name, axis)
                    self.assertEqual(raw.measurement_type, measurement_type)
                    self.assertEqual(raw.geometry.kind, "single_scan")
                    self.assertEqual(raw.geometry.inner_drives, (axis,))
                    self.assertEqual(raw.geometry.source, "explicit")
                    self.assertEqual(scan.scan_path.primary_drive, axis)
                    self.assertEqual(scan.scan_path.moving_drives, (axis,))
                    self.assertEqual(scan.metadata["scan_axis_code"], code)
                    np.testing.assert_allclose(scan.axis[[0, -1]], [start, end])

    def test_explicit_theta_ranges_with_varying_two_theta_form_rsm(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rsm.raw"
            write_v3(path, [
                make_v3_range(
                    axis_code=3,
                    theta=21.0,
                    two_theta=44.0 + 0.02 * index,
                    step=0.01,
                    values=(1.0, 2.0, 3.0) if index < 2 else (1.0,),
                )
                for index in range(3)
            ])
            raw = read_bruker_raw(path)
            self.assertTrue(raw.is_rsm)
            self.assertEqual(raw.measurement_type, "rsm")
            self.assertEqual(raw.geometry.kind, "rsm")
            self.assertEqual(raw.geometry.inner_drives, ("Theta",))
            self.assertEqual(raw.geometry.outer_drives, ("2Theta",))
            self.assertEqual(raw.geometry.source, "explicit")
            scans = read_raw_scans(path, raw=raw)
            self.assertEqual(len(scans), 2)
            self.assertTrue(all(scan.axis_name == "Theta" for scan in scans))
            self.assertTrue(all(scan.metadata["is_rsm"] for scan in scans))

    def test_explicit_phi_ranges_with_varying_chi_form_pole_figure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pole.raw"
            write_v3(path, [
                make_v3_range(
                    axis_code=5,
                    theta=11.0,
                    two_theta=22.0,
                    chi=10.0 * index,
                    phi=0.0,
                    step=1.0,
                )
                for index in range(3)
            ])

            raw = read_bruker_raw(path)

            self.assertTrue(raw.is_pole_figure)
            self.assertEqual(raw.measurement_type, "pole_figure")
            self.assertEqual(raw.geometry.inner_drives, ("Phi",))
            self.assertEqual(raw.geometry.outer_drives, ("Chi",))

    def test_unknown_code_uses_point_index_and_is_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unknown.raw"
            write_v3(path, [
                make_v3_range(
                    axis_code=99,
                    theta=21.0,
                    two_theta=44.0 + index,
                )
                for index in range(3)
            ])
            raw = read_bruker_raw(path)
            self.assertFalse(raw.is_rsm)
            self.assertEqual(raw.measurement_type, "scan_series")
            self.assertEqual(raw.geometry.source, "ambiguous")
            self.assertEqual(raw.geometry.inner_drives, ())
            self.assertTrue(all(scan.axis_name == "Point" for scan in raw.ranges))
            np.testing.assert_array_equal(raw.ranges[0].axis, [0.0, 1.0, 2.0])

    def test_locked_coupled_v4_path_retains_both_moving_drives(self) -> None:
        path = _infer_v4_scan_path(
            "Locked Coupled",
            20.0,
            0.02,
            {"2Theta": 20.0, "Theta": 10.0},
        )
        scan = RawRange(
            index=0,
            scan_type="Locked Coupled",
            axis_name="2Theta",
            start_angle=20.0,
            step_size=0.02,
            time_per_step=1.0,
            drives={"2Theta": 20.0, "Theta": 10.0},
            data=np.ones((3, 1)),
            channel_names=["intensity"],
            scan_path=path,
        )

        self.assertEqual(path.primary_drive, "2Theta")
        self.assertEqual(path.moving_drives, ("2Theta", "Theta"))
        np.testing.assert_allclose(scan.coordinate("2Theta"), [20.0, 20.02, 20.04])
        np.testing.assert_allclose(scan.coordinate("Theta"), [10.0, 10.01, 10.02])

    def test_unlocked_coupled_v4_path_does_not_invent_a_ratio(self) -> None:
        path = _infer_v4_scan_path(
            "Unlocked Coupled",
            20.0,
            0.02,
            {"2Theta": 20.0, "Theta": 10.0},
        )

        self.assertEqual(path.primary_drive, "2Theta")
        self.assertEqual(path.moving_drives, ("2Theta",))
        self.assertNotIn("Theta", path.drive_steps)

    def test_multi_range_csv_exports_every_moving_drive_coordinate(self) -> None:
        path = _infer_v4_scan_path(
            "Locked Coupled",
            20.0,
            0.02,
            {"2Theta": 20.0, "Theta": 10.0},
        )
        ranges = [
            RawRange(
                index=index,
                scan_type="Locked Coupled",
                axis_name="2Theta",
                start_angle=20.0,
                step_size=0.02,
                time_per_step=1.0,
                drives={"2Theta": 20.0, "Theta": 10.0},
                data=np.ones((3, 1)),
                channel_names=["intensity"],
                scan_path=path,
            )
            for index in range(2)
        ]
        raw = BrukerRawFile(
            Path("coupled.raw"),
            4,
            "RAW4.00",
            "",
            "",
            {},
            ranges,
        )

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "coupled.csv"
            export_csv(raw, target)
            rows = target.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rows[0], "range,scan_axis,x,intensity,2Theta,Theta")
        self.assertEqual(rows[2].split(",")[-2:], ["20.02", "10.01"])

    def test_diagnostic_report_retains_raw_headers_without_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.raw"
            write_v3(path, [make_v3_range(
                axis_code=1,
                theta=5.0,
                two_theta=10.0,
            )])
            raw = read_bruker_raw(path)
            report = raw_metadata_report(raw)
            encoded = json.dumps(report)
            self.assertNotIn('"data"', encoded)
            self.assertEqual(report["geometry"]["kind"], "single_scan")
            self.assertEqual(
                report["ranges"][0]["scan_path"]["primary_drive"],
                "2Theta",
            )
            self.assertEqual(report["metadata"]["goniometer_code"], 21)
            self.assertEqual(report["metadata"]["goniometer_stage_code"], 13)
            self.assertAlmostEqual(report["metadata"]["goniometer_radius"], 300.0)
            self.assertEqual(
                report["ranges"][0]["metadata"]["supplementary_segments"][0]["type"],
                110,
            )
            self.assertEqual(len(report["metadata"]["file_header_raw_hex"]), 1424)
            self.assertEqual(
                len(report["ranges"][0]["metadata"]["range_header_raw_hex"]),
                608,
            )


if __name__ == "__main__":
    unittest.main()
