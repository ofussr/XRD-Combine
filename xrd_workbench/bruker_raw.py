#!/usr/bin/env python3
"""
Чтение двоичных файлов Siemens/Bruker RAW.

Поддерживаются:
    RAW1.01 — RAW v3;
    RAW4.00 — RAW v4.

Библиотека сохраняет исходные интенсивности без нормировки, интерполяции
и заполнения пропусков.

Примеры:
    python bruker_raw.py scan.raw
    python bruker_raw.py scan.raw --csv scan.csv
    python bruker_raw.py scan.raw --plot scan.png --log
    python bruker_raw.py multi_range.raw --range 4 --csv range_4.csv

Использование из другого сценария:
    from bruker_raw import read_bruker_raw

    raw = read_bruker_raw("scan.raw")
    scan = raw.ranges[0]
    x = scan.axis
    y = scan.intensity
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MAGIC_V3 = b"RAW1.01\x00"
MAGIC_V4 = b"RAW4.00\x00"
V3_FILE_HEADER_SIZE = 712
V3_RANGE_HEADER_SIZE = 304

# Confirmed against real RAW1.01 measurements made on the same Bruker D8.
# Unknown values are deliberately not assigned a physical drive until a
# controlled measurement or vendor documentation confirms them.
V3_SCAN_AXIS_CODES = {
    1: "2Theta",
    3: "Theta",
    5: "Phi",
}


@dataclass
class ScanPath:
    """Per-point movement of every known drive inside one RAW range."""

    primary_drive: str | None
    drive_steps: dict[str, float] = field(default_factory=dict)
    source: str = "ambiguous"  # explicit | inferred | manual | ambiguous
    reason: str = ""

    @property
    def moving_drives(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, step in self.drive_steps.items()
            if np.isfinite(step) and not np.isclose(step, 0.0, atol=1e-15)
        )

    @property
    def is_ambiguous(self) -> bool:
        return self.primary_drive is None or self.source == "ambiguous"


@dataclass
class MeasurementGeometry:
    """Whole-file inner and outer scan dimensions."""

    kind: str
    inner_drives: tuple[str, ...]
    outer_drives: tuple[str, ...]
    source: str
    reason: str = ""


@dataclass
class RawRange:
    """Один измерительный диапазон внутри RAW-файла."""

    index: int
    scan_type: str
    axis_name: str
    start_angle: float
    step_size: float
    time_per_step: float
    drives: dict[str, float]
    data: np.ndarray
    channel_names: list[str]
    generator_voltage: float | None = None
    generator_current: float | None = None
    wavelength: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scan_path: ScanPath = field(default_factory=lambda: ScanPath(None))

    @property
    def point_count(self) -> int:
        return self.data.shape[0]

    @property
    def channel_count(self) -> int:
        return self.data.shape[1]

    @property
    def axis(self) -> np.ndarray:
        primary = self.scan_path.primary_drive
        if primary:
            return self.coordinate(primary)
        return self.start_angle + self.step_size * np.arange(self.point_count)

    @property
    def intensity(self) -> np.ndarray:
        """Первый измерительный канал, обычно основная интенсивность."""
        if self.channel_count == 0:
            return np.empty(self.point_count, dtype=float)
        return self.data[:, 0]

    @property
    def theta(self) -> float:
        return self.drives.get("Theta", np.nan)

    @property
    def two_theta(self) -> float:
        return self.drives.get("2Theta", np.nan)

    @property
    def chi(self) -> float:
        return self.drives.get("Chi", np.nan)

    @property
    def phi_start(self) -> float:
        return self.drives.get("Phi", np.nan)

    @property
    def phi(self) -> np.ndarray:
        if self.axis_name == "Phi":
            return self.axis
        return np.full(self.point_count, self.phi_start)

    def coordinate(self, drive_name: str) -> np.ndarray:
        """Вернуть меняющуюся либо постоянную координату указанного привода."""
        if self.scan_path.primary_drive is not None:
            start = self.drives.get(drive_name, np.nan)
            step = self.scan_path.drive_steps.get(drive_name, 0.0)
            return start + step * np.arange(self.point_count, dtype=float)
        if drive_name == self.axis_name:
            return self.axis
        return np.full(self.point_count, self.drives.get(drive_name, np.nan))

    def set_scan_path(
        self,
        primary_drive: str,
        *,
        factors: dict[str, float] | None = None,
        source: str = "manual",
        reason: str = "manual scan geometry override",
    ) -> None:
        """Override scan motion; factors are relative to the stored step."""
        if factors is None:
            factors = {primary_drive: 1.0}
        self.scan_path = ScanPath(
            primary_drive,
            {
                name: float(factor) * float(self.step_size)
                for name, factor in factors.items()
            },
            source,
            reason,
        )
        self.axis_name = primary_drive
        if primary_drive in self.drives:
            self.start_angle = float(self.drives[primary_drive])


@dataclass
class BrukerRawFile:
    """Общее представление RAW v3 и RAW v4."""

    path: Path
    version: int
    magic: str
    date: str
    time: str
    metadata: dict[str, Any]
    ranges: list[RawRange]
    geometry: MeasurementGeometry = field(
        default_factory=lambda: MeasurementGeometry(
            "unknown", (), (), "ambiguous", "not classified"
        )
    )

    @property
    def is_pole_figure(self) -> bool:
        return self.geometry.kind == "pole_figure"

    @property
    def is_rsm(self) -> bool:
        return self.geometry.kind == "rsm"

    @property
    def measurement_type(self) -> str:
        """Conservative whole-file measurement classification."""
        if self.is_pole_figure:
            return "pole_figure"
        if self.is_rsm:
            return "rsm"
        if self.geometry.kind == "coupled_scan":
            return "coupled_scan"
        nonempty = [scan for scan in self.ranges if scan.point_count]
        if len(nonempty) == 1:
            return {
                "2Theta": "two_theta_scan",
                "Theta": "theta_scan",
                "Omega": "omega_scan",
                "Phi": "phi_scan",
                "Chi": "chi_scan",
            }.get(nonempty[0].axis_name, "single_scan")
        if nonempty:
            return "scan_series"
        return "empty"

    def override_scan_geometry(
        self,
        primary_drive: str,
        *,
        factors: dict[str, float] | None = None,
        range_indices: list[int] | None = None,
    ) -> None:
        """Manually assign a simple or coupled path to selected ranges."""
        wanted = None if range_indices is None else set(range_indices)
        for scan in self.ranges:
            if wanted is None or scan.index in wanted:
                scan.set_scan_path(primary_drive, factors=factors)
        self.geometry = _classify_file_geometry(self.ranges)


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


def _f64(data: bytes, offset: int) -> float:
    return struct.unpack_from("<d", data, offset)[0]


def _text(data: bytes, offset: int, length: int) -> str:
    raw = data[offset : offset + length].split(b"\x00", 1)[0]
    return raw.decode("cp1252", errors="replace").strip()


def _validate_slice(data: bytes, offset: int, length: int, description: str) -> None:
    if offset < 0 or length < 0 or offset + length > len(data):
        raise ValueError(
            f"{description}: диапазон байтов {offset}:{offset + length} "
            f"выходит за размер файла {len(data)}."
        )


def _channel_names(names: list[str], count: int) -> list[str]:
    """Сформировать неповторяющиеся имена каналов."""
    result: list[str] = []
    used: dict[str, int] = {}

    for index in range(count):
        base = names[index].strip() if index < len(names) else ""
        if not base:
            base = "intensity" if index == 0 else f"channel_{index}"

        used[base] = used.get(base, 0) + 1
        if used[base] == 1 and names.count(base) <= 1:
            result.append(base)
        else:
            result.append(f"{base}_{used[base]}")

    return result


def _varying_outer_drives(ranges: list[RawRange]) -> tuple[str, ...]:
    nonempty = [scan for scan in ranges if scan.point_count]
    if len(nonempty) < 2:
        return ()
    names = sorted({name for scan in nonempty for name in scan.drives})
    varying: list[str] = []
    inner = {
        name
        for scan in nonempty
        for name in scan.scan_path.moving_drives
    }
    for name in names:
        values = np.asarray(
            [scan.drives.get(name, np.nan) for scan in nonempty], dtype=float
        )
        if (
            name not in inner
            and np.all(np.isfinite(values))
            and np.ptp(values) > 1e-8
        ):
            varying.append(name)
    return tuple(varying)


def _classify_file_geometry(ranges: list[RawRange]) -> MeasurementGeometry:
    nonempty = [scan for scan in ranges if scan.point_count]
    if not nonempty:
        return MeasurementGeometry("empty", (), (), "ambiguous", "no measured points")

    inner_order: list[str] = []
    for scan in nonempty:
        for drive in scan.scan_path.moving_drives:
            if drive not in inner_order:
                inner_order.append(drive)
    inner = tuple(inner_order)
    outer = _varying_outer_drives(ranges)

    sources = {scan.scan_path.source for scan in nonempty}
    if sources == {"explicit"}:
        source = "explicit"
    elif sources == {"manual"}:
        source = "manual"
    elif "ambiguous" in sources:
        source = "ambiguous"
    else:
        source = "inferred"

    if "Phi" in inner and "Chi" in outer:
        return MeasurementGeometry(
            "pole_figure", inner, outer, source, "Phi inside ranges, Chi between ranges"
        )
    if ("Theta" in inner or "Omega" in inner) and "2Theta" in outer:
        return MeasurementGeometry(
            "rsm",
            inner,
            outer,
            source,
            "rocking scan inside ranges, 2Theta between ranges",
        )
    if len(nonempty) == 1:
        kind = "coupled_scan" if len(inner) > 1 else "single_scan"
        return MeasurementGeometry(kind, inner, (), source)
    if outer:
        return MeasurementGeometry("scan_series", inner, outer, source)
    if len(inner) > 1:
        return MeasurementGeometry("coupled_scan", inner, (), source)
    return MeasurementGeometry("multi_range", inner, outer, source)


def read_bruker_raw(path: str | Path) -> BrukerRawFile:
    """Автоматически определить версию Bruker RAW и прочитать файл."""
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 8:
        raise ValueError("Файл слишком короткий: отсутствует сигнатура Bruker RAW.")
    if data[:8] == MAGIC_V3:
        raw = _read_v3(path, data)
    elif data[:8] == MAGIC_V4:
        raw = _read_v4(path, data)
    else:
        signature = data[:8].rstrip(b"\x00")
        raise ValueError(
            f"Неподдерживаемая сигнатура {signature!r}. "
            "Поддерживаются RAW1.01 и RAW4.00."
        )

    raw.geometry = _classify_file_geometry(raw.ranges)
    return raw


def read_raw1_01(path: str | Path) -> BrukerRawFile:
    """Совместимое имя функции из прежнего специализированного считывателя."""
    raw = read_bruker_raw(path)
    if raw.version != 3:
        raise ValueError(f"Ожидался RAW v3, получен RAW v{raw.version}.")
    return raw


def _read_v3(path: Path, data: bytes) -> BrukerRawFile:
    _validate_slice(data, 0, V3_FILE_HEADER_SIZE, "Заголовок RAW v3")

    status_code = _u32(data, 8)
    statuses = {
        1: "done",
        2: "active",
        3: "aborted",
        4: "interrupted",
    }
    declared_ranges = _u32(data, 12)

    metadata: dict[str, Any] = {
        "status_code": status_code,
        "status": statuses.get(status_code, f"unknown ({status_code})"),
        "declared_ranges": declared_ranges,
        "user": _text(data, 36, 72),
        "site": _text(data, 108, 218),
        "sample_id": _text(data, 326, 60),
        "comment": _text(data, 386, 160),
        "anode": _text(data, 608, 4),
        "alpha_average": _f64(data, 616),
        "alpha1": _f64(data, 624),
        "alpha2": _f64(data, 632),
        "beta": _f64(data, 640),
        "alpha_ratio": _f64(data, 648),
        "measurement_time": _f32(data, 664),
        "goniometer_code": _u32(data, 548),
        "goniometer_stage_code": _u32(data, 552),
        "sample_loader_code": _u32(data, 556),
        "goniometer_controller_code": _u32(data, 560),
        "goniometer_radius": _f32(data, 564),
        "fixed_divergence_slit": _f32(data, 568),
        "fixed_sample_slit": _f32(data, 572),
        "primary_soller_slit_code": _u32(data, 576),
        "primary_monochromator_code": _u32(data, 580),
        "fixed_antiscatter_slit": _f32(data, 584),
        "fixed_detector_slit": _f32(data, 588),
        "secondary_soller_slit_code": _u32(data, 592),
        "fixed_thin_film_attachment_code": _u32(data, 596),
        "beta_filter_code": _u32(data, 600),
        "secondary_monochromator_code": _u32(data, 604),
        "unit_name": _text(data, 656, 4),
        "intensity_beta_to_alpha1": _f32(data, 660),
        "hardware_dependency_code": int(data[711]),
        "file_header_raw_hex": data[:V3_FILE_HEADER_SIZE].hex(),
    }

    ranges: list[RawRange] = []
    offset = V3_FILE_HEADER_SIZE

    for index in range(declared_ranges):
        _validate_slice(
            data, offset, V3_RANGE_HEADER_SIZE, f"Заголовок диапазона {index}"
        )

        header_size = _u32(data, offset)
        point_count = _u32(data, offset + 4)
        datum_size = _u32(data, offset + 252)
        supplementary_size = _u32(data, offset + 256)

        if header_size != V3_RANGE_HEADER_SIZE:
            raise ValueError(
                f"Диапазон {index}: ожидался заголовок длиной "
                f"{V3_RANGE_HEADER_SIZE} байта, получено {header_size}."
            )
        if datum_size == 0:
            datum_size = 4
        if datum_size % 4:
            raise ValueError(
                f"Диапазон {index}: размер точки {datum_size} "
                "не кратен размеру float32."
            )

        supplementary_offset = offset + header_size
        _validate_slice(
            data,
            supplementary_offset,
            supplementary_size,
            f"Дополнительный заголовок диапазона {index}",
        )
        supplementary_raw = data[
            supplementary_offset : supplementary_offset + supplementary_size
        ]
        supplementary_segments: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(supplementary_raw):
            remaining = len(supplementary_raw) - cursor
            if remaining < 8:
                supplementary_segments.append(
                    {
                        "type": None,
                        "size": remaining,
                        "raw_hex": supplementary_raw[cursor:].hex(),
                        "parsed": False,
                    }
                )
                break
            segment_type = _u32(supplementary_raw, cursor)
            segment_size = _u32(supplementary_raw, cursor + 4)
            if segment_size < 8 or segment_size > remaining:
                supplementary_segments.append(
                    {
                        "type": segment_type,
                        "size": remaining,
                        "raw_hex": supplementary_raw[cursor:].hex(),
                        "parsed": False,
                    }
                )
                break
            segment = supplementary_raw[cursor : cursor + segment_size]
            record: dict[str, Any] = {
                "type": segment_type,
                "size": segment_size,
                "raw_hex": segment.hex(),
                "parsed": True,
            }
            if segment_size >= 16:
                record["payload_float64_0"] = _f64(segment, 8)
            supplementary_segments.append(record)
            cursor += segment_size

        data_offset = supplementary_offset + supplementary_size
        byte_count = point_count * datum_size
        _validate_slice(data, data_offset, byte_count, f"Данные диапазона {index}")

        channel_count = datum_size // 4
        values = np.frombuffer(
            data,
            dtype="<f4",
            count=point_count * channel_count,
            offset=data_offset,
        ).astype(float, copy=True)
        values = values.reshape(point_count, channel_count)

        drives = {
            "Theta": _f64(data, offset + 8),
            "2Theta": _f64(data, offset + 16),
            "Chi": _f64(data, offset + 24),
            "Phi": _f64(data, offset + 32),
            "X-Drive": _f64(data, offset + 40),
            "Y-Drive": _f64(data, offset + 48),
            "Z-Drive": _f64(data, offset + 56),
        }
        step_size = _f64(data, offset + 176)
        scan_axis_code = _u32(data, offset + 196)
        scan_axis = V3_SCAN_AXIS_CODES.get(scan_axis_code)
        if scan_axis is None:
            scan_type = f"Unknown RAW v3 scan (axis code {scan_axis_code})"
            axis_name = "Point"
            start_angle = 0.0
            displayed_step = 1.0
            scan_path = ScanPath(
                None,
                {},
                "ambiguous",
                f"unrecognised RAW v3 scan-axis code {scan_axis_code}",
            )
        else:
            scan_type = f"{scan_axis} Scan"
            axis_name = scan_axis
            start_angle = drives[scan_axis]
            displayed_step = step_size
            scan_path = ScanPath(
                scan_axis,
                {scan_axis: step_size},
                "explicit",
                f"RAW v3 scan-axis code {scan_axis_code}",
            )

        ranges.append(
            RawRange(
                index=index,
                scan_type=scan_type,
                axis_name=axis_name,
                start_angle=start_angle,
                step_size=displayed_step,
                time_per_step=_f32(data, offset + 192),
                drives=drives,
                data=values,
                channel_names=_channel_names([], channel_count),
                generator_voltage=float(_u32(data, offset + 224)),
                generator_current=float(_u32(data, offset + 228)),
                wavelength=_f64(data, offset + 240),
                metadata={
                    "header_size": header_size,
                    "datum_size": datum_size,
                    "supplementary_header_size": supplementary_size,
                    "scan_axis_code": scan_axis_code,
                    "scan_axis_known": scan_axis is not None,
                    "nominal_step_size": step_size,
                    "detector_code": _u32(data, offset + 96),
                    "high_voltage": _f32(data, offset + 100),
                    "amplifier_gain": _f32(data, offset + 104),
                    "discriminator_1_lower_level": _f32(data, offset + 108),
                    "rotation_speed_rpm": _f32(data, offset + 208),
                    "range_header_raw_hex": data[
                        offset : offset + V3_RANGE_HEADER_SIZE
                    ].hex(),
                    "supplementary_header_raw_hex": supplementary_raw.hex(),
                    "supplementary_segments": supplementary_segments,
                    "uninterpreted_fields": {
                        "168_u32": _u32(data, offset + 168),
                        "184_hex": data[offset + 184 : offset + 192].hex(),
                        "200_u32": _u32(data, offset + 200),
                        "204_hex": data[offset + 204 : offset + 208].hex(),
                        "212_hex": data[offset + 212 : offset + 224].hex(),
                        "232_hex": data[offset + 232 : offset + 240].hex(),
                        "248_hex": data[offset + 248 : offset + 252].hex(),
                        "260_hex": data[offset + 260 : offset + 280].hex(),
                        "280_hex": data[offset + 280 : offset + 304].hex(),
                    },
                },
                scan_path=scan_path,
            )
        )
        offset = data_offset + byte_count

    trailing = data[offset:]
    if trailing and any(byte != 0 for byte in trailing):
        raise ValueError(
            f"После последнего диапазона осталось {len(trailing)} "
            "ненулевых лишних байт."
        )

    return BrukerRawFile(
        path=path,
        version=3,
        magic="RAW1.01",
        date=_text(data, 16, 10),
        time=_text(data, 26, 10),
        metadata=metadata,
        ranges=ranges,
    )


def _read_v4(path: Path, data: bytes) -> BrukerRawFile:
    _validate_slice(data, 0, 61, "Заголовок RAW v4")

    date = _text(data, 12, 12)
    time = _text(data, 24, 10)
    metadata: dict[str, Any] = {
        "variables": {},
        "drive_alignments": [],
        "global_segments": [],
    }

    offset = 61
    range_marker: int | None = None

    while offset < len(data):
        _validate_slice(data, offset, 4, "Тип глобального сегмента RAW v4")
        segment_type = _u32(data, offset)
        if segment_type in (0, 160):
            range_marker = segment_type
            break

        _validate_slice(data, offset, 8, "Заголовок глобального сегмента RAW v4")
        segment_size = _u32(data, offset + 4)
        if segment_size < 8:
            raise ValueError(
                f"Глобальный сегмент при смещении {offset}: "
                f"недопустимая длина {segment_size}."
            )
        _validate_slice(
            data, offset, segment_size, f"Глобальный сегмент при смещении {offset}"
        )
        metadata["global_segments"].append(
            {"type": segment_type, "size": segment_size}
        )

        if segment_type == 10:
            if segment_size < 36:
                raise ValueError("Сегмент VarInfo RAW v4 короче 36 байт.")
            name = _text(data, offset + 12, 24)
            value = _text(data, offset + 36, segment_size - 36)
            metadata["variables"][name] = value
        elif segment_type == 30:
            if segment_size < 120:
                raise ValueError("Сегмент HardwareConfiguration короче 120 байт.")
            metadata.update(
                {
                    "alpha_average": _f64(data, offset + 72),
                    "alpha1": _f64(data, offset + 80),
                    "alpha2": _f64(data, offset + 88),
                    "beta": _f64(data, offset + 96),
                    "alpha_ratio": _f64(data, offset + 104),
                    "anode": _text(data, offset + 116, 4),
                }
            )
        elif segment_type == 60:
            if segment_size < 76:
                raise ValueError("Сегмент DriveAlignment короче 76 байт.")
            metadata["drive_alignments"].append(
                {
                    "name": _text(data, offset + 12, 24),
                    "flag": _u32(data, offset + 8),
                    "delta": _f64(data, offset + 68),
                }
            )

        offset += segment_size

    variables = metadata["variables"]
    metadata.update(
        {
            "user": variables.get("USER", ""),
            "site": variables.get("SITE", ""),
            "sample_id": variables.get("SAMPLEID", ""),
            "comment": variables.get("COMMENT", ""),
            "creator": variables.get("CREATOR", ""),
        }
    )

    ranges: list[RawRange] = []
    while offset < len(data):
        _validate_slice(data, offset, 160, f"Основной заголовок диапазона {len(ranges)}")
        range_marker = _u32(data, offset)
        if range_marker not in (0, 160):
            raise ValueError(
                f"При смещении {offset} ожидался диапазон RAW v4, "
                f"получен сегмент типа {range_marker}."
            )

        index = len(ranges)
        scan_type = _text(data, offset + 32, 24) or "Unknown"
        start_angle = _f64(data, offset + 72)
        step_size = _f64(data, offset + 80)
        point_count = _u32(data, offset + 88)
        time_per_step = _f32(data, offset + 92)
        generator_voltage = _f32(data, offset + 100)
        generator_current = _f32(data, offset + 104)
        wavelength = _f64(data, offset + 112)
        datum_size = _u32(data, offset + 136)
        extended_size = _u32(data, offset + 140)

        if datum_size == 0 or datum_size % 4:
            raise ValueError(
                f"Диапазон {index}: размер точки {datum_size} "
                "не является положительным кратным float32."
            )

        extended_offset = offset + 160
        extended_end = extended_offset + extended_size
        _validate_slice(
            data,
            extended_offset,
            extended_size,
            f"Дополнительные заголовки диапазона {index}",
        )

        drives: dict[str, float] = {}
        detector_names: list[str] = []
        segments: list[dict[str, Any]] = []
        cursor = extended_offset

        while cursor < extended_end:
            _validate_slice(
                data, cursor, 8, f"Сегмент дополнительного заголовка диапазона {index}"
            )
            segment_type = _u32(data, cursor)
            segment_size = _u32(data, cursor + 4)
            if segment_size < 8 or cursor + segment_size > extended_end:
                raise ValueError(
                    f"Диапазон {index}: недопустимый дополнительный сегмент "
                    f"типа {segment_type}, длина {segment_size}."
                )

            segment: dict[str, Any] = {
                "type": segment_type,
                "size": segment_size,
            }
            if segment_type == 50 and segment_size >= 64:
                name = _text(data, cursor + 12, 24)
                value = _f64(data, cursor + 56)
                segment.update({"name": name, "value": value})
                if name:
                    drives[name] = value
            elif segment_type == 40 and segment_size >= 36:
                name = _text(data, cursor + 12, 24)
                segment["name"] = name
                detector_names.append(name)
            elif segment_size >= 36:
                name = _text(data, cursor + 12, 24)
                if name:
                    segment["name"] = name

            segments.append(segment)
            cursor += segment_size

        if cursor != extended_end:
            raise ValueError(f"Диапазон {index}: нарушена граница заголовков.")

        channel_count = datum_size // 4
        data_offset = extended_end
        byte_count = point_count * datum_size
        _validate_slice(data, data_offset, byte_count, f"Данные диапазона {index}")
        values = np.frombuffer(
            data,
            dtype="<f4",
            count=point_count * channel_count,
            offset=data_offset,
        ).astype(float, copy=True)
        values = values.reshape(point_count, channel_count)

        scan_path = _infer_v4_scan_path(
            scan_type, start_angle, step_size, drives
        )
        axis_name = scan_path.primary_drive or "ScanAxis"
        if scan_path.primary_drive in drives:
            start_angle = drives[scan_path.primary_drive]

        ranges.append(
            RawRange(
                index=index,
                scan_type=scan_type,
                axis_name=axis_name,
                start_angle=start_angle,
                step_size=step_size,
                time_per_step=time_per_step,
                drives=drives,
                data=values,
                channel_names=_channel_names(detector_names, channel_count),
                generator_voltage=generator_voltage,
                generator_current=generator_current,
                wavelength=wavelength,
                metadata={
                    "range_marker": range_marker,
                    "datum_size": datum_size,
                    "extended_header_size": extended_size,
                    "segments": segments,
                },
                scan_path=scan_path,
            )
        )
        offset = data_offset + byte_count

    if not ranges and range_marker is None:
        raise ValueError("В RAW v4 не найдено ни одного измерительного диапазона.")

    return BrukerRawFile(
        path=path,
        version=4,
        magic="RAW4.00",
        date=date,
        time=time,
        metadata=metadata,
        ranges=ranges,
    )


def _matching_start_drives(
    start_angle: float, drives: dict[str, float]
) -> list[str]:
    return [
        name
        for name, value in drives.items()
        if np.isfinite(value)
        and np.isclose(float(value), float(start_angle), rtol=0.0, atol=1e-6)
    ]


def _named_drive(scan_type: str, drives: dict[str, float]) -> str | None:
    lower = scan_type.casefold().replace("θ", "theta").replace("ω", "omega")
    for token, drive in (
        ("2theta", "2Theta"),
        ("two theta", "2Theta"),
        ("omega", "Omega"),
        ("theta", "Theta"),
        ("chi", "Chi"),
        ("phi", "Phi"),
        ("x-drive", "X-Drive"),
        ("y-drive", "Y-Drive"),
        ("z-drive", "Z-Drive"),
    ):
        if token in lower and drive in drives:
            return drive
    return None


def _infer_v4_scan_path(
    scan_type: str,
    start_angle: float,
    step_size: float,
    drives: dict[str, float],
) -> ScanPath:
    """Construct a simple or coupled motion path from RAW v4 metadata."""
    lower = scan_type.casefold()
    matches = _matching_start_drives(start_angle, drives)

    # "unlocked coupled" contains "locked coupled" and must be checked first.
    if "unlocked coupled" in lower:
        primary = (
            "2Theta"
            if "2Theta" in matches
            else (matches[0] if len(matches) == 1 else None)
        )
        if primary:
            return ScanPath(
                primary,
                {primary: step_size},
                "explicit",
                "RAW v4 Unlocked Coupled: secondary ratio is not encoded",
            )

    if "locked coupled" in lower:
        primary = "2Theta" if "2Theta" in drives else (matches[0] if matches else None)
        if primary == "Theta":
            steps = {"Theta": step_size, "2Theta": 2.0 * step_size}
        else:
            primary = primary or "2Theta"
            steps = {"2Theta": step_size, "Theta": 0.5 * step_size}
        return ScanPath(
            primary,
            {name: step for name, step in steps.items() if name in drives},
            "explicit",
            "RAW v4 Locked Coupled scan",
        )

    if "rocking curve" in lower:
        primary = next(
            (name for name in ("Theta", "Omega", "2Theta") if name in matches),
            None,
        )
        if primary is None:
            primary = next(
                (name for name in ("Theta", "Omega", "2Theta") if name in drives),
                None,
            )
        if primary:
            return ScanPath(
                primary,
                {primary: step_size},
                "explicit",
                f"RAW v4 scan type {scan_type!r}",
            )

    if "pole figure" in lower and "Phi" in drives:
        return ScanPath(
            "Phi",
            {"Phi": step_size},
            "explicit",
            f"RAW v4 scan type {scan_type!r}",
        )

    named = _named_drive(scan_type, drives)
    if named:
        return ScanPath(
            named,
            {named: step_size},
            "explicit",
            f"drive named by RAW v4 scan type {scan_type!r}",
        )

    if len(matches) == 1:
        primary = matches[0]
        return ScanPath(
            primary,
            {primary: step_size},
            "inferred",
            "unique drive start matches the RAW v4 scan start",
        )

    return ScanPath(
        None,
        {},
        "ambiguous",
        "RAW v4 scan path could not be determined unambiguously",
    )


def pole_grid(
    raw: BrukerRawFile, channel: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вернуть оси φ, χ и матрицу полюсной фигуры.

    Незаписанные элементы остаются NaN. Интерполяция не выполняется.
    """
    if not raw.is_pole_figure:
        raise ValueError("Файл не распознан как полюсная фигура.")

    nonempty = [scan for scan in raw.ranges if scan.point_count]
    if not nonempty:
        raise ValueError("Во всех диапазонах отсутствуют точки.")
    for scan in nonempty:
        if channel >= scan.channel_count:
            raise ValueError(
                f"Диапазон {scan.index}: отсутствует канал с номером {channel}."
            )

    phi_start = nonempty[0].start_angle
    step = nonempty[0].step_size
    for scan in nonempty[1:]:
        if not np.isclose(scan.start_angle, phi_start):
            raise ValueError("Начальные значения φ в диапазонах различаются.")
        if not np.isclose(scan.step_size, step):
            raise ValueError("Шаг φ в диапазонах различается.")

    point_count = max(scan.point_count for scan in raw.ranges)
    phi = phi_start + step * np.arange(point_count)
    chi = np.array([scan.drives["Chi"] for scan in raw.ranges], dtype=float)
    intensity = np.full((len(raw.ranges), point_count), np.nan)

    for row, scan in enumerate(raw.ranges):
        if scan.point_count:
            intensity[row, : scan.point_count] = scan.data[:, channel]

    order = np.argsort(chi)
    return phi, chi[order], intensity[order]


def export_csv(
    raw: BrukerRawFile,
    output: str | Path,
    range_index: int | None = None,
    channel: int = 0,
) -> None:
    """
    Выгрузить данные в CSV.

    Полюсная фигура без выбранного диапазона:
        chi_deg, phi_deg, intensity

    Один выбранный или единственный диапазон:
        сканируемая ось и все измерительные каналы

    Несколько остальных диапазонов:
        range, x, intensity и начальные положения приводов
    """
    output = Path(output)
    selected = _select_ranges(raw, range_index)

    if raw.is_pole_figure and range_index is None:
        phi, chi, intensity = pole_grid(raw, channel)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["chi_deg", "phi_deg", "intensity"])
            for row, chi_value in enumerate(chi):
                valid = np.isfinite(intensity[row])
                for phi_value, value in zip(phi[valid], intensity[row, valid]):
                    writer.writerow((f"{chi_value:.12g}", f"{phi_value:.12g}", f"{value:.12g}"))
        return

    if len(selected) == 1:
        scan = selected[0]
        table = np.column_stack((scan.axis, scan.data))
        axis_label = f"{scan.axis_name}_deg"
        header = ",".join([axis_label, *scan.channel_names])
        np.savetxt(
            output,
            table,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.12g",
        )
        return

    drive_names = sorted({name for scan in selected for name in scan.drives})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["range", "scan_axis", "x", "intensity", *drive_names]
        )
        for scan in selected:
            if channel >= scan.channel_count:
                raise ValueError(
                    f"Диапазон {scan.index}: отсутствует канал {channel}."
                )
            drive_coordinates = {
                name: scan.coordinate(name) for name in drive_names
            }
            for point_index, (point, value) in enumerate(
                zip(scan.axis, scan.data[:, channel])
            ):
                coordinates = [
                    drive_coordinates[name][point_index]
                    for name in drive_names
                ]
                writer.writerow(
                    [
                        scan.index,
                        scan.axis_name,
                        f"{point:.12g}",
                        f"{value:.12g}",
                        *[f"{coordinate:.12g}" for coordinate in coordinates],
                    ]
                )


def plot_raw(
    raw: BrukerRawFile,
    output: str | Path | None = None,
    range_index: int | None = None,
    channel: int = 0,
    logarithmic: bool = False,
) -> None:
    """Построить обычный скан либо полюсную фигуру."""
    import matplotlib.pyplot as plt

    if raw.is_pole_figure and range_index is None:
        _plot_pole_figure(raw, output, channel, logarithmic)
        return

    selected = _select_ranges(raw, range_index)
    figure, axis = plt.subplots(figsize=(8, 5))
    for scan in selected:
        if channel >= scan.channel_count:
            raise ValueError(
                f"Диапазон {scan.index}: отсутствует канал {channel}."
            )
        label = scan.scan_type if len(selected) == 1 else f"Диапазон {scan.index}"
        axis.plot(scan.axis, scan.data[:, channel], label=label)

    x_names = {scan.axis_name for scan in selected}
    axis.set_xlabel(
        f"{next(iter(x_names))}, °" if len(x_names) == 1 else "Координата скана"
    )
    axis.set_ylabel("Интенсивность, отсчёты")
    axis.grid(alpha=0.25)
    if logarithmic:
        axis.set_yscale("log")
    if len(selected) > 1:
        axis.legend()
    axis.set_title(raw.path.name)
    figure.tight_layout()

    if output is None:
        plt.show()
    else:
        figure.savefig(output, dpi=200)
        plt.close(figure)


def _plot_pole_figure(
    raw: BrukerRawFile,
    output: str | Path | None,
    channel: int,
    logarithmic: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    phi, chi, intensity = pole_grid(raw, channel)
    masked = np.ma.masked_invalid(intensity)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    norm = None
    if logarithmic:
        masked = np.ma.masked_less_equal(masked, 0)
        positive = masked.compressed()
        if positive.size == 0:
            raise ValueError("Для логарифмической шкалы нет положительных значений.")
        norm = LogNorm(vmin=max(1.0, positive.min()), vmax=positive.max())

    figure, axis = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    mesh = axis.pcolormesh(
        np.deg2rad(phi),
        chi,
        masked,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0, np.nanmax(chi))
    first = next(scan for scan in raw.ranges if scan.point_count)
    axis.set_title(
        f"{raw.path.name}\n2θ = {first.two_theta:g}°, θ = {first.theta:g}°"
    )
    axis.set_ylabel("χ, °")
    figure.colorbar(mesh, ax=axis, pad=0.1, label="Интенсивность, отсчёты")
    figure.tight_layout()

    if output is None:
        plt.show()
    else:
        figure.savefig(output, dpi=200)
        plt.close(figure)


def _select_ranges(
    raw: BrukerRawFile, range_index: int | None
) -> list[RawRange]:
    if range_index is None:
        return raw.ranges
    if range_index < 0 or range_index >= len(raw.ranges):
        raise ValueError(
            f"Диапазон {range_index} отсутствует; допустимы номера "
            f"0–{len(raw.ranges) - 1}."
        )
    return [raw.ranges[range_index]]


def print_summary(raw: BrukerRawFile) -> None:
    metadata = raw.metadata
    print(f"Файл: {raw.path}")
    print(f"Формат: Siemens/Bruker {raw.magic} (RAW v{raw.version})")
    print(f"Дата и время: {raw.date} {raw.time}")
    if metadata.get("status"):
        print(f"Состояние: {metadata['status']}")
    if metadata.get("user"):
        print(f"Пользователь: {metadata['user']}")
    if metadata.get("site"):
        print(f"Место: {metadata['site']}")
    if metadata.get("creator"):
        print(f"Создан: {metadata['creator']}")
    if metadata.get("anode"):
        wavelength = metadata.get("alpha1")
        suffix = f"; Kα1 = {wavelength:g} Å" if wavelength is not None else ""
        print(f"Анод: {metadata['anode']}{suffix}")
    print(f"Диапазонов: {len(raw.ranges)}")
    print(f"Тип измерения: {raw.measurement_type}")

    if raw.is_pole_figure:
        sizes = np.array([scan.point_count for scan in raw.ranges])
        maximum = int(sizes.max(initial=0))
        full = int(np.count_nonzero(sizes == maximum)) if maximum else 0
        partial = int(np.count_nonzero((sizes > 0) & (sizes < maximum)))
        empty = int(np.count_nonzero(sizes == 0))
        first = next(scan for scan in raw.ranges if scan.point_count)
        print(
            f"Полных диапазонов: {full}; частичных: {partial}; "
            f"пустых: {empty}; точек в полном диапазоне: {maximum}"
        )
        print(
            f"2θ = {first.two_theta:g}°; θ = {first.theta:g}°; "
            f"χ = {min(scan.chi for scan in raw.ranges):g}–"
            f"{max(scan.chi for scan in raw.ranges):g}°"
        )
        print(
            f"φ: начало {first.start_angle:g}°, шаг {first.step_size:g}°"
        )
        return

    if raw.is_rsm:
        nonempty = [scan for scan in raw.ranges if scan.point_count]
        sizes = np.asarray([scan.point_count for scan in raw.ranges], dtype=int)
        maximum = int(sizes.max(initial=0))
        partial = int(np.count_nonzero((sizes > 0) & (sizes < maximum)))
        empty = int(np.count_nonzero(sizes == 0))
        two_theta = np.asarray(
            [scan.drives["2Theta"] for scan in nonempty], dtype=float
        )
        print(
            f"RSM: {len(nonempty)} записанных диапазонов; "
            f"частичных: {partial}; пустых: {empty}; "
            f"2Theta = {two_theta.min():g}–{two_theta.max():g}°"
        )

    shown = raw.ranges if len(raw.ranges) <= 10 else raw.ranges[:5] + raw.ranges[-2:]
    for scan in shown:
        end = float(scan.axis[-1]) if scan.point_count else scan.start_angle
        channels = ", ".join(scan.channel_names) or "-"
        print(
            f"Диапазон {scan.index}: {scan.scan_type}; ось {scan.axis_name}; "
            f"{scan.start_angle:g}–{end:g}°; шаг {scan.step_size:g}°; "
            f"точек {scan.point_count}; каналы: {channels}"
        )
        fixed = [
            f"{name}={value:g}°"
            for name, value in scan.drives.items()
            if name not in scan.scan_path.moving_drives and np.isfinite(value)
        ]
        if fixed:
            print("  Постоянные координаты: " + "; ".join(fixed))
    if len(raw.ranges) > len(shown):
        print(f"  … пропущено диапазонов: {len(raw.ranges) - len(shown)}")


def raw_metadata_report(raw: BrukerRawFile) -> dict[str, Any]:
    """Build a JSON-ready diagnostic report without intensity arrays."""
    ranges: list[dict[str, Any]] = []
    for scan in raw.ranges:
        ranges.append(
            {
                "index": scan.index,
                "scan_type": scan.scan_type,
                "axis_name": scan.axis_name,
                "axis_start": float(scan.axis[0]) if scan.point_count else None,
                "axis_end": float(scan.axis[-1]) if scan.point_count else None,
                "point_count": int(scan.point_count),
                "channel_count": int(scan.channel_count),
                "channel_names": scan.channel_names,
                "step_size": scan.step_size,
                "time_per_step": scan.time_per_step,
                "drives": scan.drives,
                "generator_voltage": scan.generator_voltage,
                "generator_current": scan.generator_current,
                "wavelength": scan.wavelength,
                "scan_path": {
                    "primary_drive": scan.scan_path.primary_drive,
                    "drive_steps": scan.scan_path.drive_steps,
                    "source": scan.scan_path.source,
                    "reason": scan.scan_path.reason,
                },
                "metadata": scan.metadata,
            }
        )
    return {
        "path": str(raw.path),
        "version": raw.version,
        "magic": raw.magic,
        "date": raw.date,
        "time": raw.time,
        "measurement_type": raw.measurement_type,
        "geometry": {
            "kind": raw.geometry.kind,
            "inner_drives": raw.geometry.inner_drives,
            "outer_drives": raw.geometry.outer_drives,
            "source": raw.geometry.source,
            "reason": raw.geometry.reason,
        },
        "metadata": raw.metadata,
        "ranges": ranges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Чтение Siemens/Bruker RAW v3 и RAW v4."
    )
    parser.add_argument("raw", type=Path, help="исходный файл .raw")
    parser.add_argument("--csv", type=Path, help="выгрузить данные в CSV")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="show",
        help="показать график либо сохранить его в указанный PNG/PDF/SVG",
    )
    parser.add_argument(
        "--range", type=int, dest="range_index", help="номер отдельного диапазона"
    )
    parser.add_argument(
        "--channel", type=int, default=0, help="номер измерительного канала"
    )
    parser.add_argument(
        "--log", action="store_true", help="логарифмическая шкала интенсивности"
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_output",
        help=(
            "сохранить метаданные и исходные заголовки в JSON "
            "без массивов интенсивности"
        ),
    )
    args = parser.parse_args()

    raw = read_bruker_raw(args.raw)
    print_summary(raw)

    if args.csv:
        export_csv(raw, args.csv, args.range_index, args.channel)
        print(f"CSV сохранён: {args.csv}")

    if args.json_output:
        args.json_output.write_text(
            json.dumps(raw_metadata_report(raw), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON сохранён: {args.json_output}")

    if args.plot:
        output = None if args.plot == "show" else Path(args.plot)
        plot_raw(raw, output, args.range_index, args.channel, args.log)
        if output is not None:
            print(f"График сохранён: {output}")


if __name__ == "__main__":
    main()
