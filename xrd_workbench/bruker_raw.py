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
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MAGIC_V3 = b"RAW1.01\x00"
MAGIC_V4 = b"RAW4.00\x00"
V3_FILE_HEADER_SIZE = 712
V3_RANGE_HEADER_SIZE = 304


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

    @property
    def point_count(self) -> int:
        return self.data.shape[0]

    @property
    def channel_count(self) -> int:
        return self.data.shape[1]

    @property
    def axis(self) -> np.ndarray:
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
        if drive_name == self.axis_name:
            return self.axis
        return np.full(self.point_count, self.drives.get(drive_name, np.nan))


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

    @property
    def is_pole_figure(self) -> bool:
        nonempty = [scan for scan in self.ranges if scan.point_count]
        return (
            len(self.ranges) > 1
            and bool(nonempty)
            and all(scan.axis_name == "Phi" for scan in nonempty)
            and all("Chi" in scan.drives for scan in self.ranges)
        )


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


def read_bruker_raw(path: str | Path) -> BrukerRawFile:
    """Автоматически определить версию Bruker RAW и прочитать файл."""
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 8:
        raise ValueError("Файл слишком короткий: отсутствует сигнатура Bruker RAW.")
    if data[:8] == MAGIC_V3:
        return _read_v3(path, data)
    if data[:8] == MAGIC_V4:
        return _read_v4(path, data)

    signature = data[:8].rstrip(b"\x00")
    raise ValueError(
        f"Неподдерживаемая сигнатура {signature!r}. "
        "Поддерживаются RAW1.01 и RAW4.00."
    )


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

        data_offset = offset + header_size + supplementary_size
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

        ranges.append(
            RawRange(
                index=index,
                scan_type="Unknown RAW v3 range",
                axis_name="2Theta",
                start_angle=drives["2Theta"],
                step_size=step_size,
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
                },
            )
        )
        offset = data_offset + byte_count

    trailing = data[offset:]
    if trailing and any(byte != 0 for byte in trailing):
        raise ValueError(
            f"После последнего диапазона осталось {len(trailing)} "
            "ненулевых лишних байт."
        )

    _classify_v3_ranges(ranges)

    return BrukerRawFile(
        path=path,
        version=3,
        magic="RAW1.01",
        date=_text(data, 16, 10),
        time=_text(data, 26, 10),
        metadata=metadata,
        ranges=ranges,
    )


def _classify_v3_ranges(ranges: list[RawRange]) -> None:
    """
    Распознать характерную многодиапазонную полюсную съёмку RAW v3.

    Признаки намеренно строгие: постоянные θ и 2θ, меняющийся χ и полный
    оборот φ внутри диапазона. Для остальных измерений ось остаётся 2Theta,
    как в опубликованном считывателе xylib.
    """
    nonempty = [scan for scan in ranges if scan.point_count]
    if len(ranges) < 2 or not nonempty:
        return

    theta = np.array([scan.drives["Theta"] for scan in ranges])
    two_theta = np.array([scan.drives["2Theta"] for scan in ranges])
    chi = np.array([scan.drives["Chi"] for scan in ranges])
    phi_start = np.array([scan.drives["Phi"] for scan in ranges])
    max_points = max(scan.point_count for scan in nonempty)
    reference = max(nonempty, key=lambda scan: scan.point_count)
    phi_span = reference.step_size * max(0, max_points - 1)
    phi_coverage = reference.step_size * max_points

    is_pole = (
        np.allclose(theta, theta[0])
        and np.allclose(two_theta, two_theta[0])
        and np.ptp(chi) > 0
        and np.allclose(phi_start, phi_start[0])
        and (
            np.isclose(
                abs(phi_span), 360.0, atol=max(0.1, abs(reference.step_size))
            )
            or np.isclose(
                abs(phi_coverage), 360.0, atol=max(0.1, abs(reference.step_size))
            )
        )
    )
    if not is_pole:
        return

    for scan in ranges:
        scan.scan_type = "Pole Figure"
        scan.axis_name = "Phi"
        scan.start_angle = scan.drives["Phi"]
        scan.metadata["axis_inference"] = "pole-figure geometry"


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

        axis_name = _infer_v4_axis(scan_type, start_angle, drives)
        if axis_name in drives:
            start_angle = drives[axis_name]

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


def _infer_v4_axis(
    scan_type: str, start_angle: float, drives: dict[str, float]
) -> str:
    """Определить сканируемый привод по типу скана и начальному углу."""
    priorities: dict[str, list[str]] = {
        "Rocking Curve": ["Theta", "Omega", "2Theta"],
        "Pole Figure": ["Phi", "Chi", "Theta", "2Theta"],
        "Locked Coupled": ["2Theta", "Theta"],
        "Unlocked Coupled": ["2Theta", "Theta"],
    }
    ordered = priorities.get(scan_type, []) + [
        "2Theta",
        "Theta",
        "Omega",
        "Chi",
        "Phi",
        "X-Drive",
        "Y-Drive",
        "Z-Drive",
    ]

    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        if name in drives and np.isclose(
            drives[name], start_angle, rtol=0.0, atol=1e-6
        ):
            return name

    fallback = priorities.get(scan_type, [])
    for name in fallback:
        if name in drives:
            return name
    return "ScanAxis"


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
            for point, value in zip(scan.axis, scan.data[:, channel]):
                coordinates = [
                    point if name == scan.axis_name else scan.drives.get(name, np.nan)
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

    if raw.is_pole_figure:
        sizes = np.array([scan.point_count for scan in raw.ranges])
        maximum = int(sizes.max(initial=0))
        full = int(np.count_nonzero(sizes == maximum)) if maximum else 0
        partial = int(np.count_nonzero((sizes > 0) & (sizes < maximum)))
        empty = int(np.count_nonzero(sizes == 0))
        first = next(scan for scan in raw.ranges if scan.point_count)
        print("Тип измерения: полюсная фигура")
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

    shown = raw.ranges if len(raw.ranges) <= 10 else raw.ranges[:5] + raw.ranges[-2:]
    for scan in shown:
        end = (
            scan.start_angle + scan.step_size * (scan.point_count - 1)
            if scan.point_count
            else scan.start_angle
        )
        channels = ", ".join(scan.channel_names) or "-"
        print(
            f"Диапазон {scan.index}: {scan.scan_type}; ось {scan.axis_name}; "
            f"{scan.start_angle:g}–{end:g}°; шаг {scan.step_size:g}°; "
            f"точек {scan.point_count}; каналы: {channels}"
        )
        fixed = [
            f"{name}={value:g}°"
            for name, value in scan.drives.items()
            if name != scan.axis_name and np.isfinite(value)
        ]
        if fixed:
            print("  Постоянные координаты: " + "; ".join(fixed))
    if len(raw.ranges) > len(shown):
        print(f"  … пропущено диапазонов: {len(raw.ranges) - len(shown)}")


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
    args = parser.parse_args()

    raw = read_bruker_raw(args.raw)
    print_summary(raw)

    if args.csv:
        export_csv(raw, args.csv, args.range_index, args.channel)
        print(f"CSV сохранён: {args.csv}")

    if args.plot:
        output = None if args.plot == "show" else Path(args.plot)
        plot_raw(raw, output, args.range_index, args.channel, args.log)
        if output is not None:
            print(f"График сохранён: {output}")


if __name__ == "__main__":
    main()
