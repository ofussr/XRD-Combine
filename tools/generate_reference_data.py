#!/usr/bin/env python3
"""Regenerate compact runtime reference tables from upstream datasets.

This developer-only helper is not imported by XRD Combine.  It deliberately
keeps the heavy source packages out of the application and its frozen build.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import sys


def generate_atom_styles(database: Path, output: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT atomic_number, symbol, jmol_color, cpk_color,
                   molcas_gv_color, covalent_radius_pyykko,
                   covalent_radius_cordero
            FROM elements ORDER BY atomic_number
            """
        ).fetchall()
    finally:
        connection.close()
    data = {
        "source": {
            "project": "mendeleev",
            "version": "1.2.0",
            "url": "https://github.com/lmmentel/mendeleev",
            "license": "MIT",
        },
        "units": {"covalent_radii": "pm"},
        "elements": {
            symbol: {
                "atomic_number": number,
                "jmol": jmol,
                "cpk": cpk,
                "molcas_gv": molcas,
                "covalent_radius_pyykko": pyykko,
                "covalent_radius_cordero": cordero,
            }
            for number, symbol, jmol, cpk, molcas, pyykko, cordero in rows
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _term(row: list[int], translation: float) -> str:
    variables = "xyz"
    parts: list[str] = []
    for coefficient, variable in zip(row, variables):
        if coefficient == 0:
            continue
        if coefficient == 1:
            token = variable
        elif coefficient == -1:
            token = f"-{variable}"
        else:
            token = f"{coefficient}*{variable}"
        if parts and not token.startswith("-"):
            token = "+" + token
        parts.append(token)
    fraction = Fraction(float(translation) % 1.0).limit_denominator(48)
    if fraction:
        token = (
            str(fraction.numerator)
            if fraction.denominator == 1
            else f"{fraction.numerator}/{fraction.denominator}"
        )
        if parts:
            token = "+" + token
        parts.append(token)
    return "".join(parts) or "0"


def generate_space_groups(spglib_path: Path, output: Path) -> None:
    sys.path.insert(0, str(spglib_path))
    import spglib  # type: ignore[import-not-found]

    settings = []
    for hall_number in range(1, 531):
        kind = spglib.get_spacegroup_type(hall_number)
        symmetry = spglib.get_symmetry_from_database(hall_number)
        if kind is None or symmetry is None:
            raise RuntimeError(f"No spglib data for Hall number {hall_number}")
        rotations = symmetry["rotations"].tolist()
        translations = symmetry["translations"].tolist()
        operations = [
            ",".join(_term(list(row), shift) for row, shift in zip(rotation, translation))
            for rotation, translation in zip(rotations, translations)
        ]
        settings.append(
            {
                "hall_number": hall_number,
                "number": int(kind.number),
                "international_short": kind.international_short,
                "international_full": kind.international_full,
                "choice": kind.choice,
                "hall_symbol": kind.hall_symbol,
                "operations": operations,
            }
        )
    data = {
        "source": {
            "project": "spglib",
            "version": spglib.__version__,
            "url": "https://github.com/spglib/spglib",
            "license": "BSD-3-Clause",
        },
        "settings": settings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mendeleev-db", type=Path, required=True)
    parser.add_argument("--spglib-path", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    arguments = parser.parse_args()
    generate_atom_styles(
        arguments.mendeleev_db,
        arguments.resources / "atom_styles.json",
    )
    generate_space_groups(
        arguments.spglib_path,
        arguments.resources / "space_groups.json",
    )


if __name__ == "__main__":
    main()
