"""Toolkit-independent CIF tokenisation and basic data-block parsing."""

from __future__ import annotations

from pathlib import Path

from ..cif_lexer import CifLexError, tokenize_cif_text
from ..models.crystal import CifData, CifLoop
from ..models.data_errors import XRDDataError


def tokenize_cif(text: str) -> list[str]:
    """Tokenize CIF 1.1 text and report a structured parse error."""

    try:
        return tokenize_cif_text(text)
    except CifLexError as exc:
        raise XRDDataError(
            "cif_lex",
            line_number=exc.line_number,
            reason=exc.reason,
        ) from exc


def read_cif_data(path: str | Path) -> CifData:
    """Read scalar values and loops without importing a GUI toolkit."""

    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    tokens = tokenize_cif(text)
    values: dict[str, str] = {}
    loops: list[CifLoop] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        low = token.lower()
        if low == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index])
                index += 1
            if not tags:
                raise XRDDataError("cif_loop_no_columns")

            raw: list[str] = []
            while index < len(tokens):
                next_low = tokens[index].lower()
                if (
                    tokens[index].startswith("_")
                    or next_low == "loop_"
                    or next_low == "stop_"
                    or next_low.startswith("data_")
                    or next_low.startswith("save_")
                ):
                    break
                raw.append(tokens[index])
                index += 1

            if len(raw) % len(tags) != 0:
                raise XRDDataError(
                    "cif_loop_width",
                    value_count=len(raw),
                    column_count=len(tags),
                )
            rows = [
                raw[start : start + len(tags)]
                for start in range(0, len(raw), len(tags))
            ]
            loops.append(CifLoop(tags, rows))
            if index < len(tokens) and tokens[index].lower() == "stop_":
                index += 1
            continue

        if token.startswith("_"):
            if index + 1 >= len(tokens):
                raise XRDDataError("cif_field_missing", field=token)
            values[token.lower()] = tokens[index + 1]
            index += 2
            continue

        index += 1

    return CifData(source, values, loops)


__all__ = ["read_cif_data", "tokenize_cif"]
