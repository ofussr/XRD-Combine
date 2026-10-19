"""Toolkit-independent CIF tokenisation and basic data-block parsing."""

from __future__ import annotations

from pathlib import Path

from ..cif_lexer import CifLexError, CifToken, lex_cif_text
from ..models.crystal import CifData, CifLoop
from ..models.data_errors import XRDDataError


def tokenize_cif(text: str) -> list[str]:
    """Tokenize CIF 1.1 text and report a structured parse error."""
    return [token.value for token in _tokens(text)]


def _tokens(text: str) -> list[CifToken]:
    try:
        return lex_cif_text(text)
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
    tokens = _tokens(text)
    values: dict[str, str] = {}
    loops: list[CifLoop] = []
    index = 0
    block_seen = False

    while index < len(tokens):
        token = tokens[index]
        if token.kind == "data":
            # A single CifData cannot represent independent structures. Refuse
            # additional blocks rather than combine their cells and atom loops.
            if block_seen:
                raise XRDDataError("cif_multiple_blocks")
            block_seen = True
            index += 1
            continue
        if token.kind == "loop":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].kind == "tag":
                tags.append(tokens[index].value)
                index += 1
            if not tags:
                raise XRDDataError("cif_loop_no_columns")

            raw: list[str] = []
            while index < len(tokens):
                if tokens[index].kind != "value":
                    break
                raw.append(tokens[index].value)
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
            if index < len(tokens) and tokens[index].kind == "stop":
                index += 1
            continue

        if token.kind == "tag":
            if index + 1 >= len(tokens) or tokens[index + 1].kind != "value":
                raise XRDDataError("cif_field_missing", field=token.value)
            values[token.value.lower()] = tokens[index + 1].value
            index += 2
            continue

        index += 1

    return CifData(source, values, loops)


__all__ = ["read_cif_data", "tokenize_cif"]
