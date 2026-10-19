"""Small CIF 1.1 lexer shared by the reflection and pole-figure tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CifLexError(ValueError):
    """A syntax error whose reason can be localized by the user interface."""

    line_number: int
    reason: str


@dataclass(frozen=True)
class CifToken:
    value: str
    line_number: int
    quoted: bool = False

    @property
    def kind(self) -> str:
        if not self.quoted:
            lower = self.value.lower()
            if lower.startswith("_"):
                return "tag"
            if lower.startswith("data_"):
                return "data"
            if lower.startswith("save_"):
                return "save"
            if lower in {"loop_", "stop_", "global_"}:
                return lower[:-1]
        return "value"


def _line_tokens(line: str, line_number: int) -> list[CifToken]:
    """Tokenize one ordinary CIF line.

    CIF quotes close only at a token boundary.  That differs from ``shlex`` and
    matters for names such as ``'D\\'eputier, S.'`` found in public COD files.
    Backslash-escaped quote characters are accepted as a common CIF extension.
    """

    tokens: list[CifToken] = []
    index = 0
    length = len(line)
    while index < length:
        while index < length and line[index].isspace():
            index += 1
        if index >= length or line[index] == "#":
            break

        quote = line[index] if line[index] in {"'", '"'} else None
        if quote is None:
            start = index
            while index < length and not line[index].isspace():
                if line[index] == "#":
                    break
                index += 1
            token = line[start:index]
            if token:
                tokens.append(CifToken(token, line_number))
            if index < length and line[index] == "#":
                break
            continue

        index += 1
        value: list[str] = []
        closed = False
        while index < length:
            character = line[index]
            if character == "\\" and index + 1 < length and line[index + 1] == quote:
                value.append(quote)
                index += 2
                continue
            if character == quote and (
                index + 1 == length
                or line[index + 1].isspace()
                or line[index + 1] == "#"
            ):
                index += 1
                closed = True
                break
            value.append(character)
            index += 1
        if not closed:
            raise CifLexError(line_number, "unclosed_quote")
        tokens.append(CifToken("".join(value), line_number, quoted=True))

    return tokens


def lex_cif_text(text: str) -> list[CifToken]:
    """Retain the category, source line and quoting of CIF tokens."""

    tokens: list[CifToken] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            block = [line[1:]]
            start_line = index + 1
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                block.append(lines[index])
                index += 1
            if index >= len(lines):
                raise CifLexError(start_line, "unclosed_multiline")
            tokens.append(CifToken("\n".join(block).strip(), start_line, quoted=True))
            index += 1
            continue
        tokens.extend(_line_tokens(line, index + 1))
        index += 1
    return tokens


def tokenize_cif_text(text: str) -> list[str]:
    """Compatibility value-only API; parsers should use lex_cif_text."""
    return [token.value for token in lex_cif_text(text)]
