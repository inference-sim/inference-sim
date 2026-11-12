"""Structural checks on the hand-written pages under docs/.

A page that repeats a table row is broken in a way no link check sees: the contracts'
reference index once carried one row interleaved with every line of the page
(blis-schemas#96). Markdown renders such a stray row as text, so the build passes. Run with
`DOCS_RELEASE=main python3 -m pytest docs/hooks -q`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent


def is_separator(line: str) -> bool:
    return "-" in line and set(line) <= set("|-: ")


def blocks(text: str) -> list[list[str]]:
    """Each run of consecutive lines that start with "|", outside fenced code."""
    out, current, fenced = [], [], False
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        if not fenced and stripped.startswith("|"):
            current.append(stripped)
            continue
        if current:
            out.append(current)
        current = []
    return out


def problems(text: str) -> list[str]:
    """A run of row lines that is not a table (a header, then a separator row), and a row
    that appears twice in one table."""
    found = []
    for rows in blocks(text):
        if len(rows) < 2 or not is_separator(rows[1]):
            found.append(f"not a table: {rows[0]}")
            continue
        body = [r for r in rows[2:] if not is_separator(r)]
        found.extend(f"repeated: {r}" for r, n in Counter(body).items() if n > 1)
    return found


def test_every_page_has_well_formed_tables():
    pages = sorted(DOCS.rglob("*.md"))
    assert pages, f"no pages under {DOCS}"
    found = {str(p.relative_to(DOCS)): problems(p.read_text(encoding="utf-8")) for p in pages}
    found = {page: rows for page, rows in found.items() if rows}
    assert not found, f"pages with broken tables: {found}"


def test_a_row_interleaved_with_the_text_is_found():
    row = "| x | y |"
    page = "\n".join(["| A | B |", "|---|---|", row, "", "text", row, "more text", row, ""])
    assert problems(page) == [f"not a table: {row}", f"not a table: {row}"]


def test_a_row_repeated_in_one_table_is_found():
    page = "| A | B |\n|---|---|\n| x | y |\n| z | w |\n| x | y |\n"
    assert problems(page) == ["repeated: | x | y |"]


def test_two_tables_may_share_a_header_and_a_row():
    table = "| Key | Meaning |\n|---|---|\n| `sha256` | its digest |\n"
    assert problems(table + "\ntext\n\n" + table) == []


def test_rows_in_fenced_code_are_not_checked():
    page = "```\n| A |\n| A |\n```\n"
    assert problems(page) == []


def test_the_corruption_of_the_reference_index_is_found():
    """blis-schemas#96: the reference landing page carried one table row after almost every
    source line, 29 copies in a 58-line file. The guard must find that exact damage on the real
    page, and find nothing on the page as it stands."""
    page = (DOCS / "contracts" / "reference" / "index.md").read_text(encoding="utf-8")
    assert problems(page) == []
    row = next(line for line in page.splitlines() if line.startswith("| [Acceptance profiles]"))
    damaged = "\n".join(line + "\n" + row for line in page.splitlines()) + "\n"
    found = problems(damaged)
    assert len(found) >= 10, found
    assert all(row in f for f in found if f.startswith("repeated")) or any(f.startswith("not a table") for f in found)
