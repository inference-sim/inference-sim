"""MkDocs hook: render the coefficient sets into the documentation at build time.

Every number the site shows about the registry's contents is computed here, from the
committed files under coefficients/, on every build. Nothing is retyped into a page, because
a retyped count is the defect this registry has caught most often (docs/methodology.md, §8.2).

A page asks for generated content with an HTML comment on a line of its own:

    <!-- registry:census -->                 one row per set: entries, methods, parts
    <!-- registry:chart -->                  method composition of each set, as an SVG
    <!-- registry:sources -->                entries grouped by the source they rest on
    <!-- registry:usage units -->            vocabulary terms in use (units, method,
                                             source-kind, source-role), with counts
    <!-- registry:set NAME [section=RE] [sort=natural] -->
                                             the set as a coefficient-by-part matrix; RE,
                                             if given, has one group and splits the matrix
                                             into one table per distinct match; natural
                                             sorts rows by name, numbers compared as numbers
                                             (otherwise rows keep file order)
    <!-- registry:entry SET NAME PART -->    one entry, quoted verbatim from its file
    <!-- registry:stat KEY -->               one number, inline (total, sets, names,
                                             parts, measured, vendor_spec, assumed, ...)
    <!-- registry:acceptance NAME [part] -->  an acceptance-profile set under acceptance/:
                                             part is "values" (default; AL by K per
                                             model and method), "provenance" (one row
                                             per source file) or "caveats"
    <!-- registry:ref -->                    the revision the pages were generated from
    <!-- registry:release -->                the newest release tag reachable from it

An unknown directive, a set with no reference page, or data the matrix cannot represent
faithfully fails the build rather than rendering something approximate.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

REPO_URL = "https://github.com/inference-sim/blis-substrate"

# Column order for parts: by generation, then anything new alphabetically. A part absent
# from this list still renders; it only sorts last.
# Scope keys beyond hardware a matrix row can carry as a label: a model-scoped or
# fabric-scoped entry is a different row from the part-wide one of the same name.
TAGGED_SCOPES = {"model", "fabric"}

PART_ORDER = [
    "l40s", "a100-sxm", "a100-80", "h100", "h200",
    "b200", "b300", "gb200-nvl72", "gb300",
]

# Display order and glyph per method. The glyph's shape carries the method on its own, so
# the encoding survives colour-blindness, greyscale print and forced-colours mode; colour
# is a second channel, set in extra.css.
METHODS = ["measured", "vendor_spec", "literature", "copied", "assumed", "not_charged"]
GLYPH = {
    "measured": "■", "vendor_spec": "◆", "literature": "▲",
    "copied": "▼", "assumed": "○", "not_charged": "–",
}

DIRECTIVE = re.compile(r"^<!-- registry:(\w+)(?: ([^>]*?))? -->$", re.MULTILINE)
INLINE = re.compile(r"<!-- registry:(stat|ref|release)(?: ([^>]*?))? -->")


@dataclass(frozen=True)
class Entry:
    name: str
    body: dict
    line: int  # 1-based line of the entry's name in its file
    value_text: str = ""  # the value exactly as written in the file

    @property
    def method(self) -> str:
        return self.body["method"]

    @property
    def hardware(self) -> list[str]:
        return list(self.body.get("scope", {}).get("hardware", []))

    @property
    def fabric(self) -> list[str]:
        return list(self.body.get("scope", {}).get("fabric", []))

    @property
    def other_scope(self) -> dict:
        return {k: v for k, v in self.body.get("scope", {}).items() if k != "hardware"}


@dataclass(frozen=True)
class CoefficientSet:
    name: str
    path: str  # repo-relative
    entries: list[Entry]


# --- loading -----------------------------------------------------------------------------


def _load_set(root: Path, path: Path) -> CoefficientSet:
    text = path.read_text()
    data = yaml.safe_load(text)
    node = yaml.compose(text)
    found = []  # (line, value text), in file order
    for key, value in node.value:
        if key.value == "coefficients":
            for item in value.value:
                name_node, body_node = item.value[0]
                text = next((v.value for k, v in body_node.value if k.value == "value"), "")
                found.append((name_node.start_mark.line + 1, text))
    entries = []
    for item, (line, text) in zip(data["coefficients"], found, strict=True):
        ((name, body),) = item.items()
        entries.append(Entry(name, body, line, text))
    return CoefficientSet(data["name"], str(path.relative_to(root.parent)), entries)


@dataclass(frozen=True)
class AcceptanceSet:
    name: str
    path: str  # repo-relative
    caveats: list[str]
    profiles: list[tuple[int, dict]]  # (1-based line of the profile, the profile)


ACCEPTANCE_PAGE = "registry/reference/acceptance.md"


def _load_acceptance(root: Path) -> dict[str, AcceptanceSet]:
    d = root / "acceptance"
    out = {}
    for path in sorted(d.glob("*")) if d.is_dir() else []:
        if path.suffix.lower() not in (".yaml", ".yml"):
            continue
        text = path.read_text()
        data = yaml.safe_load(text)
        node = dict((k.value, v) for k, v in yaml.compose(text).value)
        lines = [item.start_mark.line + 1 for item in node["profiles"].value]
        out[data["name"]] = AcceptanceSet(
            data["name"], str(path.relative_to(root.parent)), list(data.get("caveats", [])),
            list(zip(lines, data["profiles"], strict=True)))
    return out


def _source_ref(root: Path) -> str:
    ref = os.environ.get("DOCS_SOURCE_REF")
    if ref:
        return ref
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def _release(root: Path) -> str:
    tag = os.environ.get("DOCS_RELEASE")
    if tag:
        return tag
    try:
        return subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"], cwd=root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise PluginError(
            "registry hook: no release tag is reachable from HEAD. Fetch tags "
            "(actions/checkout with fetch-depth: 0), or set DOCS_RELEASE.") from None


_state: dict = {}


def on_config(config):
    root = Path(config["config_file_path"]).resolve().parent / "registry"
    paths = sorted(p for p in (root / "coefficients").rglob("*")
                   if p.suffix.lower() in (".yaml", ".yml"))
    if not paths:
        raise PluginError("registry hook: no coefficient sets under coefficients/")
    sets = [_load_set(root, p) for p in paths]
    _state.clear()
    _state.update(root=root, sets={s.name: s for s in sets}, ref=_source_ref(root),
                  release=_release(root), acceptance=_load_acceptance(root))
    return config


def on_files(files, config):
    """Every committed set must have a reference page, and that page must be in the nav."""
    nav_pages = set()

    def walk(item):
        if isinstance(item, str):
            nav_pages.add(item)
        elif isinstance(item, dict):
            for v in item.values():
                walk(v)
        elif isinstance(item, list):
            for v in item:
                walk(v)

    walk(config["nav"])
    for name in _state["sets"]:
        page = f"registry/reference/sets/{name}.md"
        if files.get_file_from_path(page) is None:
            raise PluginError(
                f"registry hook: coefficients/ holds set {name!r} but docs/{page} does not "
                "exist. Every set gets a reference page; copy an existing one.")
        if page not in nav_pages:
            raise PluginError(f"registry hook: docs/{page} is not listed in mkdocs.yml nav")
    if _state["acceptance"]:
        page = files.get_file_from_path(ACCEPTANCE_PAGE)
        if page is None or ACCEPTANCE_PAGE not in nav_pages:
            raise PluginError(f"registry hook: acceptance/ holds profile sets, so docs/"
                              f"{ACCEPTANCE_PAGE} must exist and be listed in the nav")
        text = Path(page.abs_src_path).read_text()
        for name in _state["acceptance"]:
            if not re.search(rf"^<!-- registry:acceptance {re.escape(name)}(?: values)? -->$",
                             text, re.MULTILINE):
                raise PluginError(f"registry hook: docs/{ACCEPTANCE_PAGE} does not render "
                                  f"the values of acceptance set {name!r}")
    return files


# --- formatting --------------------------------------------------------------------------


def _parts(entries) -> list[str]:
    found = {h for e in entries for h in e.hardware} | {
        f for e in entries if not e.hardware for f in e.fabric}
    known = [p for p in PART_ORDER if p in found]
    return known + sorted(found - set(known))


def _fmt_value(v, text: str = "") -> str:
    """A value as committed, with the integer part grouped in thousands for reading.

    The literal text is used when there is one, so a page shows 3250.0 where the file says
    3250.0 rather than a re-rendering of the parsed float. Only digit grouping is added.
    """
    if isinstance(v, bool):
        raise PluginError(f"registry hook: boolean value {v!r}")
    text = text or repr(v)
    m = re.fullmatch(r"(-?)(\d+)(\.\d+)?", text)
    if not m:
        return text  # exponent form, kept verbatim
    sign, whole, frac = m.group(1), int(m.group(2)), m.group(3) or ""
    return f"{sign}{whole:,}{frac}" if whole >= 1000 else text


def _source_link(s: CoefficientSet, e: Entry) -> str:
    return f"{REPO_URL}/blob/{_state['ref']}/{s.path}#L{e.line}"


def _glyph(method: str) -> str:
    return (f'<span class="reg-m reg-m-{method}" aria-hidden="true">'
            f'{GLYPH.get(method, "?")}</span>')


def _cell(s: CoefficientSet, e: Entry) -> str:
    title = f"{e.name} · {e.method} · {e.body['units']} — open the entry and its provenance"
    return (f'<span class="reg-cell">{_glyph(e.method)}<span><a class="reg-v" '
            f'href="{_source_link(s, e)}" title="{html.escape(title)}">'
            f'{_fmt_value(e.body["value"], e.value_text)}</a></span></span>')


def _cells(s: CoefficientSet, entries: list[Entry]) -> str:
    """A part's cell. When several entries admit the part, the kernel keeps the last one in
    file order (it overwrites by name as it reads), so that one is shown and the others are
    listed in the mark's tooltip. A shadowed entry with a different value is flagged."""
    shown = _cell(s, entries[-1])
    if len(entries) == 1:
        return shown
    shown = shown.removesuffix("</span></span>")
    others = entries[:-1]
    differs = any(o.body["value"] != entries[-1].body["value"] for o in others)
    title = "; ".join(f"line {o.line}: {_fmt_value(o.body['value'], o.value_text)}"
                      for o in others)
    title = f"Also in scope, overridden by this entry: {title}"
    cls = "reg-shadow reg-shadow-differs" if differs else "reg-shadow"
    return (f'{shown}<sup class="{cls}"><a href="{_source_link(s, others[0])}" '
            f'title="{html.escape(title)}">+{len(others)}</a></sup></span></span>')


def _legend(methods) -> str:
    items = " ".join(
        f'<span class="reg-legend-item">{_glyph(m)} <code>{m}</code></span>'
        for m in METHODS if m in methods)
    return f'<p class="reg-legend">{items}</p>'


def _table(header: list[str], rows: list[list[str]], numeric_from: int) -> str:
    def row(cells, tag):
        out = []
        for i, c in enumerate(cells):
            cls = ' class="reg-num"' if i >= numeric_from else ""
            out.append(f"<{tag}{cls}>{c}</{tag}>")
        return "<tr>" + "".join(out) + "</tr>"

    head = row(header, "th")
    body = "\n".join(row(r, "td") for r in rows)
    return (f'<div class="reg-table-wrap"><table class="reg-table">'
            f"<thead>{head}</thead><tbody>\n{body}\n</tbody></table></div>")


# --- directives --------------------------------------------------------------------------


def _fold(e: Entry) -> tuple[str, list[str]]:
    """The row an entry belongs to, and the part columns it fills.

    Some families key the part in the name rather than only in the scope (the kernel builds
    collective keys as collective_<param>_<op>_<dtype>_<N>rank_<chip>). For those, the part
    named by the suffix is folded out of the row name and into the column, so the family
    renders as a matrix rather than a diagonal. Such an entry fills only the suffix's
    column even when its scope lists more parts, because a kernel on another part builds a
    different name and never looks this one up.
    """
    # An entry scoped beyond hardware (today: to one model) is a different row from the
    # part-wide entry of the same name, because the kernel admits it only for that
    # deployment. The extra scope is carried in the row label so the matrix stays one
    # value per (row, part) and a reader sees the restriction.
    tag = "".join(f" ‹{k}: {', '.join(str(x) for x in (v if isinstance(v, list) else [v]))}›"
                  for k, v in sorted(e.other_scope.items()))
    for part in e.hardware:
        suffix = "_" + part.replace("-", "_")
        if e.name.endswith(suffix):
            return e.name[: -len(suffix)] + "_‹part›" + tag, [part]
    return e.name + tag, e.hardware


def _stem(e: Entry) -> str:
    return _fold(e)[0]


def matrix(s: CoefficientSet):
    """The set as rows of part columns: {row: {part: [entries, in file order]}}, and each
    row's units. Fails on anything the matrix cannot show faithfully."""
    rows: dict[str, dict[str, list[Entry]]] = {}
    units: dict[str, str] = {}
    for e in s.entries:
        if set(e.other_scope) - TAGGED_SCOPES:
            raise PluginError(
                f"registry hook: {s.name}:{e.line} {e.name} is scoped by "
                f"{sorted(set(e.other_scope) - TAGGED_SCOPES)}; the matrix view shows hardware, "
                f"model and fabric scopes only. Extend the hook.")
        if not e.hardware and set(e.other_scope) == {"fabric"}:
            # A fabric-only entry (cost-model-fabric) belongs to a network, not a part: its
            # columns are the fabrics it names.
            stem = e.name
            cells = rows.setdefault(stem, {})
            if units.setdefault(stem, e.body["units"]) != e.body["units"]:
                raise PluginError(f"registry hook: {stem} carries two units in {s.name}")
            for f in e.fabric:
                cells.setdefault(f, []).append(e)
            continue
        if not e.hardware:
            raise PluginError(
                f"registry hook: {s.name}:{e.line} {e.name} names no hardware, so it has "
                f"no column in the matrix view. Extend the hook.")
        stem, columns = _fold(e)
        cells = rows.setdefault(stem, {})
        if units.setdefault(stem, e.body["units"]) != e.body["units"]:
            raise PluginError(f"registry hook: {stem} carries two units in {s.name}")
        for part in columns:
            cells.setdefault(part, []).append(e)
    return rows, units


def render_set(name: str, args: str) -> str:
    s = _state["sets"].get(name)
    if s is None:
        raise PluginError(f"registry hook: no coefficient set named {name!r}")
    section, natural = None, False
    for arg in args.split():
        key, _, value = arg.partition("=")
        if key == "section" and value:
            section = re.compile(value)
        elif key == "sort" and value == "natural":
            natural = True
        else:
            raise PluginError(f"registry hook: bad argument {arg!r} for set {name}")

    parts = _parts(s.entries)
    rows, units = matrix(s)

    groups: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, str] = {}
    for stem in rows:
        key, labels[stem] = "", stem
        if section is not None:
            m = section.search(stem)
            if not m:
                raise PluginError(f"registry hook: section pattern misses {stem} in {name}")
            key = m.group(1)
            # Inside a section the row is labelled by what follows the section's prefix;
            # the full name is kept as the label's tooltip.
            labels[stem] = stem[m.end():] or stem
        groups[key].append(stem)

    if natural:
        def natural_key(stem):
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", stem)]
        for stems in groups.values():
            stems.sort(key=natural_key)

    out = [_legend({e.method for e in s.entries})]
    for key, stems in groups.items():
        group_units = {units[stem] for stem in stems}
        shared = group_units.pop() if len(group_units) == 1 and key else None
        header = ["coefficient"] + [f"<code>{p}</code>" for p in parts]
        table_rows = []
        for stem in stems:
            cells = rows[stem]
            label = f'<code title="{html.escape(stem)}">{html.escape(labels[stem])}</code>'
            if not shared:
                label += f'<span class="reg-units">{units[stem]}</span>'
            table_rows.append(
                [label]
                + [_cells(s, cells[p]) if p in cells else '<span class="reg-empty">·</span>'
                   for p in parts])
        if key:
            out.append(f"\n### `{key}_…`" + (f" · `{shared}`" if shared else "") + "\n")
        out.append(_table(header, table_rows, numeric_from=1))
    shadowed = sum(len(c) - 1 for r in rows.values() for c in r.values())
    methods = Counter(e.method for e in s.entries)
    tally = ", ".join(f"{methods[m]:,} {m}" for m in METHODS if methods[m])
    out.append(
        f'<p class="reg-caption">{len(s.entries):,} entries ({tally}) across '
        f"{len(rows):,} coefficients and {len(parts)} parts. A dot (·) is a part the set "
        f"holds no entry for. Each value links to its entry in "
        f'<a href="{REPO_URL}/blob/{_state["ref"]}/{s.path}"><code>{s.path}</code></a>, '
        f"where its sources, rationale and scope are recorded.</p>")
    if shadowed:
        out.append(
            f'<p class="reg-caption"><sup class="reg-shadow">+n</sup> marks a cell where '
            f"{'an earlier entry' if shadowed == 1 else 'earlier entries'} with an "
            f"overlapping scope also apply ({shadowed:,} in this set). The kernel uses the "
            f"entry that comes last in the file, which is the value shown; hover the mark "
            f"for the others.</p>")
    return "\n\n".join(out)


def render_census(_args: str) -> str:
    header = ["set", "entries", "coefficients"] + [f"<code>{m}</code>" for m in METHODS
                                                   if _method_total(m)] + ["parts"]
    rows = []
    for s in _state["sets"].values():
        methods = Counter(e.method for e in s.entries)
        rows.append(
            [f'<a href="sets/{s.name}/"><code>{s.name}</code></a>',
             f"{len(s.entries):,}", f"{len({_stem(e) for e in s.entries}):,}"]
            + [f"{methods[m]:,}" if methods[m] else "—" for m in METHODS if _method_total(m)]
            + [str(len(_parts(s.entries)))])
    allentries = _all_entries()
    total = Counter(e.method for e in allentries)
    rows.append(
        ["<strong>all sets</strong>", f"<strong>{len(allentries):,}</strong>",
         f"<strong>{_stat('names')}</strong>"]
        + [f"<strong>{total[m]:,}</strong>" for m in METHODS if _method_total(m)]
        + [f"<strong>{len(_parts(allentries))}</strong>"])
    return _table(header, rows, numeric_from=1)


def render_chart(_args: str) -> str:
    """One horizontal bar per set, segmented by method, as shares of the set.

    Shares rather than counts, because the sets differ in size by two orders of magnitude
    (7 entries to 864) and the question the chart answers is what each set rests on. The
    count is printed beside each bar, and the census table above is the exact view.
    """
    sets = list(_state["sets"].values())
    used = [m for m in METHODS if _method_total(m)]
    label_w, bar_w, count_w, row_h, bar_h, top = 210, 440, 70, 30, 16, 34
    width = label_w + bar_w + count_w
    height = top + row_h * len(sets) + 8
    gap = 2  # surface-coloured gap between adjacent segments
    parts = [f'<svg class="reg-chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-labelledby="reg-chart-title" xmlns="http://www.w3.org/2000/svg">',
             '<title id="reg-chart-title">Share of each coefficient set by method</title>']
    # legend
    x = label_w
    for m in used:
        parts.append(f'<rect class="reg-fill-{m}" x="{x}" y="6" width="10" height="10" rx="2"/>'
                     f'<text class="reg-chart-text" x="{x + 14}" y="15">{m}</text>')
        x += 14 + 8 * len(m) + 18
    for i, s in enumerate(sets):
        y = top + i * row_h
        methods = Counter(e.method for e in s.entries)
        n = len(s.entries)
        parts.append(f'<text class="reg-chart-text" x="{label_w - 10}" y="{y + bar_h - 3}" '
                     f'text-anchor="end">{s.name}</text>')
        x = float(label_w)
        segs = [(m, methods[m]) for m in used if methods[m]]
        for j, (m, k) in enumerate(segs):
            w = bar_w * k / n
            drawn = max(w - (gap if j < len(segs) - 1 else 0), 1.0)
            parts.append(
                f'<rect class="reg-fill-{m}" x="{x:.1f}" y="{y}" width="{drawn:.1f}" '
                f'height="{bar_h}" rx="2"><title>{s.name}: {k:,} of {n:,} {m} '
                f"({100 * k / n:.1f}%)</title></rect>")
            x += w
        parts.append(f'<text class="reg-chart-text reg-chart-muted" x="{label_w + bar_w + 8}" '
                     f'y="{y + bar_h - 3}">{n:,}</text>')
    parts.append("</svg>")
    return "".join(parts)


def classify_source(e: Entry) -> str:
    """The evidence an entry rests on, read from its first primary source.

    The rules are deliberately few and literal, so a reader can check any row by reading the
    cite string: an AISimulate sweep path names its framework lane as the third component
    after systems/data/<sku>/<operator>/.
    """
    sources = e.body.get("sources") or []
    if not sources:
        return "no source cited"
    src = next((x for x in sources if x.get("role") == "primary"), sources[0])
    cite = src["cite"]
    if cite.startswith("Extrapolated") or cite.startswith("blis-registry"):
        return "derived from other registry entries"
    m = re.search(r"AISimulate systems/data/[^/\s]+/[^/\s]+/([a-z]+)/", cite)
    if m:
        return f"AISimulate sweep, {m.group(1)} lane"
    if re.search(r"AISimulate systems/[^/\s]+\.yaml", cite):
        return "AISimulate system descriptor"
    if "AISimulate" in cite:
        return "AISimulate source code"
    if "vLLM start-up logs" in cite:
        return "vLLM start-up logs"
    return f"other ({src['kind']})"


def render_sources(_args: str) -> str:
    by_class: dict[str, Counter] = defaultdict(Counter)
    for e in _all_entries():
        by_class[classify_source(e)][e.method] += 1
    used = [m for m in METHODS if _method_total(m)]
    header = ["primary source", "entries"] + [f"<code>{m}</code>" for m in used]
    total = len(_all_entries())
    rows = []
    for cls, methods in sorted(by_class.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(methods.values())
        rows.append([html.escape(cls), f"{n:,} ({100 * n / total:.1f}%)"]
                    + [f"{methods[m]:,}" if methods[m] else "—" for m in used])
    return _table(header, rows, numeric_from=1)


def render_usage(args: str) -> str:
    field = args.strip()
    counts: Counter = Counter()
    for e in _all_entries():
        if field in ("units", "method"):
            counts[e.body[field]] += 1
        elif field in ("source-kind", "source-role"):
            for src in e.body.get("sources") or []:
                counts[src[field.split("-")[1]]] += 1
        elif field == "scope":
            for k in e.body.get("scope", {}):
                counts[k] += 1
        elif field == "optional":
            for k in ("sources", "rationale", "ci95", "copied_from", "supersedes",
                      "validated", "unsupported"):
                if k in e.body:
                    counts[k] += 1
        else:
            raise PluginError(f"registry hook: unknown usage field {field!r}")
    rows = [[f"<code>{k}</code>", f"{v:,}"] for k, v in counts.most_common()]
    noun = "sources" if field.startswith("source") else "entries"
    return _table(["term", noun], rows, numeric_from=1)


def render_entry(args: str) -> str:
    """Quote one entry exactly as committed, so an example on a page cannot drift from the
    file it claims to show."""
    try:
        set_name, name, part = args.split()
    except ValueError:
        raise PluginError(f"registry hook: entry takes SET NAME PART, got {args!r}") from None
    s = _state["sets"].get(set_name)
    if s is None:
        raise PluginError(f"registry hook: no coefficient set named {set_name!r}")
    matches = [e for e in s.entries if e.name == name
               and (part in e.hardware or (not e.hardware and part in e.fabric))]
    if len(matches) != 1:
        raise PluginError(f"registry hook: {len(matches)} entries match {args!r}")
    e = matches[0]
    lines = (_state["root"].parent / s.path).read_text().splitlines()
    later = [x.line for x in s.entries if x.line > e.line]
    end = min(later) - 1 if later else len(lines)
    body = lines[e.line - 1:end]
    while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
        body.pop()
    indent = len(body[0]) - len(body[0].lstrip())
    text = "\n".join(line[indent:] for line in body)
    link = _source_link(s, e)
    return (f'```yaml title="{s.path}, line {e.line}"\n{text}\n```\n\n'
            f'<p class="reg-caption">Quoted from <a href="{link}">the committed file</a> '
            f"at build time.</p>")


def _fmt_al(al: float) -> str:
    """An acceptance length to two places, as the sources publish them, unless it has more."""
    return f"{al:.2f}" if round(al, 2) == al else repr(al)


def acceptance_grid(a: AcceptanceSet) -> dict[tuple[str, str, str], dict]:
    """(model, method, category) -> thinking -> K -> (line, profile), in file order."""
    grid: dict = {}
    for line, p in a.profiles:
        cell = grid.setdefault((p["model"], p["speculative_method"], p["category"]), {})
        slot = cell.setdefault(p["thinking"], {})
        if p["num_spec_tokens"] in slot:
            raise PluginError(f"registry hook: {a.path}:{line} repeats a profile key")
        slot[p["num_spec_tokens"]] = (line, p)
    return grid


def _acceptance_values(a: AcceptanceSet) -> str:
    out = [_legend({p["method"] for _, p in a.profiles})]
    for (model, method, category), modes in acceptance_grid(a).items():
        kmax = max(k for m in modes.values() for k in m)
        rows = []
        for thinking in sorted(modes):
            row = [f"<code>{thinking}</code>"]
            for k in range(1, kmax + 1):
                if k not in modes[thinking]:
                    row.append('<span class="reg-empty" title="no profile">·</span>')
                    continue
                line, p = modes[thinking][k]
                link = f"{REPO_URL}/blob/{_state['ref']}/{a.path}#L{line}"
                title = html.escape(f"{p['method']} · measured on {p['provenance']['hardware']}"
                                    " — open the profile and its provenance")
                row.append(f'<span class="reg-cell">{_glyph(p["method"])}<span><a '
                           f'class="reg-v" href="{link}" title="{title}">'
                           f'{_fmt_al(p["acceptance_length"])}</a></span></span>')
            rows.append(row)
        out.append(f"### `{model}` · `{method}` · `{category}`\n\n"
                   + _table(["thinking"] + [f"K={k}" for k in range(1, kmax + 1)], rows,
                            numeric_from=1))
    return "\n\n".join(out)


def _acceptance_provenance(a: AcceptanceSet) -> str:
    by_source: dict[str, list[dict]] = {}
    for _, p in a.profiles:
        by_source.setdefault(p["provenance"]["source"], []).append(p)
    rows = []
    for source, ps in by_source.items():
        v = ps[0]["provenance"]
        sampling = ", ".join(f"{k} {x}" for k, x in v["sampling"].items())
        settings = sorted({f"{k}: {x}" for p in ps
                           for k, x in p["provenance"].get("settings", {}).items()})
        run = v.get("run", "")
        rows.append([
            f'<a href="{source}"><code>{source.rsplit("/", 1)[-1]}</code></a>',
            f'<a href="{run}">{run.rsplit("/", 1)[-1]}</a>' if run else "—",
            ", ".join(sorted({f"<code>{p['model']}</code>" for p in ps})),
            ", ".join(sorted({p["method"] for p in ps})),
            f"<code>{v['hardware']}</code>",
            html.escape(v["engine_image"]),
            str(v.get("tp", "—")),
            html.escape(sampling),
            "<br>".join(html.escape(s) for s in settings) or "—",
        ])
    return _table(["file", "run", "model", "method", "measured on", "engine image", "TP",
                   "sampling", "settings"], rows, numeric_from=99)


def render_acceptance(args: str) -> str:
    name, _, part = args.partition(" ")
    a = _state["acceptance"].get(name)
    if a is None:
        raise PluginError(f"registry hook: no acceptance set {name!r} under acceptance/")
    part = part.strip() or "values"
    if part == "values":
        return _acceptance_values(a)
    if part == "provenance":
        return _acceptance_provenance(a)
    if part == "caveats":
        return "\n".join(f"- {c}" for c in a.caveats)
    raise PluginError(f"registry hook: registry:acceptance part {part!r} is not one of "
                      "values, provenance, caveats")


def _all_entries() -> list[Entry]:
    return [e for s in _state["sets"].values() for e in s.entries]


def _method_total(m: str) -> int:
    return sum(1 for e in _all_entries() if e.method == m)


def _stat(key: str) -> str:
    es = _all_entries()
    if key == "total":
        return f"{len(es):,}"
    if key == "sets":
        return str(len(_state["sets"]))
    if key == "names":
        return f"{sum(len({_stem(e) for e in s.entries}) for s in _state['sets'].values()):,}"
    if key == "parts":
        return str(len(_parts(es)))
    if key in METHODS:
        return f"{_method_total(key):,}"
    raise PluginError(f"registry hook: unknown stat {key!r}")


BLOCKS = {
    "set": None,  # handled separately: takes the set name
    "census": render_census,
    "chart": render_chart,
    "sources": render_sources,
    "usage": render_usage,
    "entry": render_entry,
    "acceptance": render_acceptance,
}


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("registry/"):
        return markdown

    def inline(m):
        if m.group(1) == "release":
            return _state["release"]
        if m.group(1) == "ref":
            ref = _state["ref"]
            short = ref[:12] if re.fullmatch(r"[0-9a-f]{40}", ref) else ref
            return f"[`{short}`]({REPO_URL}/tree/{ref})"
        return _stat((m.group(2) or "").strip())

    def block(m):
        kind, args = m.group(1), (m.group(2) or "").strip()
        if kind == "set":
            name, _, rest = args.partition(" ")
            return render_set(name, rest.strip())
        if kind in BLOCKS:
            return BLOCKS[kind](args)
        if kind in ("stat", "ref", "release"):
            return inline(m)
        raise PluginError(f"registry hook: unknown directive registry:{kind} in {page.file.src_uri}")

    markdown = DIRECTIVE.sub(block, markdown)
    return INLINE.sub(inline, markdown)
