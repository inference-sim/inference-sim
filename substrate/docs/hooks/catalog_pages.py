"""MkDocs hook: render the catalog's reference pages from the committed data.

The reference section of the documentation is not written by hand. Every table,
figure and per-model page under docs/catalog/reference/ is generated here, at build time,
from the same YAML and JSON files the simulator and the kernel read. A page
that restated a figure would be a second copy of it, and a second copy drifts;
generating the page means the documentation for a release describes exactly the
data in that release.

Hand-written pages reach the data through placeholders of the form

    [[catalog.NAME]]

which on_page_markdown replaces, in prose and in code blocks alike. Any text of the form [[catalog....]] whose name is
not known fails the build rather than rendering literally, so a typo cannot ship. To
show the syntax itself, write [[!catalog.NAME]]; it renders as [[catalog.NAME]].

The release the pages describe comes from the SUBSTRATE_RELEASE environment variable when
it is set; the docs workflow sets it to the release tag, or to the empty string for
`main`. When it is unset, a build of a clean checkout at a tagged commit uses that tag,
and any other build, including one with uncommitted changes, describes `main`.

The hook needs PyYAML and the standard library, nothing else.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_URL = "https://github.com/inference-sim/blis-substrate"
HF_URL = "https://huggingface.co"

MODELS_DIR = "catalog/reference/models"
PLACEHOLDER = re.compile(r"\[\[(!?)catalog\.([^\]]*)\]\]")

# Okabe-Ito, which stays distinguishable under the common color-vision deficiencies
# and reads on both the light and the dark theme.
KIND_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")


# --- Loading --------------------------------------------------------------------


@dataclass
class Model:
    name: str
    identity: dict[str, Any]
    config: dict[str, Any]
    graph: dict[str, Any]

    @property
    def architecture(self) -> str:
        arches = (self.config.get("architectures")
                  or self.config.get("text_config", {}).get("architectures") or ["?"])
        return arches[0]

    @property
    def layers(self) -> list[str]:
        """The stack expanded the way contracts/spec/model's Stack.Expand expands it."""
        s = self.graph["stack"]
        return (list(s.get("prologue", []))
                + list(s.get("pattern", [])) * s.get("repeat", 0)
                + list(s.get("epilogue", [])))

    def nodes(self) -> list[dict[str, Any]]:
        """Every node of every layer kind, including kinds only the speculator uses."""
        return [n for k in self.graph["layer_kinds"] for n in k["nodes"]]

    def stack_nodes(self) -> list[dict[str, Any]]:
        """The nodes of the kinds the main stack uses, excluding speculator-only kinds."""
        used = set(self.layers)
        return [n for k in self.graph["layer_kinds"] if k["id"] in used for n in k["nodes"]]

    def attention_kinds(self) -> list[str]:
        return sorted({n["kind"] for n in self.stack_nodes() if n["op"] == "Attention"})

    def recurrent_kinds(self) -> list[str]:
        return sorted({n["recurrent_kind"] for n in self.stack_nodes()
                       if n["op"] == "RecurrentUpdate"})

    def routed_experts(self) -> tuple[int, int] | None:
        """The widest GroupedGEMM's expert count and top-k, or None for a dense model."""
        grouped = [n for n in self.stack_nodes() if n["op"] == "GroupedGEMM"]
        if not grouped:
            return None
        widest = max(grouped, key=lambda n: n.get("experts", 0))
        return widest.get("experts", 0), widest.get("top_k", 0)

    def dtype_overrides(self) -> list[str]:
        base = self.graph["global"]["weight_dtype"]
        nodes = self.nodes() + list(self.graph.get("head", []))
        return sorted({n["weight_dtype"] for n in nodes
                       if n.get("weight_dtype") and n["weight_dtype"] != base})


@dataclass
class Catalog:
    root: Path
    release: str | None
    ref: str
    deriver_version: str
    models: list[Model] = field(default_factory=list)
    chips: dict[str, dict[str, Any]] = field(default_factory=dict)
    fabrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    fabric_headers: dict[str, list[str]] = field(default_factory=dict)
    workloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    tiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    routers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def source(self, path: str) -> str:
        return f"{REPO_URL}/blob/{self.ref}/catalog/{path}"


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _leading_comments(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        out.append(line.lstrip("#").strip())
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else None


def _first_match(pattern: str, path: Path, what: str) -> str:
    m = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    if not m:
        raise RuntimeError(f"{path}: could not find {what} (pattern {pattern!r})")
    return m.group(1)


def release_of(root: Path) -> str | None:
    """The release a build describes, or None for `main`. See the module docstring."""
    if "SUBSTRATE_RELEASE" in os.environ:
        release = os.environ["SUBSTRATE_RELEASE"].strip()
        if release and not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", release):
            raise RuntimeError(f"SUBSTRATE_RELEASE={release!r} is not vMAJOR.MINOR.PATCH")
        return release or None
    # A tag describes the tagged files only; a checkout with edits on top of it does
    # not hold that release, so it is described as `main`.
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        return None
    return _git(root, "describe", "--tags", "--exact-match", "HEAD")


def load(root: Path) -> Catalog:
    release = release_of(root)
    cat = Catalog(
        root=root,
        release=release,
        ref=release or "main",
        deriver_version=_first_match(r"^DERIVER_VERSION = (\d+)$",
                                     root / "scripts/derive_graph.py",
                                     "DERIVER_VERSION"),
    )
    for d in sorted(p for p in (root / "models").iterdir() if p.is_dir()):
        if d.name == "index":
            raise RuntimeError("models/index would collide with the models index page")
        cat.models.append(Model(
            name=d.name,
            identity=_yaml(d / "model.yaml"),
            # A derived-only model commits no config.json: only its hash and revision.
            config=(json.loads((d / "config.json").read_text(encoding="utf-8"))
                    if (d / "config.json").exists() else {}),
            graph=_yaml(d / "graph.yaml"),
        ))
    for p in sorted((root / "hardware").glob("*.yaml")):
        cat.chips[p.stem] = _yaml(p)
    for p in sorted((root / "networks").glob("*.yaml")):
        cat.fabrics[p.stem] = _yaml(p)
        cat.fabric_headers[p.stem] = _leading_comments(p)
    for p in sorted((root / "workloads").glob("*.yaml")):
        cat.workloads[p.stem] = _yaml(p)
    for p in sorted((root / "routers").glob("*.yaml")):
        cat.routers[p.stem] = _yaml(p)
    storage = root / "devices" / "storage.yaml"
    if storage.is_file():
        # A tier states a description and no number: its rates and latency are registry
        # coefficients (SUB-18), so only the comment keys are read.
        cat.tiers = {name: {k: str(v) for k, v in tier.items()
                            if k.startswith("_comment")}
                     for name, tier in _yaml(storage).items()}
    else:
        raise RuntimeError(f"{storage}: missing; the storage reference has nothing to show")
    return cat


# --- Formatting -------------------------------------------------------------------


def num(v: float | int | None, unit: str = "") -> str:
    """A figure as the file states it: no rounding, no trailing .0 on an integer."""
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int):
        s = f"{v:,}".replace(",", " ")  # narrow no-break space groups thousands
    else:
        s = f"{v:g}"
    return f"{s}{unit}"


def cell(s: Any) -> str:
    return str(s).replace("|", "\\|")


def table(header: list[str], align: str, rows: list[list[Any]]) -> str:
    """A Markdown table. align is one character per column: l or r."""
    rule = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


URL = re.compile(r"https?://[^\s<>\"'`]+[^\s<>\"'`.,;:]")


def linkify(text: str) -> str:
    """Escape a comment for HTML and turn its bare URLs into links.

    URLs are found in the raw text and escaped separately from the prose around them,
    so an escaped character can never become part of a link.
    """
    out, pos = [], 0
    for m in URL.finditer(text):
        url = m.group(0)
        # A URL inside parentheses ends before the closing one, (see https://x/y), but
        # keeps a balanced pair of its own, https://x/Foo_(bar).
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        out.append(html.escape(text[pos:m.start()]))
        out.append(f'<a href="{html.escape(url)}">{html.escape(url)}</a>')
        pos = m.start() + len(url)
    out.append(html.escape(text[pos:]))
    return "".join(out)


def notes_block(title: str, notes: list[tuple[str, str]]) -> str:
    """A collapsed note in the same markup pymdownx.details emits, so it is styled
    identically, holding the file's own provenance prose verbatim."""
    paras = "".join(f'<p><code>{html.escape(k)}</code> {linkify(v)}</p>' for k, v in notes)
    return (f'<details class="quote"><summary>{html.escape(title)}</summary>'
            f"{paras}</details>")


def comment_notes(doc: dict[str, Any]) -> list[tuple[str, str]]:
    return [(k, str(v)) for k, v in doc.items() if k.startswith("_comment")]


# --- SVG --------------------------------------------------------------------------


def nice_ticks(hi: float, count: int = 4) -> list[float]:
    if not hi > 0:
        raise ValueError(f"cannot scale an axis to {hi}; every charted figure is positive")
    raw = hi / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    n = math.ceil(hi / step - 1e-9)
    return [i * step for i in range(n + 1)]


def svg_open(width: int, height: int, label: str) -> str:
    # The wrapper scrolls sideways on a narrow screen rather than shrinking the figure
    # until its labels cannot be read; svg_close() ends it.
    return (f'<div class="cf-scroll"><svg class="cf-figure" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(label)}" xmlns="http://www.w3.org/2000/svg">'
            f"<title>{html.escape(label)}</title>")


def dot_panels(rows: list[str], panels: list[tuple[str, list[float | None]]],
               label: str) -> str:
    """Cleveland dot plots side by side, one row per item, sharing the row labels.

    Each panel has its own linear scale from zero, because the quantities are in
    different units and a shared scale would be meaningless.
    """
    label_w, panel_w, gap, row_h, top, bottom = 104, 132, 22, 22, 34, 26
    # The right margin leaves room for the last panel's final tick label.
    width = label_w + len(panels) * panel_w + (len(panels) - 1) * gap + 16
    height = top + row_h * len(rows) + bottom
    out = [svg_open(width, height, label)]
    for i, name in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(f'<text class="cf-label" x="{label_w - 10}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{html.escape(name)}</text>')
    for p, (title, values) in enumerate(panels):
        x0 = label_w + p * (panel_w + gap)
        ticks = nice_ticks(max(v for v in values if v is not None))
        hi = ticks[-1]
        sx = lambda v: x0 + panel_w * v / hi  # noqa: E731
        out.append(f'<text class="cf-title" x="{x0}" y="14">{html.escape(title)}</text>')
        for t in ticks:
            out.append(f'<line class="cf-grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" '
                       f'y1="{top - 6}" y2="{height - bottom + 4}"/>')
            out.append(f'<text class="cf-tick" x="{sx(t):.1f}" y="{height - 8}" '
                       f'text-anchor="middle">{num(t)}</text>')
        for i, v in enumerate(values):
            y = top + row_h * i + row_h / 2
            out.append(f'<line class="cf-rule" x1="{x0}" x2="{x0 + panel_w}" '
                       f'y1="{y:.1f}" y2="{y:.1f}"/>')
            if v is not None:
                out.append(f'<circle class="cf-dot" cx="{sx(v):.1f}" cy="{y:.1f}" r="4">'
                           f"<title>{html.escape(rows[i])}: {num(v)}</title></circle>")
    out.append("</svg></div>")
    return "".join(out)


def range_panels(rows: list[str], panels: list[tuple[str, list[dict[str, int]]]],
                 label: str) -> str:
    """Token-length ranges on a log scale: a thin line from tokens_min to tokens_max,
    a thick bar one standard deviation either side of the mean (clipped to the
    bounds), and a dot at the mean."""
    label_w, panel_w, gap, row_h, top, bottom = 104, 250, 48, 26, 34, 26
    lo_exp, hi_exp = 0, 5  # 1 to 100,000 tokens
    tick_text = ("1", "10", "100", "1k", "10k", "100k")
    width = label_w + len(panels) * panel_w + (len(panels) - 1) * gap + 16
    height = top + row_h * len(rows) + bottom
    out = [svg_open(width, height, label)]
    for i, name in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(f'<text class="cf-label" x="{label_w - 10}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{html.escape(name)}</text>')
    for p, (title, dists) in enumerate(panels):
        x0 = label_w + p * (panel_w + gap)

        def sx(v: float) -> float:
            v = max(v, 10 ** lo_exp)
            return x0 + panel_w * (math.log10(v) - lo_exp) / (hi_exp - lo_exp)

        out.append(f'<text class="cf-title" x="{x0}" y="14">{html.escape(title)}</text>')
        for e in range(lo_exp, hi_exp + 1):
            out.append(f'<line class="cf-grid" x1="{sx(10 ** e):.1f}" x2="{sx(10 ** e):.1f}" '
                       f'y1="{top - 6}" y2="{height - bottom + 4}"/>')
            out.append(f'<text class="cf-tick" x="{sx(10 ** e):.1f}" y="{height - 8}" '
                       f'text-anchor="middle">{tick_text[e]}</text>')
        for i, d in enumerate(dists):
            y = top + row_h * i + row_h / 2
            mean, sd = d["tokens"], d.get("tokens_stdev", 0)
            lo, hi = d.get("tokens_min") or mean, d.get("tokens_max") or mean
            out.append(f'<line class="cf-range" x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" '
                       f'y1="{y:.1f}" y2="{y:.1f}"/>')
            a, b = max(mean - sd, lo, 1), min(mean + sd, hi)
            out.append(f'<line class="cf-spread" x1="{sx(a):.1f}" x2="{sx(b):.1f}" '
                       f'y1="{y:.1f}" y2="{y:.1f}"/>')
            tip = (f"{rows[i]}: mean {num(mean)}, stdev {num(sd)}, "
                   f"min {num(lo)}, max {num(hi)}")
            out.append(f'<circle class="cf-dot" cx="{sx(mean):.1f}" cy="{y:.1f}" r="4">'
                       f"<title>{html.escape(tip)}</title></circle>")
    out.append("</svg></div>")
    return "".join(out)


def layer_strip(m: Model) -> str:
    """One cell per layer, colored by layer kind, in stack order."""
    layers = m.layers
    kinds = [k["id"] for k in m.graph["layer_kinds"]]
    color = {k: KIND_COLORS[i % len(KIND_COLORS)] for i, k in enumerate(kinds)}
    width, strip_h, top = 720, 22, 4
    cw = width / len(layers)
    counts = Counter(layers)
    legend_kinds = [k for k in kinds if counts[k]]
    height = top + strip_h + 30
    out = [svg_open(width, height,
                    f"{m.name}: {len(layers)} layers by kind, in stack order")]
    for i, k in enumerate(layers):
        out.append(f'<rect x="{i * cw:.2f}" y="{top}" width="{cw:.2f}" height="{strip_h}" '
                   f'fill="{color[k]}"><title>layer {i}: {html.escape(k)}</title></rect>')
    # Hairline separators keep adjacent layers of one kind countable without a border
    # that would dominate a 108-layer strip.
    if len(layers) <= 128:
        for i in range(1, len(layers)):
            out.append(f'<line class="cf-sep" x1="{i * cw:.2f}" x2="{i * cw:.2f}" '
                       f'y1="{top}" y2="{top + strip_h}"/>')
    x = 0.0
    for k in legend_kinds:
        text = f"{k} ×{counts[k]}"
        out.append(f'<rect x="{x:.1f}" y="{top + strip_h + 12}" width="10" height="10" '
                   f'fill="{color[k]}"/>')
        out.append(f'<text class="cf-label" x="{x + 15:.1f}" y="{top + strip_h + 21}">'
                   f"{html.escape(text)}</text>")
        x += 15 + 7.2 * len(text) + 18
    out.append("</svg></div>")
    return "".join(out)


# --- Pages and fragments -------------------------------------------------------------


def stack_summary(stack: dict[str, Any]) -> str:
    """The stack in words: runs of the prologue, the repeated pattern, the epilogue."""

    def runs(seq: list[str]) -> str:
        parts, i = [], 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            parts.append(f"`{seq[i]}` ×{j - i}" if j - i > 1 else f"`{seq[i]}`")
            i = j
        return ", ".join(parts)

    parts = []
    if stack.get("prologue"):
        parts.append(f"a prologue of {runs(stack['prologue'])}")
    if stack.get("pattern") and stack.get("repeat"):
        pat = ", ".join(f"`{k}`" for k in stack["pattern"])
        n = stack["repeat"]
        parts.append(f"the pattern ({pat}) " + ("once" if n == 1 else f"repeated {n} times"))
    if stack.get("epilogue"):
        parts.append(f"an epilogue of {runs(stack['epilogue'])}")
    return "; then ".join(parts)


def node_rows(nodes: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for i, n in enumerate(nodes):
        params = ", ".join(f"{k}={v}" for k, v in n.items()
                           if k not in ("op", "role", "emit"))
        rows.append([str(i), f"`{n['op']}`", n.get("role", ""),
                     f"`{n['emit']}`" if n.get("emit") else "", params])
    return rows


def mixing(m: Model) -> str:
    return " + ".join(m.attention_kinds() + m.recurrent_kinds())


def weights(m: Model) -> str:
    base = m.graph["global"]["weight_dtype"]
    over = m.dtype_overrides()
    return f"{base} ({', '.join(over)} for some nodes)" if over else base


def experts(m: Model) -> str:
    e = m.routed_experts()
    return f"{e[0]}, top-{e[1]}" if e else "—"


def draft(m: Model) -> str:
    s = m.graph.get("speculator")
    if not s:
        return "—"
    n = s["num_spec_tokens"]
    return f"`{s['method']}`, {n} token{'s' if n != 1 else ''}"


def model_page(cat: Catalog, m: Model) -> str:
    g, src = m.graph, m.identity["source"]
    repo, rev = src["repo"], src["revision"]
    config_path = src.get("path", "config.json")
    multimodal = g.get("modality") == "text_decoder_of_multimodal"
    lines = [
        # The page is reached from the models table, not the sidebar, so the sidebar's
        # width goes to the node tables instead.
        "---",
        "hide:",
        "  - navigation",
        "---",
        "",
        f"# {m.name}",
        "",
        (f"[`{repo}`]({HF_URL}/{repo}/tree/{rev}), architecture `{m.architecture}`: "
         f"{len(m.layers)} layers, weights in {weights(m)}. The graph below is derived "
         f"from the model's [`{config_path}`]({HF_URL}/{repo}/blob/{rev}/{config_path}) at "
         f"revision `{rev[:12]}`; [Model graphs](../../concepts/model-graphs.md) explains "
         f"how to read it, and the [glossary](../../concepts/glossary.md) defines its "
         f"terms."),
        "",
    ]
    if multimodal:
        lines += [
            '!!! note "Text decoder only"',
            "    The vendor config describes a multimodal model. This graph prices the "
            "text decoder and records `modality: text_decoder_of_multimodal`; a "
            "request carrying an image or audio input costs more than it predicts.",
            "",
        ]
    lines += [
        "## Layer stack",
        "",
        layer_strip(m),
        "",
        f"The stack is {stack_summary(g['stack'])}.",
        "",
        *([("*Sequence mixing* lists every attention and recurrent kind in the stack, "
            "including the small `gqa` attention an indexer uses to choose which tokens "
            "a sparse layer reads.")] if "sparse_mla" in m.attention_kinds() else []),
        "",
        "## Dimensions",
        "",
        table(["Quantity", "Value"], "lr", [
            ["Hidden size", num(g["global"]["hidden_size"])],
            ["Vocabulary", num(g["global"]["vocab_size"])],
            ["Tied embeddings", "yes" if g["global"]["tie_word_embeddings"] else "no"],
            ["Weights", weights(m)],
            ["Sequence mixing", mixing(m)],
            ["Routed experts", experts(m)],
            ["Speculator", draft(m)],
        ]),
        "",
        "## Layer kinds",
        "",
        ("Each layer kind is a small graph of primitives. Nodes are listed in the order "
         "`graph.yaml` gives them; the edges between them are in the file. *Runs when* "
         "is a collective's emit condition, which depends on the deployment's "
         "parallelism; a blank means the node always runs. *Parameters* are the "
         "dimensions the primitive is priced from, named as in `graph.yaml`."),
        "",
    ]
    for k in g["layer_kinds"]:
        n = Counter(m.layers)[k["id"]]
        where = f"{n} layer{'s' if n != 1 else ''}" if n else "speculator only"
        lines += [f"### `{k['id']}`", "", f"{where}.", "",
                  table(["#", "Op", "Role", "Runs when", "Parameters"], "rllll",
                        node_rows(k["nodes"])), ""]
    if g.get("head"):
        lines += ["### Head", "", "Runs once per forward pass, after the last layer.", "",
                  table(["#", "Op", "Role", "Runs when", "Parameters"], "rllll",
                        node_rows(g["head"])), ""]
    if g.get("speculator"):
        s = g["speculator"]
        lines += ["### Speculator", "",
                  (f"Method `{s['method']}`, proposing {s['num_spec_tokens']} draft "
                   f"token{'s' if s['num_spec_tokens'] != 1 else ''} per step. Its stack "
                   f"is {stack_summary(s['stack'])}."), ""]
    d = g["derived_from"]
    lines += [
        "## Provenance",
        "",
        table(["", ""], "ll", [
            ["Source", f"[`{repo}`]({HF_URL}/{repo}/tree/{rev}) on {src['provider']}"],
            ["Revision", f"`{rev}`"],
            ["Retrieved", str(src.get("retrieved", "—"))],
            ["Config digest", f"`sha256:{d['sha256']}`"],
            ["Deriver version", str(d["deriver_version"])],
        ]),
        "",
        "Files in this release: "
        + (f"[`config.json`]({cat.source(f'models/{m.name}/config.json')}) · "
           if (cat.root / "models" / m.name / "config.json").exists() else "") +
        f"[`model.yaml`]({cat.source(f'models/{m.name}/model.yaml')}) · "
        f"[`graph.yaml`]({cat.source(f'models/{m.name}/graph.yaml')})",
        "",
    ]
    return "\n".join(lines)


def models_table(cat: Catalog) -> str:
    rows = [[f"[{m.name}]({m.name}.md)", num(len(m.layers)), weights(m), mixing(m),
             experts(m), draft(m)] for m in cat.models]
    return table(["Model", "Layers", "Weights", "Sequence mixing", "Routed experts",
                  "Speculator"], "lrllll", rows)


def chip_rows(cat: Catalog) -> list[str]:
    return sorted(cat.chips, key=lambda c: (-cat.chips[c]["TFlopsPeak"], c))


def hardware_figure(cat: Catalog) -> str:
    rows = chip_rows(cat)
    c = cat.chips
    return dot_panels(rows, [
        ("Dense BF16, TFLOP/s", [c[r]["TFlopsPeak"] for r in rows]),
        ("HBM bandwidth, TB/s", [c[r]["BwPeakTBs"] for r in rows]),
        ("HBM capacity, GiB", [c[r]["MemoryGiB"] for r in rows]),
        ("BF16 FLOP per HBM byte", [c[r]["TFlopsPeak"] / c[r]["BwPeakTBs"] for r in rows]),
    ], "Peak compute, memory bandwidth, memory capacity and their ratio for each chip")


def hardware_table(cat: Catalog) -> str:
    rows = []
    for name in chip_rows(cat):
        h = cat.chips[name]
        rows.append([f"`{name}`", num(h["TFlopsPeak"]), num(h.get("TFlopsFP8") or None),
                     num(h.get("TFlopsNVFP4") or None), num(h["BwPeakTBs"]),
                     num(h["MemoryGiB"]), num(h["IntraNodeBwGBps"]), num(h["SMCount"])])
    return table(["Chip", "BF16", "FP8", "NVFP4", "HBM TB/s", "HBM GiB", "Intra-node GB/s",
                  "SMs"], "lrrrrrrr", rows)


def rack_table(cat: Catalog) -> str:
    rows = [[f"`{name}`", num(h["GPUsPerNode"]), num(h["GPUsPerRack"]),
             num(h["IntraNodeBwGBps"]), num(h["IntraRackBwGBps"])]
            for name in chip_rows(cat) if (h := cat.chips[name]).get("GPUsPerRack")]
    return table(["Chip", "GPUs per node", "GPUs per rack", "Intra-node GB/s",
                  "Intra-rack GB/s"], "lrrrr", rows)


def hardware_notes(cat: Catalog) -> str:
    return "\n\n".join(
        notes_block(f"{name}: provenance notes", comment_notes(cat.chips[name]))
        for name in sorted(cat.chips) if comment_notes(cat.chips[name]))


def networks_table(cat: Catalog) -> str:
    rows = []
    for name, f in sorted(cat.fabrics.items(), key=lambda kv: -kv[1]["InterNodeBwGBps"]):
        header = cat.fabric_headers[name][0] if cat.fabric_headers[name] else ""
        header = re.sub(r"\s+—\s+reusable inter-node fabric CLASS\.?$", "", header)
        rdma = {True: "true", False: "false"}.get(f.get("RDMA"), "not set")
        rows.append([f"`{name}`", header, num(f["InterNodeBwGBps"]), rdma])
    return table(["Fabric", "Description", "GB/s per GPU", "RDMA field"], "llrl", rows)


def networks_notes(cat: Catalog) -> str:
    return "\n\n".join(
        notes_block(f"{name}: provenance notes", comment_notes(f))
        for name, f in sorted(cat.fabrics.items()) if comment_notes(f))


def router_key(doc: dict[str, Any]) -> str:
    """The key a router-table entry is looked up by: router, mode and, when set, the
    session-affinity timeout (contracts/spec/routertable, Entry.Key)."""
    key = f"{doc['router']}/{doc['mode']}"
    if doc.get("session_affinity_ttl_s") is not None:
        key += f"/affinity-ttl-{doc['session_affinity_ttl_s']}"
    return key


def citation_link(c: dict[str, Any]) -> str:
    """A citation as a link to its lines at the pinned commit."""
    first, _, last = str(c["lines"]).partition("-")
    anchor = f"#L{first}" + (f"-L{last}" if last else "")
    url = f"https://github.com/{c['repository']}/blob/{c['commit']}/{c['path']}{anchor}"
    return f"[{c['repository']}@{c['commit'][:12]}:{c['path']}:{c['lines']}]({url})"


def routers_table(cat: Catalog) -> str:
    rows = []
    for name, e in sorted(cat.routers.items(), key=lambda kv: router_key(kv[1])):
        ttl = e.get("session_affinity_ttl_s")
        n = Counter(c["side"] for c in e["evidence"])
        rows.append([f"`{name}`", e["router"], e["mode"], "—" if ttl is None else str(ttl),
                     f"`{e['policy']}`", f"{n['router']} / {n['llm-d']}"])
    return table(["Entry", "Router", "Mode", "Affinity TTL (s)", "llm-d policy",
                  "Citations (router / llm-d)"], "lllrlr", rows)


def routers_entries(cat: Catalog) -> str:
    """Each entry's reason and its citations, as the file states them."""
    out = []
    for name, e in sorted(cat.routers.items(), key=lambda kv: router_key(kv[1])):
        out.append(f"### `{name}`\n\n{' '.join(str(e['reason']).split())}\n")
        for side, title in (("router", "The router"), ("llm-d", "llm-d's router")):
            out.append(f"**{title}**\n")
            out += [f"- {citation_link(c)}, for the {c['field'].replace('_', ' ')}: {c['claim']}"
                    for c in e["evidence"] if c["side"] == side]
            out.append("")
    return "\n".join(out)


def is_replay(doc: dict[str, Any]) -> bool:
    """A file whose top-level key is replay: is a replay workload, not a shape
    (contracts.LoadCatalogWorkload)."""
    return "replay" in doc


def workload_rows(cat: Catalog) -> list[str]:
    shapes = [w for w in cat.workloads if not is_replay(cat.workloads[w])]
    return sorted(shapes, key=lambda w: (cat.workloads[w]["prompt"]["tokens"], w))


def replay_rows(cat: Catalog) -> list[str]:
    return sorted(w for w in cat.workloads if is_replay(cat.workloads[w]))


def workloads_figure(cat: Catalog) -> str:
    rows = workload_rows(cat)
    w = cat.workloads
    return range_panels(rows, [
        ("Prompt tokens", [w[r]["prompt"] for r in rows]),
        ("Output tokens", [w[r]["output"] for r in rows]),
    ], "Prompt and output token distributions for each workload, on a log scale")


def workloads_table(cat: Catalog) -> str:
    rows = []
    for name in workload_rows(cat):
        w = cat.workloads[name]
        p, o = w["prompt"], w["output"]
        rows.append([f"`{name}`", num(p["tokens"]), num(p.get("tokens_stdev")),
                     f"{num(p.get('tokens_min'))}–{num(p.get('tokens_max'))}",
                     num(o["tokens"]), num(o.get("tokens_stdev")),
                     f"{num(o.get('tokens_min'))}–{num(o.get('tokens_max'))}",
                     num(w.get("prefix_tokens", 0))])
    return table(["Workload", "Prompt mean", "sd", "range", "Output mean", "sd", "range",
                  "Shared prefix"], "lrrrrrrr", rows)


def code(v: Any) -> str:
    """A stated value in code type, or a dash for an unstated one."""
    return "—" if v is None else f"`{v}`"


def replays_table(cat: Catalog) -> str:
    rows = []
    for name in replay_rows(cat):
        r = cat.workloads[name]["replay"]
        src = r["corpus"]["source"]
        corpus = (f"[{src['huggingface']}](https://huggingface.co/datasets/"
                  f"{src['huggingface']}/tree/{src['revision']})")
        rows.append([f"`{name}`", f"`{r['scenario']}`", r["harness_version"], corpus,
                     f"`{src['revision'][:12]}`", num(r.get("random_seed")),
                     num(r.get("benchmark_duration_s")), num(r.get("grace_period_s")),
                     code(r.get("max_context_length"))])
    return table(["Workload", "Scenario", "Harness", "Corpus", "Revision", "Seed",
                  "Duration s", "Grace s", "Max context"], "llllrrrrl", rows)


def replays_corpus_table(cat: Catalog) -> str:
    rows = []
    for name in replay_rows(cat):
        c = cat.workloads[name]["replay"]["corpus"]
        rows.append([f"`{name}`", f"`{c['loader']}`", f"`{c['format']}`",
                     code(c["source"].get("file")), code(c.get("sha256"))])
    return table(["Workload", "Loader", "Format", "File", "SHA-256"], "lllll", rows)


def replays_notes(cat: Catalog) -> str:
    return "\n\n".join(
        notes_block(f"{name}: provenance notes", comment_notes(cat.workloads[name]))
        for name in replay_rows(cat) if comment_notes(cat.workloads[name]))


def storage_table(cat: Catalog) -> str:
    """Each tier, what it is, and the three registry coefficients that price a transfer to it."""
    rows = []
    for name, t in sorted(cat.tiers.items()):
        what = " ".join(t.get("_comment", "").split())
        # The description's first sentence is the class; the rest says where its terms are.
        what = what.split(". ")[0].rstrip(".")
        rows.append([f"`{name}`", what,
                     f"`offload_tier_{name}_read_rate`, `_write_rate`, `_base_latency`"])
    return table(["Tier", "What it is", "Registry coefficients"], "lll", rows)






def release_text(cat: Catalog) -> str:
    if cat.release:
        return f"release [{cat.release}]({REPO_URL}/releases/tag/{cat.release})"
    return "the `main` branch, which is not a release"


def fragments(cat: Catalog) -> dict[str, str]:
    primitives = sorted({n["op"] for m in cat.models for n in m.nodes()})
    return {
        "release": release_text(cat),
        "ref": cat.ref,
        "deriver_version": cat.deriver_version,
        "contracts_tree": f"{REPO_URL}/tree/{cat.ref}/contracts",
        "count.models": str(len(cat.models)),
        "count.chips": str(len(cat.chips)),
        "count.fabrics": str(len(cat.fabrics)),
        "count.workloads": str(len(cat.workloads)),
        "count.tiers": str(len(cat.tiers)),
        "count.routers": str(len(cat.routers)),
        "routers.table": routers_table(cat),
        "routers.entries": routers_entries(cat),
        "count.primitives_used": str(len(primitives)),
        "primitives_used": ", ".join(f"`{p}`" for p in primitives),
        "models.table": models_table(cat),
        "hardware.figure": hardware_figure(cat),
        "hardware.table": hardware_table(cat),
        "hardware.racks": rack_table(cat),
        "hardware.notes": hardware_notes(cat),
        "networks.table": networks_table(cat),
        "networks.notes": networks_notes(cat),
        "workloads.figure": workloads_figure(cat),
        "workloads.table": workloads_table(cat),
        "count.replays": str(len(replay_rows(cat))),
        "workloads.replays": replays_table(cat),
        "workloads.replays_corpus": replays_corpus_table(cat),
        "workloads.replays_notes": replays_notes(cat),
        "storage.table": storage_table(cat),
    }


def substitute(markdown: str, frags: dict[str, str], where: str) -> str:
    def repl(m: re.Match[str]) -> str:
        escaped, name = m.group(1), m.group(2)
        if escaped:
            return f"[[catalog.{name}]]"
        if name not in frags:
            raise KeyError(f"{where}: unknown placeholder [[catalog.{name}]]; "
                           f"known: {', '.join(sorted(frags))}")
        return frags[name]

    return PLACEHOLDER.sub(repl, markdown)


# --- MkDocs events ------------------------------------------------------------------

_state: dict[str, Any] = {}


def on_config(config):
    cat = load(Path(config.config_file_path).parent / "catalog")
    _state["catalog"] = cat
    _state["fragments"] = fragments(cat)
    return config


def on_files(files, config):
    from mkdocs.structure.files import File, InclusionLevel

    cat = _state["catalog"]
    for m in cat.models:
        # Reached from the models table, not from the navigation, so not listed in it.
        files.append(File.generated(config, f"{MODELS_DIR}/{m.name}.md",
                                    content=model_page(cat, m),
                                    inclusion=InclusionLevel.NOT_IN_NAV))
    return files


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("catalog/"):
        return markdown
    return substitute(markdown, _state["fragments"], page.file.src_uri)
