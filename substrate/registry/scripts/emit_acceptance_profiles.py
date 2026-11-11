#!/usr/bin/env python3
"""Generate acceptance/inferencex-golden-al.yaml from InferenceX's golden acceptance curves.

    emit_acceptance_profiles.py [--check]

Every acceptance length, and every provenance field a curve file states, is read from
SemiAnalysis InferenceX's `inferencex-e2e/infx/golden_al_distribution/*.yaml`, vendored
under validator/testdata/inferencex/ at INFERENCEX_SHA. Nothing is typed in here but the
classification of each file, the one model-name alias below, and the hardware of the runs
whose file names none (RUN_HARDWARE); each states its evidence.

A curve is carried only if its model resolves to exactly one blis-catalog model. The
collector names a curve's model by its Hugging Face repository's basename, lower-cased
(`MODEL_KEY` in benchmarks/single_node/speedbench/*.sh), so a key resolves to the catalog
model whose `source.repo` has that basename. A key that resolves to none is refused, and
the set's caveats name it, rather than mapped to the nearest name.

$INFERENCEX_GOLDEN and $BLIS_CATALOG override the vendored copies; an empty value means the
vendored copy. `--check` re-renders and exits non-zero if the committed file would change.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "acceptance" / "inferencex-golden-al.yaml"
SET_NAME = "inferencex-golden-al"

INFERENCEX_SHA = "9fa76586ed1ef66caadc6b95a9b1448acde94c6b"
# The rule that moves a measured curve to a model InferenceX marks provisional. It is the one
# InferenceX's own file states: the target shares the source model's base and layer topology,
# so the source's acceptance length is carried unchanged for the draft lengths both run.
TRANSFER_RULE = ("the source model's acceptance length is carried unchanged, because the file "
                 "states the target shares its base and layer topology")
INFERENCEX_URL = "https://github.com/SemiAnalysisAI/InferenceX"
GOLDEN_PATH = "inferencex-e2e/infx/golden_al_distribution"
GOLDEN = Path(os.environ.get("INFERENCEX_GOLDEN")
              or REPO / "validator" / "testdata" / "inferencex" / "golden_al_distribution")
CATALOG = Path(os.environ.get("BLIS_CATALOG") or REPO.parent / "catalog")

THINKING = {"thinking_on": "on", "thinking_off": "off"}

# Curves measured with a separate drafter checkpoint rather than the model's own in-graph
# speculator, with the evidence. A profile can name its drafter only when the drafter is a
# catalog model, so these name none and the caveat says why. A curve
# not listed here was measured with the model's own speculator: its MTP layers, or DSpark for
# a model whose catalog graph states DSpark as its speculator.
SEPARATE_DRAFTER = {
    "minimaxm3_eagle3": (
        "the curve is for the base EAGLE3 drafter. InferenceX's MiniMax-M3 runs serve EAGLE3 "
        "through a separate draft_model, which the catalog graph does not describe "
        "(the catalog lists no such drafter), and its minimaxm3_eagle3_gqa curve is for a second one."),
}

_KIMI = ("the two Kimi-K3 DSpark curves differ only in the draft sampler (greedy, and "
         "probabilistic with block rejection), which a profile's key cannot state, and "
         "InferenceX's own resolver (curves.py curve_name) refuses to pick one without it")
_GQA = ("it is the curve for the Inferact/MiniMax-M3-EAGLE3-GQA drafter and shares "
        "minimaxm3_eagle3's key, since a profile's key cannot state the drafter; InferenceX's "
        "resolver uses it only for a GQA drafter, and the base curve otherwise")

# Every curve file, with the engine's speculative method and, for a curve not carried on
# grounds other than the catalog, why. A file missing from this table fails the run, so a
# new curve is a visible decision rather than a silent omission.
CURVES: dict[str, tuple[str, str | None]] = {
    "dsv41flash_dspark": ("dspark", None),
    "dsv4_mtp": ("mtp", None),
    "dsv4-pro-0813-dspark": ("dspark", None),
    "glm5.2_mtp": ("mtp", None),
    "glm5.3_mtp": ("mtp", None),
    "kimik3_dspark": ("dspark", _KIMI),
    "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method": (
        "dspark", _KIMI),
    "minimaxm3_eagle3": ("eagle3", None),
    "minimaxm3_eagle3_gqa": ("eagle3", _GQA),
    "qwen3.5_mtp": ("mtp", None),
    "qwen3.8next_mtp": ("mtp", None),
}

# InferenceX keys that are not a catalog repository's basename, with the catalog model,
# the repository the catalog must still record for it, and the evidence.
ALIASES = {
    "glm-5.3-fp8": (
        "glm-5.3", "zai-org/GLM-5.3",
        "InferenceX's glm5.3-fp8-mi355x-tilert-agentic configuration "
        "(inferencex-e2e/configs/amd-master.yaml) serves zai-org/GLM-5.3 at precision fp8 "
        "and names glm5.3_mtp.yaml as its golden curve; the catalog's glm-5.3 is that "
        "repository, whose config.json carries an fp8 quantization_config"),
}

# The part each source run ran on, from its job's runner, read through the GitHub API
# (repos/SemiAnalysisAI/InferenceX/actions/runs/<id>/jobs) on 2026-10-14. Used where a file
# names no hardware, and checked against the file where it names one.
RUN_HARDWARE = {
    "34493175056": (
        "b200", "its collect-al job 102924945774 ran on runner b200-nscale-slurm_00 "
                "(label cluster:b200-nscale) and launched "
                "runners/launch_b200-nscale-compat.sh with BENCH_BASE "
                "benchmarks/single_node/dsv41flash_fp4_b200"),
    "27180633016": ("b300", "job 80238725958, runner b300-nv_16, label b300"),
    "28058352479": ("b300", "job 83066260243, runner b300-nv_17, label b300"),
    "28061204145": ("b300", "job 83075546305, runner b300-nv_10, label b300"),
}

IMAGE_UNRECORDED = "unrecorded: the source file names no engine image"

# Header keys (`key: value`, one or several per line split on ` | `) and where each goes.
SAMPLING_KEYS = {"temperature": float, "top_p": float, "top_k": int, "output_len": int}
SETTING_KEYS = {
    "draft_sample_method": "draft_sample_method",
    "rejection_sample_method": "rejection_sample_method",
    "enable_adaptive_verification": "enable_adaptive_verification",
    "engram cpu_offload": "engram_cpu_offload",
    "draft": "draft_model",
}
OTHER_KEYS = {"Source GitHub Actions run", "dataset", "model", "image", "TP", "method",
              "thinking_on chat_template_kwargs", "thinking_off chat_template_kwargs"}
KNOWN_KEYS = SAMPLING_KEYS.keys() | SETTING_KEYS.keys() | OTHER_KEYS
PIECE = re.compile(r"([^:=]+?)\s*:\s*(\S.*)")
# `speculative-config: a=b | c=d`: every piece after the prefix is `key=value`.
SPEC_CONFIG = "speculative-config:"
# `thinking_on : temp 0.6 ...` or `thinking_on: temperature=1.0, ...`: per-mode sampling.
PER_MODE = re.compile(r"(thinking_on|thinking_off)\s*:\s*(\S.*)")
# `Measured on <key> (<HW>, vLLM <method>[, draft: <repo>]), per num_speculative_tokens.`
MEASURED_ON = re.compile(r"Measured on (\S+) \(([^)]*)\), per num_speculative_tokens\.")
# Lines with a ':' or '=' that are prose, or read by a pattern of their own in collect().
# Every other such line must parse into keys this emitter transcribes.
PROSE = [re.compile(p) for p in (
    r"key = num_speculative_tokens\b.*",
    r"Draft lengths \d+-\d+ rejected by the image \(no measured AL\): \S+",
    r"Inherited value provenance \([^)]*\):",
)]

# The collector each carried curve's header names, at INFERENCEX_SHA: the precision it
# serves at. speedbench-al.yml labels every collection PRECISION fp4 and requires the
# collector be named <prefix>_fp4_b300_vllm.sh, so the fp4 in a name is not a precision.
# Read from inferencex-e2e/benchmarks/single_node/speedbench/<collector>.
COLLECTORS = {
    "dsv4_fp4_b300_vllm.sh": (
        "fp8", "serves the staged checkpoint with no quantization flag, and "
               "--kv-cache-dtype fp8 (line 123)"),
    "dsv4dspark_fp4_b300_vllm.sh": (
        "fp8", "serves the staged checkpoint with no quantization flag, and "
               "--kv-cache-dtype fp8 (line 169)"),
    "glm52_fp4_b300_vllm.sh": (
        "fp8", "serves the staged checkpoint with no quantization flag, and "
               "--kv-cache-dtype fp8 (line 131)"),
    "minimaxm3_fp4_b300_vllm.sh": (
        None, "serves the staged checkpoint, which its comment (lines 16-19) states is "
              "unquantized BF16, with no --kv-cache-dtype"),
}


class Refused(Exception):
    """A curve the set does not carry, with the reason its caveats state."""


@dataclass
class Header:
    stem: str
    lines: list[str]
    keys: dict[str, str] = field(default_factory=dict)
    per_mode: dict[str, str] = field(default_factory=dict)
    measured_on: tuple[str, str, str] | None = None  # (model key, hardware, engine)

    def get(self, key: str) -> str | None:
        return self.keys.get(key)

    def search(self, pattern: str) -> re.Match | None:
        for line in self.lines:
            if m := re.search(pattern, line):
                return m
        return None

    def put(self, key: str, value: str) -> None:
        if key not in KNOWN_KEYS:
            raise SystemExit(f"{self.stem}.yaml: header key {key!r} is not transcribed; "
                             "map it in emit_acceptance_profiles.py or refuse the curve")
        if self.keys.setdefault(key, value) != value:
            raise SystemExit(f"{self.stem}.yaml: {key!r} is stated twice, differently")


def read_header(path: Path) -> Header:
    """Every `key: value` or `key=value` a header states, or an error.

    The measured-on line is parsed; any other line with neither ':' nor '=' is prose, and so
    are the PROSE lines and the provisional block (its PROVISIONAL line to the inherited
    provenance), which is quoted into a rationale. Every other piece stating a value must be
    a key this emitter transcribes, so none is silently dropped.
    """
    lines = []
    for raw in path.read_text().splitlines():
        if not raw.startswith("#"):
            break
        lines.append(raw.lstrip("#").strip())
    h = Header(path.stem, lines)
    start = next((i for i, l in enumerate(lines) if l.startswith("PROVISIONAL")), None)
    end = next((i for i, l in enumerate(lines) if l.startswith("Inherited value")),
               len(lines))
    for i, line in enumerate(lines):
        if start is not None and start <= i < end:
            continue
        if m := MEASURED_ON.fullmatch(line):
            hardware, engine, *rest = (x.strip() for x in m.group(2).split(","))
            for piece in rest:
                key, sep, value = piece.partition(":")
                if not sep:
                    raise SystemExit(f"{path.name}: measured-on detail {piece!r} is not "
                                     "transcribed")
                h.put(key.strip(), value.strip())
            if h.measured_on and h.measured_on != (m.group(1), hardware, engine):
                raise SystemExit(f"{path.name}: two different measured-on lines")
            h.measured_on = (m.group(1), hardware, engine)
            continue
        if not re.search(r"[:=]", line) or any(p.fullmatch(line) for p in PROSE):
            continue
        if m := PER_MODE.fullmatch(line):
            h.per_mode[m.group(1)] = m.group(2)
            continue
        if line.startswith(SPEC_CONFIG):
            for piece in line[len(SPEC_CONFIG):].split(" | "):
                key, sep, value = piece.strip().partition("=")
                if not sep or not key or not value:
                    raise SystemExit(f"{path.name}: speculative-config piece {piece!r} is "
                                     "not key=value")
                h.put(key.strip(), value.strip())
            continue
        for piece in line.split(" | "):
            piece = piece.strip()
            if m := PIECE.fullmatch(piece):
                h.put(*m.groups())
            elif re.search(r"[:=]", piece):
                raise SystemExit(f"{path.name}: header piece {piece!r} states a value this "
                                 "emitter does not transcribe")
    return h


def read_curve(path: Path) -> tuple[str, dict[str, dict[int, float]]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise SystemExit(f"{path.name}: expected one model key")
    [(key, modes)] = data.items()
    out: dict[str, dict[int, float]] = {}
    for mode, cells in modes.items():
        if mode not in THINKING:
            raise SystemExit(f"{path.name}: unknown thinking mode {mode!r}")
        out[THINKING[mode]] = {}
        for k, al in cells.items():
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise SystemExit(f"{path.name}: {mode} key {k!r} is not a K")
            if isinstance(al, bool) or not isinstance(al, (int, float)) \
                    or not 1 <= al <= k + 1:
                raise SystemExit(f"{path.name}: {mode}/{k} AL {al!r} outside [1, K+1]")
            out[THINKING[mode]][k] = float(al)
    return str(key), out


def catalog_models(root: Path) -> dict[str, str]:
    models = {}
    for p in sorted((root / "models").glob("*/model.yaml")):
        d = yaml.safe_load(p.read_text())
        # A separate drafter is not what a collector's model key names, and one may share its
        # target's repository (MiniMax-M3.1-Flash ships its DSpark drafter under dspark/).
        if d.get("targets"):
            continue
        models[d["name"]] = d["source"]["repo"]
    if not models:
        raise SystemExit(f"no catalog models under {root / 'models'}")
    return models


def catalog_architectures(root: Path) -> dict[str, str]:
    """Each catalog model's architecture, as its graph records it (global.architecture)."""
    out = {}
    for p in sorted((root / "models").glob("*/graph.yaml")):
        g = yaml.safe_load(p.read_text())
        arch = (g.get("global") or {}).get("architecture")
        if not arch:
            raise SystemExit(f"{p.parent.name}: its graph records no global.architecture, which "
                             "a lookup that borrows across targets reads (design D8)")
        out[g["name"]] = arch
    return out


def catalog_name(key: str, models: dict[str, str]) -> tuple[str, str | None]:
    """The catalog model an InferenceX key names, and the alias evidence if it took one."""
    hits = sorted(n for n, repo in models.items() if repo.rsplit("/", 1)[-1].lower() == key)
    if len(hits) > 1:
        raise SystemExit(f"InferenceX model {key!r} matches several catalog models: {hits}")
    if hits:
        return hits[0], None
    if key in ALIASES:
        name, repo, why = ALIASES[key]
        if models.get(name) != repo:
            raise SystemExit(f"alias {key} -> {name} rests on the catalog's {name} being "
                             f"{repo}, but it is {models.get(name)!r}")
        return name, why
    raise Refused(f"its model {key} is no catalog model's repository, and no alias maps it")


def source_url(stem: str) -> str:
    return f"{INFERENCEX_URL}/blob/{INFERENCEX_SHA}/{GOLDEN_PATH}/{stem}.yaml"


@dataclass
class Carried:
    stem: str
    model: str
    method: str
    values: dict[str, dict[int, float]]
    provenance: dict
    mode_settings: dict[str, dict[str, str]]
    rationale: str | None = None
    copied_from: str | None = None
    copied_source: str | None = None
    alias: tuple[str, str] | None = None
    rejected: tuple[int, int, str] | None = None
    image_stated: bool = True
    collector: str | None = None


def collector_of(h: Header) -> str | None:
    m = h.search(r"Auto-generated by \S*?([^/\s]+\.sh) ")
    return m.group(1) if m else None


def provenance(h: Header, method: str, measured_key: str):
    """The provenance a header states, its per-mode settings, a rationale if the hardware
    came from RUN_HARDWARE rather than the file, and the category."""
    if h.per_mode:
        raise SystemExit(f"{h.stem}: per-mode sampling ({h.per_mode}) is not transcribed; "
                         "extend provenance() before carrying it")
    if (stated_method := h.get("method")) is not None:
        if stated_method.lower() != method:
            raise SystemExit(f"{h.stem}: the header states method {stated_method!r}, "
                             f"CURVES says {method!r}")
    elif h.measured_on:
        if h.measured_on[2].split()[-1].lower() != method:
            raise SystemExit(f"{h.stem}: measured with {h.measured_on[2]!r}, CURVES says "
                             f"{method!r}")
    elif method not in " ".join(h.lines).lower():
        raise SystemExit(f"{h.stem}: the header never names method {method!r}")
    run = h.get("Source GitHub Actions run")
    if not run:
        raise SystemExit(f"{h.stem}: no source Actions run")
    bench = h.search(r"measured with (SPEED-Bench(?: Qualitative)?)\b")
    category = h.get("dataset") or (m.group(1) if (m := h.search(
        r"SPEED-Bench Qualitative (\w+)")) else None)
    if not bench or not category:
        raise SystemExit(f"{h.stem}: no benchmark or category stated")
    run_id = run.rstrip("/").rsplit("/", 1)[-1]
    stated = h.measured_on
    if stated and stated[0] != measured_key:
        raise SystemExit(f"{h.stem}: measured on {stated[0]}, keyed {measured_key}")
    observed = RUN_HARDWARE.get(run_id)
    rationale = None
    if stated:
        hardware = stated[1].lower()
        if observed and observed[0] != hardware:
            raise SystemExit(f"{h.stem}: file states {hardware}, run {run_id} ran on "
                             f"{observed[0]}")
    elif observed:
        hardware = observed[0]
        rationale = (f"The source file names no hardware. {hardware} is the part run "
                     f"{run_id} ran on: {observed[1]}.")
    else:
        raise Refused(f"neither it nor RUN_HARDWARE states the hardware of run {run_id}")
    if not (CATALOG / "hardware" / f"{hardware}.yaml").is_file():
        raise SystemExit(f"{h.stem}: hardware {hardware} is not a catalog part")
    sampling = {}
    for key, cast in SAMPLING_KEYS.items():
        if (v := h.get(key)) is not None:
            sampling["max_tokens" if key == "output_len" else key] = cast(v)
    if "temperature" not in sampling:
        raise SystemExit(f"{h.stem}: no temperature stated")
    prov = {"source": source_url(h.stem), "run": run, "benchmark": bench.group(1),
            "hardware": hardware, "engine_image": h.get("image") or IMAGE_UNRECORDED}
    if tp := h.get("TP"):
        prov["tp"] = int(tp)
    prov["sampling"] = sampling
    settings = {name: h.get(key) for key, name in SETTING_KEYS.items() if h.get(key)}
    if collector := collector_of(h):
        if collector not in COLLECTORS:
            raise SystemExit(f"{h.stem}: collector {collector} is not in COLLECTORS; read "
                             "the precision it serves at before carrying the curve")
        if kv := COLLECTORS[collector][0]:
            settings["kv_cache_dtype"] = kv
    if settings:
        prov["settings"] = settings
    mode_settings = {}
    for mode in THINKING.values():
        if kw := h.get(f"thinking_{mode} chat_template_kwargs"):
            mode_settings[mode] = {"chat_template_kwargs": kw}
    return prov, mode_settings, rationale, category


def collect(golden: Path = GOLDEN, catalog: Path = CATALOG):
    files = sorted(golden.glob("*.yaml"))
    stems = {p.stem for p in files}
    if unknown := sorted(stems - CURVES.keys()):
        raise SystemExit(f"unclassified golden curves {unknown}: add each to CURVES")
    if missing := sorted(CURVES.keys() - stems):
        raise SystemExit(f"classified curves absent from {golden}: {missing}")
    models = catalog_models(catalog)
    curves = {p.stem: read_curve(p) for p in files}
    # Every header is read, carried or not, so a format change anywhere fails here.
    headers = {p.stem: read_header(p) for p in files}
    carried: list[Carried] = []
    skipped: list[tuple[str, str, str]] = []
    categories = set()
    for path in files:
        stem = path.stem
        method, reason = CURVES[stem]
        key, values = curves[stem]
        try:
            if reason:
                raise Refused(reason)
            model, alias = catalog_name(key, models)
            h = headers[stem]
            if (repo := h.get("model")) and repo != models[model]:
                raise SystemExit(f"{stem}: states model {repo}, but the catalog's {model} "
                                 f"is {models[model]}")
            copied = h.search(r"PROVISIONAL\W+copied from (\S+?)\.yaml")
            measured_key = curves[copied.group(1)][0] if copied else key
            prov, mode_settings, rationale, category = provenance(h, method, measured_key)
        except Refused as e:
            skipped.append((stem, key, str(e)))
            continue
        categories.add(category)
        c = Carried(stem, model, method, values, prov, mode_settings, rationale,
                    alias=(key, alias) if alias else None, image_stated=h.get("image") is not None,
                    collector=collector_of(h))
        if m := h.search(r"Draft lengths (\d+)-(\d+) rejected by the image[^:]*: (\S+)"):
            c.rejected = (int(m.group(1)), int(m.group(2)), m.group(3))
        if copied:
            c.copied_source = copied.group(1)
            start = next(i for i, l in enumerate(h.lines) if l.startswith("PROVISIONAL"))
            end = next(i for i, l in enumerate(h.lines) if l.startswith("Inherited value"))
            c.rationale = "As its source states: " + " ".join(
                l for l in h.lines[start:end] if l)
        carried.append(c)
    if len(categories) != 1:
        raise SystemExit(f"expected one category, found {sorted(categories)}")
    return carried, skipped, categories.pop()


def key_string(model, method, k, thinking, category) -> str:
    return f"{model}/{method}/k{k}/thinking-{thinking}/{category}"


def profiles(carried: list[Carried], category: str, architectures: dict[str, str]) -> list[dict]:
    by_stem = {c.stem: c for c in carried}
    out = []
    for c in carried:
        src = by_stem.get(c.copied_source) if c.copied_source else None
        if c.copied_source and src is None:
            raise SystemExit(f"{c.stem} is copied from {c.copied_source}, which is not carried")
        for thinking in sorted(c.values):
            for k in sorted(c.values[thinking]):
                al = c.values[thinking][k]
                p = {"model": c.model, "architecture": architectures[c.model],
                     "speculative_method": c.method, "num_spec_tokens": k,
                     "thinking": thinking, "category": category, "acceptance_length": al}
                prov = dict(c.provenance)
                if settings := {**prov.get("settings", {}),
                                **c.mode_settings.get(thinking, {})}:
                    prov["settings"] = settings
                if src:
                    if src.values.get(thinking, {}).get(k) != al:
                        raise SystemExit(f"{c.stem} {thinking}/{k} = {al} is not the value "
                                         f"of {src.stem}, which it states it copies")
                    p["method"] = "transferred"
                    p["transferred_from"] = key_string(src.model, src.method, k, thinking, category)
                    p["transfer_rule"] = TRANSFER_RULE
                else:
                    p["method"] = "measured"
                alias = (f"InferenceX names the model {c.alias[0]}: {c.alias[1]}."
                         if c.alias else None)
                rationale = " ".join(s for s in (c.rationale, alias) if s)
                if rationale:
                    p["rationale"] = rationale
                p["provenance"] = prov
                out.append(p)
    out.sort(key=lambda p: (p["model"], p["speculative_method"], p["thinking"],
                            p["num_spec_tokens"]))
    return out


def caveats(carried: list[Carried], skipped, category: str) -> list[str]:
    out = [f"{category.capitalize()} only. Every profile is SPEED-Bench's {category} "
           "category, the only one InferenceX publishes; no other category has a value. A "
           f"consumer must refuse other traffic, or fall back to {category} with the "
           "substitution recorded; it must not substitute silently."]
    parts: dict[str, list[str]] = {}
    for c in carried:
        parts.setdefault(c.provenance["hardware"], []).append(c.model)
    for hw, models in sorted(parts.items(), key=lambda kv: -len(kv[1])):
        out.append(f"Measured on {hw}: {', '.join(sorted(set(models)))}. Acceptance depends "
                   "mostly on the model, drafter and workload rather than the chip; a "
                   "resolution on another part records the reuse.")
    for c in carried:
        if c.rationale and not c.copied_source:
            out.append(f"{c.model} {c.method}: {c.rationale[0].lower()}{c.rationale[1:]}")
    for c in carried:
        if c.rejected:
            lo, hi, run = c.rejected
            out.append(f"{c.model} with {c.method} at K {lo}-{hi} was rejected by the image "
                       f"{c.provenance['engine_image']} ({run}), so those K have no value.")
        if c.copied_source:
            ks = sorted({k for v in c.values.values() for k in v})
            out.append(f"{c.model} {c.method} (K {', '.join(map(str, ks))}) is provisional: "
                       f"copied from {c.copied_source}.yaml, not measured on {c.model}.")
    collected = {}
    for c in carried:
        if collector := c.collector:
            collected.setdefault(collector, []).append(c.model)
    if collected:
        clauses = "; ".join(
            f"{name} ({', '.join(sorted(set(models)))}) {COLLECTORS[name][1]}"
            for name, models in sorted(collected.items()))
        out.append(
            "Deployment precision. The fp4 in the collectors' names is the label "
            "speedbench-al.yml gives every collection (PRECISION fp4, and the "
            "<prefix>_fp4_b300_vllm.sh name it requires), not the precision served. Each "
            "collector serves its model's checkpoint at that checkpoint's own weight "
            f"precision, which is the catalog model's. At InferenceX {INFERENCEX_SHA[:12]}: "
            f"{clauses}. kv_cache_dtype is recorded where a collector sets it; a "
            "collector may have changed since its run.")
    unimaged = sorted({c.model for c in carried if not c.image_stated and not c.copied_source})
    if unimaged:
        out.append(f"The source files for {', '.join(unimaged)} name no engine image, so "
                   f"their engine_image reads \"{IMAGE_UNRECORDED}\".")
    for c in carried:
        if c.stem in SEPARATE_DRAFTER:
            out.append(f"{c.model} {c.method}: {SEPARATE_DRAFTER[c.stem]} The catalog lists no "
                       "such drafter yet, so a profile cannot name one (its drafter must be a "
                       "catalog model) and these profiles name none. Until it does, a run that "
                       "uses the model's own speculator would borrow them as another drafter's.")
    for stem, key, reason in skipped:
        out.append(f"Not carried: {stem}.yaml ({key}), because {reason}.")
    return out


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def generate(golden: Path = GOLDEN, catalog: Path = CATALOG) -> str:
    carried, skipped, category = collect(golden, catalog)
    doc = {"kind": "AcceptanceProfileSet", "name": SET_NAME,
           "caveats": caveats(carried, skipped, category),
           "profiles": profiles(carried, category, catalog_architectures(catalog))}
    header = (
        "# Speculative acceptance lengths, from SemiAnalysis InferenceX's golden curves.\n"
        "#\n"
        "# GENERATED by scripts/emit_acceptance_profiles.py from\n"
        f"# {INFERENCEX_URL}/tree/{INFERENCEX_SHA}/{GOLDEN_PATH}\n"
        "# Re-run it and commit the diff rather than editing values here;\n"
        "# `emit_acceptance_profiles.py --check` is the drift gate.\n")
    body = yaml.dump(doc, Dumper=_Dumper, sort_keys=False, width=88, allow_unicode=True,
                     default_flow_style=False)
    return header + body


def main(argv: list[str]) -> int:
    text = generate()
    if argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text() != text:
            print(f"{TARGET} differs from what emit_acceptance_profiles.py generates",
                  file=sys.stderr)
            return 1
        return 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
