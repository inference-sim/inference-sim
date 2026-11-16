#!/usr/bin/env python3
"""Generate acceptance/inferencex-golden-al.yaml from InferenceX's golden acceptance curves.

    emit_acceptance_profiles.py [--check]

Every acceptance length, and every provenance field, is read from SemiAnalysis InferenceX's
`inferencex-e2e/infx/golden_al_distribution/`: its curve files and the `identities.yaml`
that states how each curve was collected. No InferenceX file is committed (X12):
validator/testdata/inferencex/curves.json pins the commit and the SHA-256 of each file, and
scripts/fetch_inferencex_curves.py fetches them; the emitter refuses files that are not the
pinned ones. Nothing is typed in here but the explicit
skips and transfers below and the hardware of the runs whose file names none (RUN_HARDWARE),
each with its evidence.

Every curve is carried by default (design D8, §12.2). A curve is carried, with its drafter,
when its model and its drafter resolve in the catalog:

  - the collector names a curve's model by its Hugging Face repository's basename,
    lower-cased, so the key resolves to the catalog model whose `source.repo` has that
    basename;
  - a curve's drafter is `native` (shipped with the target) or a Hugging Face repository.
    A repository resolves to the catalog drafter entry with that `source.repo` that the
    model lists and that names the model as a target. `native` resolves to the model's own
    in-graph speculator when its entry lists `speculator`, and otherwise to the one drafter
    entry it lists that ships in its own repository.

A curve that does not resolve is not carried, and the set's caveats say why. So is a curve
whose profile key another curve also claims: a key states the target, drafter, method, K,
thinking mode and category, not the draft sampler or the verifier, and D8 refuses two
profiles that share a key under different settings rather than choose one.

$INFERENCEX_GOLDEN names the curve directory (default: the fetch cache at the pinned commit) and
$BLIS_CATALOG the catalog; an empty value means the default. `--check` re-renders and exits non-zero if the committed file would change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The shared drafter rule lives with the catalog: catalog/scripts/drafter_links.py of the catalog
# this run reads ($BLIS_CATALOG), else of this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "catalog" / "scripts"))
if os.environ.get("BLIS_CATALOG"):
    sys.path.insert(0, str(Path(os.environ["BLIS_CATALOG"]) / "scripts"))
import drafter_links  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "acceptance" / "inferencex-golden-al.yaml"
SET_NAME = "inferencex-golden-al"

# The InferenceX commit and the SHA-256 of every file the emitter reads. The files themselves
# are not committed (X12): scripts/fetch_inferencex_curves.py fetches them at the commit.
LOCK = REPO / "validator" / "testdata" / "inferencex" / "curves.json"
_LOCK = json.loads(LOCK.read_text())
INFERENCEX_SHA = _LOCK["commit"]
INFERENCEX_URL = _LOCK["repository"]
GOLDEN_PATH = _LOCK["path"]
GOLDEN = Path(os.environ.get("INFERENCEX_GOLDEN")
              or LOCK.parent / "cache" / INFERENCEX_SHA / "golden_al_distribution")
CATALOG = Path(os.environ.get("BLIS_CATALOG") or REPO.parent / "catalog")
IDENTITIES = "identities.yaml"

THINKING = {"thinking_on": "on", "thinking_off": "off"}
# A curve file is <model>__<method>__<draft>__<draft_sample>-<verify>.yaml (README.md,
# "Curve identities and names").
STEM = re.compile(r"(?P<model>[a-z0-9.-]+)__(?P<method>[a-z0-9]+)__(?P<draft>[a-z0-9.-]+)"
                  r"__(?P<sample>[a-z]+)-(?P<verify>[a-z]+)")

# Curves not carried although they resolve, each with its reason. Empty: every resolving
# curve is carried.
SKIP: dict[str, str] = {}

# Curves carried to a catalog model they were not measured on, with the rule and its
# evidence. Each is a profile of method transferred. A transfer applies only while the curve's
# own model is not in the catalog; once it is, the curve is carried as measured on that model.
TRANSFERS: dict[str, tuple[str, str, str]] = {
    "glm-5.3-nvfp4__mtp__native__default-default": (
        "glm-5.3",
        "the acceptance length measured on an NVFP4 quantization of the same model is carried "
        "unchanged to the model, whose base weights and MTP layers it shares",
        "InferenceX measures GLM-5.3 only on its NVFP4 checkpoint (glm5.3_fp4_b300_sglang.sh "
        "serves glm-5.3-nvfp4), which is a quantization of zai-org/GLM-5.3, the catalog's "
        "glm-5.3. The quantized target accepts slightly differently from the checkpoint the "
        "catalog describes; no curve measures that checkpoint"),
}

# The part each source run ran on, read from its job's runner through the GitHub API
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

# The collection facts of identities.yaml, transcribed into a profile's settings under the
# same names when the source states a value. A fact this table does not name fails the run.
FACTS = ("draft_sample", "verify", "adaptive", "topology", "draft_precision",
         "draft_kv_cache_dtype", "kv_cache_dtype")
# Sampling a collection block states, and where it goes. A key not named fails the run.
SAMPLING = {"temperature": float, "top_p": float, "top_k": int}
SAMPLING_SETTINGS = ("presence_penalty", "repetition_penalty")
# Header pieces read as sampling (`key: value`, or `key=value` on a per-mode line).
HEADER_SAMPLING = {"temperature": "temperature", "temp": "temperature", "top_p": "top_p",
                   "top_k": "top_k", "presence_penalty": "presence_penalty",
                   "repetition_penalty": "repetition_penalty"}
# Keys a header's `key: value | ...` line states that are not sampling, each read elsewhere.
HEADER_OTHER = {"dataset", "output_len", "model", "image", "TP"}
PART = re.compile(r"\b(GB\d{3}|[ABH]\d{3}|MI\d{3}X?)\b", re.IGNORECASE)
MEASURED_ON = re.compile(r"Measured on (\S+) \((.*)\), per num_speculative_tokens\.")


class Refused(Exception):
    """A curve the set does not carry, with the reason its caveats state."""


@dataclass
class Curve:
    stem: str
    path: Path
    model_key: str
    method: str
    draft: str
    identity: dict
    values: dict[str, dict[int, float]]
    source_run: str
    collection: dict | None
    header: list[str]
    sha256: str


@dataclass
class Carried:
    curve: Curve
    model: str
    drafter: str
    provenance: dict
    mode_sampling: dict[str, dict]
    mode_settings: dict[str, dict[str, str]]
    rationale: list[str] = field(default_factory=list)
    transfer: tuple[str, str] | None = None
    rejected: tuple[int, int, str] | None = None


def read_curve(path: Path, identities: dict) -> Curve:
    stem = path.stem
    m = STEM.fullmatch(stem)
    if not m:
        raise SystemExit(f"{path.name}: not named <model>__<method>__<draft>__<sample>-<verify>")
    if stem not in identities:
        raise SystemExit(f"{path.name}: identities.yaml states no identity for it")
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise SystemExit(f"{path.name}: not a mapping")
    source_run = data.pop("source_run", None)
    collection = data.pop("collection", None)
    if not source_run:
        raise SystemExit(f"{path.name}: no source_run")
    if len(data) != 1:
        raise SystemExit(f"{path.name}: expected one model key, found {sorted(data)}")
    [(key, modes)] = data.items()
    if key != m["model"]:
        raise SystemExit(f"{path.name}: its file name names model {m['model']}, its key {key}")
    values: dict[str, dict[int, float]] = {}
    for mode, cells in modes.items():
        if mode not in THINKING:
            raise SystemExit(f"{path.name}: unknown thinking mode {mode!r}")
        values[THINKING[mode]] = {}
        for k, al in cells.items():
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise SystemExit(f"{path.name}: {mode} key {k!r} is not a K")
            if isinstance(al, bool) or not isinstance(al, (int, float)) \
                    or not 1 <= al <= k + 1:
                raise SystemExit(f"{path.name}: {mode}/{k} AL {al!r} outside [1, K+1]")
            values[THINKING[mode]][k] = float(al)
    ident = identities[stem]
    if (ident.get("select") or {}).get("method") != m["method"]:
        raise SystemExit(f"{path.name}: identities.yaml selects method "
                         f"{(ident.get('select') or {}).get('method')!r}, its name {m['method']!r}")
    header = []
    for line in raw.decode().splitlines():
        if line.startswith("#"):
            header.append(line.lstrip("#").strip())
        elif header and not line.startswith("source_run"):
            break
    return Curve(stem, path, key, m["method"], m["draft"], ident, values, source_run,
                 collection, header, hashlib.sha256(raw).hexdigest())


def load_identities(golden: Path) -> dict:
    p = golden / IDENTITIES
    if not p.is_file():
        raise SystemExit(f"{p}: missing; every curve's identity is read from it")
    d = yaml.safe_load(p.read_text())
    if not isinstance(d, dict) or not d:
        raise SystemExit(f"{p}: no identities")
    return d


@dataclass
class Catalog:
    models: dict[str, dict]           # name -> model.yaml
    architectures: dict[str, str]
    root: Path = Path(".")
    links: object = None              # drafter_links.Catalog, read on first use

    def targets(self) -> dict[str, str]:
        return {n: d["source"]["repo"] for n, d in self.models.items() if not d.get("targets")}


def load_catalog(root: Path) -> Catalog:
    models = {}
    for p in sorted((root / "models").glob("*/model.yaml")):
        d = yaml.safe_load(p.read_text())
        models[d["name"]] = d
    if not models:
        raise SystemExit(f"no catalog models under {root / 'models'}")
    archs = {}
    for p in sorted((root / "models").glob("*/graph.yaml")):
        g = yaml.safe_load(p.read_text())
        arch = (g.get("global") or {}).get("architecture")
        if not arch:
            raise SystemExit(f"{p.parent.name}: its graph records no global.architecture, which "
                             "a lookup that borrows across targets reads (design D8)")
        archs[g["name"]] = arch
    return Catalog(models, archs, root)


def catalog_name(key: str, targets: dict[str, str]) -> str:
    """The catalog model an InferenceX model key names: the one whose repository's basename,
    lower-cased, is the key. A separate drafter is never a model key (MiniMax-M3.1-Flash ships
    its DSpark drafter in its own repository, so two entries share it)."""
    hits = sorted(n for n, repo in targets.items() if repo.rsplit("/", 1)[-1].lower() == key)
    if len(hits) > 1:
        raise SystemExit(f"InferenceX model {key!r} matches several catalog models: {hits}")
    if not hits:
        raise Refused(f"its model {key} is no catalog model's repository")
    return hits[0]


def catalog_drafter(model: str, draft: str, cat: Catalog) -> str:
    """The catalog drafter a curve's `draft` names for model: "" for its own speculator. The rule
    is catalog/scripts/drafter_links.py's, which every script that resolves a drafter shares."""
    if cat.links is None:
        cat.links = drafter_links.Catalog(cat.root)
    try:
        return drafter_links.resolve(cat.links, model, draft)
    except drafter_links.Unresolved as e:
        raise Refused(str(e))


def source_url(stem: str) -> str:
    return f"{INFERENCEX_URL}/blob/{INFERENCEX_SHA}/{GOLDEN_PATH}/{stem}.yaml"


def header_pieces(line: str) -> list[tuple[str, str]]:
    """`a: b | c: d` or `a=b, c: d` pieces of one header line, as (key, value)."""
    out = []
    for piece in re.split(r"\s*\|\s*|,\s+(?=[a-z_]+\s*[:=])", line):
        if m := re.fullmatch(r"([a-z_]+)\s*[:=]\s*(\S.*)", piece.strip()):
            out.append((m.group(1), m.group(2).strip()))
    return out


def header_sampling(c: Curve) -> tuple[dict, dict[str, dict]]:
    """The sampling a curve's header states: the pieces common to every mode, and each
    mode's own (`thinking_on: temp 0.6 top_p 0.95 ...` or `thinking_on: temperature=1.0`)."""
    common: dict = {}
    per_mode: dict[str, dict] = {}
    for line in c.header:
        mm = re.match(r"(thinking_on|thinking_off)(?: \([^)]*\))?\s*:\s*(.*)", line)
        if mm:
            body = mm.group(2)
            # `temp 0.6 top_p 0.95 top_k 20 presence_penalty 0.0 | chat_template_kwargs: ...`
            first = body.split("|")[0]
            pairs = re.findall(r"\b([a-z_]+)[ =]([-\d.]+)\b", first)
            if unknown := sorted({k for k, _ in pairs} - HEADER_SAMPLING.keys()):
                raise SystemExit(f"{c.stem}: header sampling {unknown} for {mm.group(1)} is not "
                                 "transcribed; map it in emit_acceptance_profiles.py or refuse the curve")
            mode = per_mode.setdefault(THINKING[mm.group(1)], {})
            for k, v in pairs:
                put(mode, HEADER_SAMPLING[k], v, c.stem)
            continue
        if line.startswith(("dataset:", "temperature:", "model:")):
            for k, v in header_pieces(line):
                if k in HEADER_SAMPLING:
                    put(common, HEADER_SAMPLING[k], v, c.stem)
                elif k == "output_len":
                    put(common, "max_tokens", v, c.stem)
                elif k not in HEADER_OTHER:
                    raise SystemExit(f"{c.stem}: header piece {k!r} states a value this emitter "
                                     "does not transcribe")
    return common, per_mode


def put(d: dict, key: str, value: str, stem: str) -> None:
    """Record a header value, refusing one stated twice, differently."""
    if d.setdefault(key, value) != value:
        raise SystemExit(f"{stem}: {key!r} is stated twice, differently ({d[key]} and {value})")


def header_kwargs(c: Curve, mode: str):
    """The chat_template_kwargs a header states for one mode, or None when it states none:
    `thinking_on chat_template_kwargs: {...}`, or `chat_template_kwargs: {...}` on the
    mode's own line."""
    for line in c.header:
        m = re.match(rf"thinking_{mode}(?: \([^)]*\))?\s*(?::.*?)?\bchat_template_kwargs: (\{{.*\}})",
                     line)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                raise SystemExit(f"{c.stem}: chat_template_kwargs {m.group(1)!r} is not JSON")
    return None


def cast_sampling(d: dict, where: str) -> tuple[dict, dict[str, str]]:
    sampling, settings = {}, {}
    for k, v in d.items():
        if k in SAMPLING:
            sampling[k] = SAMPLING[k](v)
        elif k == "max_tokens":
            sampling[k] = int(v)
        elif k in SAMPLING_SETTINGS:
            settings[k] = str(v)
        else:
            raise SystemExit(f"{where}: sampling parameter {k!r} is not transcribed; map it in "
                             "emit_acceptance_profiles.py or refuse the curve")
    return sampling, settings


def hardware_of(c: Curve, cat_root: Path) -> tuple[str, str | None, int | None]:
    """The catalog part a curve was measured on, a rationale when RUN_HARDWARE supplied it,
    and the TP its header states."""
    run_id = c.source_run.rstrip("/").rsplit("/", 1)[-1]
    observed = RUN_HARDWARE.get(run_id)
    stated, tp, rationale = None, None, None
    lines = [m for line in c.header if (m := MEASURED_ON.fullmatch(line))]
    if len({m.group(0) for m in lines}) > 1:
        raise SystemExit(f"{c.stem}: two different measured-on lines")
    for line in c.header:
        if m := MEASURED_ON.fullmatch(line):
            if m.group(1) != c.model_key:
                raise SystemExit(f"{c.stem}: measured on {m.group(1)}, keyed {c.model_key}")
            parts = {p.lower() for p in PART.findall(m.group(2))}
            if len(parts) > 1:
                raise SystemExit(f"{c.stem}: measured-on line names several parts {sorted(parts)}")
            stated = parts.pop() if parts else None
            if t := re.search(r"\bTP ?(\d+)\b", m.group(2)):
                tp = int(t.group(1))
        for k, v in header_pieces(line):
            if k == "TP":
                tp = int(v)
        if t := re.search(r"\bTP: (\d+)\b", line):
            tp = int(t.group(1))
    if stated:
        if observed and observed[0] != stated:
            raise SystemExit(f"{c.stem}: file states {stated}, run {run_id} ran on {observed[0]}")
        hardware = stated
    elif observed:
        hardware = observed[0]
        rationale = (f"The source file names no hardware. {hardware} is the part run "
                     f"{run_id} ran on: {observed[1]}.")
    else:
        raise Refused(f"neither it nor RUN_HARDWARE states the hardware of run {run_id}")
    if not (cat_root / "hardware" / f"{hardware}.yaml").is_file():
        raise Refused(f"it was measured on {hardware}, which is not a catalog part")
    return hardware, rationale, tp


def category_of(c: Curve) -> tuple[str, str]:
    bench = None
    category = None
    for line in c.header:
        if m := re.search(r"measured with (SPEED-Bench(?: Qualitative)?)\b", line):
            bench = m.group(1)
        if m := re.search(r"SPEED-Bench Qualitative (\w+)", line):
            category = m.group(1)
        for k, v in header_pieces(line):
            if k == "dataset":
                category = v
    if not bench or not category:
        raise SystemExit(f"{c.stem}: no benchmark or category stated")
    return bench, category


def provenance(c: Curve, cat_root: Path) -> tuple[dict, dict, dict, list[str], str]:
    """The provenance a curve states, its per-mode sampling and settings, rationales, and its
    category."""
    collection = c.identity.get("collection") or {}
    if collection.get("provisional") is True:
        raise Refused("InferenceX marks it provisional: not measured for its model")
    bench, category = category_of(c)
    hardware, hw_rationale, tp = hardware_of(c, cat_root)
    image = None
    for line in c.header:
        for k, v in header_pieces(line):
            if k == "image":
                image = v
    settings = {"engine": str(collection["engine"])}
    if collection.get("draft") != "native":
        settings["draft_model"] = str(collection["draft"])
    for name in FACTS:
        fact = collection.get(name)
        if not isinstance(fact, dict) or "value" not in fact or "source" not in fact:
            raise SystemExit(f"{c.stem}: identities.yaml fact {name} is not {{value, source}}")
        if fact["value"] is not None:
            v = fact["value"]
            settings[name] = json.dumps(v) if isinstance(v, bool) else str(v)
    prov = {"source": source_url(c.stem), "sha256": c.sha256, "run": c.source_run,
            "benchmark": bench, "hardware": hardware,
            "engine_image": image or IMAGE_UNRECORDED}
    if tp:
        prov["tp"] = tp
    common, per_mode_header = header_sampling(c)
    mode_sampling: dict[str, dict] = {}
    mode_settings: dict[str, dict[str, str]] = {}
    modes = (c.collection or {}).get("modes") or {}
    for mode in c.values:
        stated = modes.get(f"thinking_{mode}")
        if stated:
            raw = dict(stated.get("sampling") or {})
            kwargs = stated.get("chat_template_kwargs")
            # The header's common pieces must agree with the collection block.
            for k, v in common.items():
                if k in raw and k in SAMPLING and SAMPLING[k](v) != SAMPLING[k](raw[k]):
                    raise SystemExit(f"{c.stem}: header {k} {v} differs from the collection "
                                     f"block's {raw[k]}")
            if "max_tokens" in common:
                raw.setdefault("max_tokens", common["max_tokens"])
        else:
            raw = {**common, **per_mode_header.get(mode, {})}
            kwargs = header_kwargs(c, mode)
        sampling, extra = cast_sampling(raw, f"{c.stem} thinking_{mode}")
        if "temperature" not in sampling:
            raise SystemExit(f"{c.stem}: no temperature stated for thinking {mode}")
        mode_sampling[mode] = sampling
        ms = dict(extra)
        if kwargs is not None:
            ms["chat_template_kwargs"] = json.dumps(kwargs, sort_keys=True,
                                                    separators=(", ", ": "))
        if ms:
            mode_settings[mode] = ms
    prov["settings"] = settings
    rationale = [hw_rationale] if hw_rationale else []
    return prov, mode_sampling, mode_settings, rationale, category


def check_curve(c: Curve) -> None:
    """Every format check that needs no catalog, run on every curve before any is resolved, so
    a change in a curve no catalog model resolves still fails the run."""
    collection = c.identity.get("collection") or {}
    unknown = sorted(set(collection) - set(FACTS) - {"engine", "draft", "provisional"})
    if unknown:
        raise SystemExit(f"{c.stem}: identities.yaml states {unknown}, which this emitter does "
                         "not transcribe")
    for name in ("engine", "draft"):
        if not isinstance(collection.get(name), str) or not collection[name]:
            raise SystemExit(f"{c.stem}: identities.yaml states no collection.{name}")
    for name in FACTS:
        fact = collection.get(name)
        if not isinstance(fact, dict) or "value" not in fact or "source" not in fact:
            raise SystemExit(f"{c.stem}: identities.yaml fact {name} is not {{value, source}}")
    expected_draft = "native" if collection["draft"] == "native" else \
        re.sub(r"[^a-z0-9.]+", "-", collection["draft"].split("/")[-1].lower())
    if expected_draft != c.draft:
        raise SystemExit(f"{c.stem}: its name's draft {c.draft!r} is not identities.yaml's "
                         f"{collection['draft']!r}")
    category_of(c)
    header_sampling(c)
    for mode in (c.collection or {}).get("modes") or {}:
        cast_sampling(dict(((c.collection["modes"][mode] or {}).get("sampling")) or {}),
                      f"{c.stem} {mode}")
    for line in c.header:
        if m := MEASURED_ON.fullmatch(line):
            if m.group(1) != c.model_key:
                raise SystemExit(f"{c.stem}: measured on {m.group(1)}, keyed {c.model_key}")


def verify_inputs(golden: Path = GOLDEN, lock: dict = _LOCK) -> None:
    """The files in golden are exactly the ones curves.json pins, each with its SHA-256."""
    if not golden.is_dir():
        raise SystemExit(f"{golden}: no InferenceX curves; run scripts/fetch_inferencex_curves.py")
    have = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in golden.glob("*.yaml")}
    if missing := sorted(lock["files"].keys() - have.keys()):
        raise SystemExit(f"{golden}: curves.json pins {missing}, which are not there")
    if extra := sorted(have.keys() - lock["files"].keys()):
        raise SystemExit(f"{golden}: {extra} are not in curves.json; move the pin "
                         "(scripts/refresh_inferencex_curves.py)")
    for name, want in sorted(lock["files"].items()):
        if have[name] != want:
            raise SystemExit(f"{golden}/{name}: SHA-256 {have[name]}, curves.json pins {want}")


def collect(golden: Path = GOLDEN, catalog: Path = CATALOG):
    identities = load_identities(golden)
    files = sorted(p for p in golden.glob("*.yaml") if p.name != IDENTITIES)
    stems = {p.stem for p in files}
    if missing := sorted(set(identities) - stems):
        raise SystemExit(f"identities.yaml names curves with no file: {missing}")
    for stem, reason in SKIP.items():
        if stem not in stems or not reason.strip():
            raise SystemExit(f"SKIP {stem}: a skip names a curve file and states a reason")
    for stem in TRANSFERS:
        if stem not in stems:
            raise SystemExit(f"TRANSFERS {stem} names no curve file; the transfer is stale")
    cat = load_catalog(catalog)
    targets = cat.targets()
    # Every file is read, carried or not, so a format change anywhere fails here.
    curves = [read_curve(p, identities) for p in files]
    for c in curves:
        check_curve(c)
    carried: list[Carried] = []
    skipped: list[tuple[str, str, str]] = []
    categories = set()
    for c in curves:
        try:
            if c.stem in SKIP:
                raise Refused(SKIP[c.stem])
            # The models the curve is carried to: its own, when the catalog has it, as measured;
            # and a TRANSFERS model, as transferred, until a curve measures that model (then
            # the transfer is stale, below).
            homes: list[tuple[str, tuple[str, str] | None]] = []
            try:
                homes.append((catalog_name(c.model_key, targets), None))
            except Refused:
                if c.stem not in TRANSFERS:
                    raise
            if c.stem in TRANSFERS:
                target, rule, why = TRANSFERS[c.stem]
                if target not in targets:
                    raise SystemExit(f"TRANSFERS {c.stem}: {target} is not a catalog model")
                homes.append((target, (rule, why)))
            model, transfer = homes[0]
            drafter = catalog_drafter(model, str((c.identity.get("collection") or {})
                                                 .get("draft")), cat)
            prov, mode_sampling, mode_settings, rationale, category = provenance(c, catalog)
            for line in c.header:
                for k, v in header_pieces(line):
                    if k == "model" and v != cat.models[model]["source"]["repo"]:
                        raise SystemExit(f"{c.stem}: states model {v}, but the catalog's "
                                         f"{model} is {cat.models[model]['source']['repo']}")
                if (m := re.search(r"draft: (\S+) @ ([0-9a-f]{7,40})", line)) and drafter:
                    rev = cat.models[drafter]["source"]["revision"]
                    if not rev.startswith(m.group(2)):
                        raise Refused(f"it was measured with {m.group(1)} at revision "
                                      f"{m.group(2)}, and the catalog's {drafter} is {rev}")
                    prov["settings"]["draft_revision"] = m.group(2)
        except Refused as e:
            skipped.append((c.stem, c.model_key, str(e)))
            continue
        categories.add(category)
        rejected = None
        for line in c.header:
            if m := re.search(r"Draft lengths (\d+)-(\d+) rejected by the image[^:]*: (\S+)", line):
                rejected = (int(m.group(1)), int(m.group(2)), m.group(3))
        for home, how in homes:
            d = drafter if home == model else catalog_drafter(
                home, str((c.identity.get("collection") or {}).get("draft")), cat)
            k = Carried(c, home, d, prov, mode_sampling, mode_settings, list(rationale), how)
            k.rejected = rejected
            carried.append(k)
    if len(categories) != 1:
        raise SystemExit(f"expected one category, found {sorted(categories)}")
    category = categories.pop()
    # A transfer fills a model no curve measures; once one does, the transfer is stale.
    for stem, (model, _, _) in TRANSFERS.items():
        for k in carried:
            if k.transfer is None and k.model == model and k.curve.method == \
                    next(c.method for c in curves if c.stem == stem):
                raise SystemExit(f"TRANSFERS {stem} is stale: {k.curve.stem} measures {model}")
    # A key two curves claim is refused for both (D8): the key cannot state what differs.
    claims: dict[tuple, list[Carried]] = {}
    for k in carried:
        for thinking, cells in k.curve.values.items():
            for n in cells:
                claims.setdefault((k.model, k.drafter, k.curve.method, n, thinking, category),
                                  []).append(k)
    clash: dict[str, set[str]] = {}
    for owners in claims.values():
        if len({o.curve.stem for o in owners}) > 1:
            for o in owners:
                clash.setdefault(o.curve.stem, set()).update(
                    x.curve.stem for x in owners if x is not o)
    kept = []
    for k in carried:
        if k.curve.stem in clash:
            others = ", ".join(f"{s}.yaml" for s in sorted(clash[k.curve.stem]))
            skipped.append((k.curve.stem, k.curve.model_key,
                            f"its profile keys are also claimed by {others}, which a profile's "
                            "key cannot tell apart (inference-sim/blis-substrate#174)"))
        else:
            kept.append(k)
    skipped.sort()
    return kept, skipped, category, cat


def key_string(model, drafter, method, k, thinking, category) -> str:
    who = f"{model}+{drafter}" if drafter else model
    return f"{who}/{method}/k{k}/thinking-{thinking}/{category}"


def profiles(carried: list[Carried], category: str, cat: Catalog) -> list[dict]:
    out = []
    for c in carried:
        for thinking in sorted(c.curve.values):
            for k in sorted(c.curve.values[thinking]):
                al = c.curve.values[thinking][k]
                p = {"model": c.model}
                if c.drafter:
                    p["drafter"] = c.drafter
                p.update({"architecture": cat.architectures[c.model],
                          "speculative_method": c.curve.method, "num_spec_tokens": k,
                          "thinking": thinking, "category": category, "acceptance_length": al})
                prov = dict(c.provenance)
                prov["sampling"] = c.mode_sampling[thinking]
                prov["settings"] = {**prov["settings"], **c.mode_settings.get(thinking, {})}
                rationale = list(c.rationale)
                if c.transfer:
                    p["method"] = "transferred"
                    p["transferred_from"] = key_string(c.curve.model_key, "", c.curve.method,
                                                       k, thinking, category)
                    p["transfer_rule"] = c.transfer[0]
                    rationale.append(c.transfer[1] + ".")
                else:
                    p["method"] = "measured"
                if rationale:
                    p["rationale"] = " ".join(rationale)
                p["provenance"] = prov
                out.append(p)
    out.sort(key=lambda p: (p["model"], p.get("drafter", ""), p["speculative_method"],
                            p["thinking"], p["num_spec_tokens"]))
    return out


def caveats(carried: list[Carried], skipped, category: str) -> list[str]:
    out = [f"{category.capitalize()} only. Every profile is SPEED-Bench's {category} "
           "category, the only one InferenceX publishes; no other category has a value. A "
           f"consumer must refuse other traffic, or fall back to {category} with the "
           "substitution recorded; it must not substitute silently."]
    parts: dict[str, set[str]] = {}
    for c in carried:
        parts.setdefault(c.provenance["hardware"], set()).add(c.model)
    for hw, models in sorted(parts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(f"Measured on {hw}: {', '.join(sorted(models))}. Acceptance depends "
                   "mostly on the model, drafter and workload rather than the chip; a "
                   "resolution on another part records the reuse.")
    out.append("Natural acceptance. Every curve was collected with real verification "
               "(InferenceX README, How a golden AL curve is collected), so each value is a "
               "natural acceptance length; InferenceX's AgentX runs then force it with the "
               "engine's synthetic rejection sampler. The set's shapes, fitted to every "
               "vendored curve, spread a natural value that states no per-position rates.")
    for c in carried:
        if c.rejected:
            lo, hi, run = c.rejected
            out.append(f"{c.model} with {c.curve.method} at K {lo}-{hi} was rejected by the "
                       f"image {c.provenance['engine_image']} ({run}), so those K have no value.")
        if c.transfer:
            out.append(f"{c.model} {c.curve.method} is transferred from {c.curve.stem}.yaml, "
                       f"measured on {c.curve.model_key}, not on {c.model}.")
    unrevised = sorted(f"{c.model}+{c.drafter} ({c.curve.stem}.yaml)" for c in carried
                       if c.drafter and "draft_revision" not in c.provenance["settings"])
    if unrevised:
        out.append(f"Drafter revision unstated: {', '.join(unrevised)}. The source names the drafter "
                   "but not the revision it was measured at, so the profile may describe a revision "
                   "other than the catalog entry's.")
    out.append(f"Collection settings are transcribed from {IDENTITIES} at {INFERENCEX_SHA[:12]}, "
               f"SHA-256 {_LOCK['files'][IDENTITIES]} ({INFERENCEX_URL}/blob/{INFERENCEX_SHA}/"
               f"{GOLDEN_PATH}/{IDENTITIES}).")
    unimaged = sorted({c.model for c in carried if c.provenance["engine_image"] == IMAGE_UNRECORDED})
    if unimaged:
        out.append(f"The source files for {', '.join(unimaged)} name no engine image, so "
                   f"their engine_image reads \"{IMAGE_UNRECORDED}\".")
    for stem, key, reason in skipped:
        out.append(f"Not carried: {stem}.yaml ({key}), because {reason}.")
    return out


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


# The decay exponents a shape may take: the grid the fit searches, [0.5, 2] in steps of 0.01,
# the bounds acceptance.Shape validates.
EXPONENTS = [round(0.5 + i / 100, 2) for i in range(151)]
SHAPE_RULE = (
    "Each curve and thinking mode measured at every K from 1 to its largest, K0, is a test: "
    "its acceptance length at K0 is spread with p_i = q^(i^b), q solved so the rates sum to "
    "AL-1, and 1 + p_1 + ... + p_k is compared with the curve's measured acceptance length at "
    "every k <= K0. b is the exponent on the grid 0.50, 0.51, ..., 2.00 that minimises the sum, "
    "over the tests, of the squared worst error of each; the smallest such b on a tie. Curves "
    "are taken from every pinned file InferenceX does not mark provisional, carried as a "
    "profile or not, since a curve shows how acceptance decays whatever its model resolves to. "
    "Curves measured under different draft samplers and verifiers are pooled within a method; "
    "the exponent is an estimate, not a measurement.")


def natural_rates(al: float, k: int, b: float) -> list[float]:
    """acceptance.NaturalRates: p_i = q^(i^b), q in [0, 1] solved so the rates sum to al-1."""
    def rates(q):
        return [q ** (i ** b) for i in range(1, k + 1)]
    want = al - 1
    if want <= 0:
        return [0.0] * k
    if want >= k:
        return [1.0] * k
    lo, hi = 0.0, 1.0
    for _ in range(200):
        q = (lo + hi) / 2
        lo, hi = (q, hi) if sum(rates(q)) < want else (lo, q)
    return rates((lo + hi) / 2)


def worst_error(cells: dict[int, float], b: float) -> float:
    k0 = max(cells)
    r = natural_rates(cells[k0], k0, b)
    return max(abs(1 + sum(r[:k]) - al) for k, al in cells.items())


def fit_exponent(tests: list[dict[int, float]]) -> float:
    return min(EXPONENTS, key=lambda b: (sum(worst_error(c, b) ** 2 for c in tests), b))


def shape_tests(golden: Path = GOLDEN) -> dict[str, list[tuple[str, dict[int, float]]]]:
    """Every curve and thinking mode a shape is fitted to, by method, with its source."""
    identities = load_identities(golden)
    out: dict[str, list] = {}
    for p in sorted(x for x in golden.glob("*.yaml") if x.name != IDENTITIES):
        c = read_curve(p, identities)
        if (c.identity.get("collection") or {}).get("provisional") is True:
            continue
        for thinking in sorted(c.values):
            cells = c.values[thinking]
            if sorted(cells) == list(range(1, max(cells) + 1)) and max(cells) > 1:
                out.setdefault(c.method, []).append((source_url(c.stem), cells))
    return out


def shapes(golden: Path = GOLDEN) -> list[dict]:
    tests = shape_tests(golden)
    every = [t for m in sorted(tests) for t in tests[m]]
    out = []
    for method, ts in [(m, tests[m]) for m in sorted(tests)] + [("", every)]:
        s = {}
        if method:
            s["speculative_method"] = method
        s["decay_exponent"] = fit_exponent([c for _, c in ts])
        s["fitted_from"] = sorted({src for src, _ in ts})
        s["rule"] = SHAPE_RULE
        # A one-parameter form fitted to measured curves: an estimate, never a measurement.
        s["method"] = "calibrated"
        out.append(s)
    return out


def generate(golden: Path = GOLDEN, catalog: Path = CATALOG) -> str:
    carried, skipped, category, cat = collect(golden, catalog)
    doc = {"kind": "AcceptanceProfileSet", "name": SET_NAME,
           "caveats": caveats(carried, skipped, category),
           "shapes": shapes(golden),
           "profiles": profiles(carried, category, cat)}
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
    verify_inputs()
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
