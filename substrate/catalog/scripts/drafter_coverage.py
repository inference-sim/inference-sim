#!/usr/bin/env python3
"""Report the drafters that published deployments run and the catalog lacks (design §12.1, §12.2).

    python3 scripts/drafter_coverage.py [--catalog ROOT] [--vllm-recipes DIR] [--inferencex DIR]
                                        [--json] [--report-only]

A model's model.yaml lists the drafters it can run with: `speculator` for the one its own graph
states, and each separate drafter checkpoint by its catalog name; a separate drafter names its
targets (validate-catalog checks the link in both directions). This script reads published
deployments and checks that the catalog holds what each of them runs:

  * vLLM's recipe files (recipes/models/<org>/<name>.yaml): the file's path is the target
    checkpoint, and every `--speculative-config` JSON in it is a deployment;
  * InferenceX's recipe files (srt-slurm-recipes): `served-model-name` is the target, and each
    `speculative-config` is a deployment.

For a deployment that names a draft checkpoint (`model` in the speculative config) the catalog
needs an entry whose source repository is that checkpoint, that names the target's entry among
its `targets`, and that the target's entry lists. For one that names none (MTP, or a draft shipped
inside the checkpoint) the target's entry must list `speculator`, or list a drafter shipped in
the target's own repository. Listing is not enough: the draft must run the deployment's method.
A graph's `speculator.method` states the method it prices (the target's graph for `speculator`,
the drafter's own graph for a separate drafter), and it is compared with the method the
deployment states after vLLM's own rewrite of a model-specific MTP name to `mtp`
(`derive_graph.vllm_speculative_method`). A target whose in-graph speculator is MTP does not
satisfy an `eagle3` deployment, and a same-repository DSpark drafter does not satisfy an `mtp`
one. A target the catalog lacks is
reported apart: it is a missing model, not a missing drafter. A recipe for a quantisation of a
checkpoint is read as the checkpoint it names; the quantised entries are checked by their own
recipes.

The exit status is 1 when a drafter is missing, unless --report-only. Run on a schedule beside the
registry import, with checkouts of the two recipe sources, and open an issue from the report; the
workflow that does is not in this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from derive_graph import vllm_speculative_method


@dataclass(frozen=True)
class Deployment:
    target: str            # the target checkpoint's HF repository
    method: str            # the speculative method the deployment states
    draft: str | None      # the draft checkpoint's repository, or None
    source: str            # the file it was read from


def _specs(node) -> list[dict]:
    """Every speculative config in a parsed YAML document: a `--speculative-config` flag
    followed by its JSON, or a speculative-config key holding JSON or a mapping."""
    out: list[dict] = []

    def take(v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return
        if isinstance(v, dict) and v.get("method"):
            out.append(v)

    def walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k in ("speculative-config", "speculative_config"):
                    take(v)
                else:
                    walk(v)
        elif isinstance(n, list):
            for i, v in enumerate(n):
                if v == "--speculative-config" and i + 1 < len(n):
                    take(n[i + 1])
                else:
                    walk(v)

    walk(node)
    return out


def _first(node, keys) -> str | None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in keys and isinstance(v, str):
                return v
        for v in node.values():
            r = _first(v, keys)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _first(v, keys)
            if r:
                return r
    return None


def read_vllm_recipes(root: Path) -> list[Deployment]:
    out = []
    for f in sorted(root.glob("*/*.yaml")):
        doc = yaml.safe_load(f.read_text())
        target = f"{f.parent.name}/{f.stem}"
        for spec in _specs(doc):
            out.append(Deployment(target, str(spec["method"]), spec.get("model"), str(f)))
    return out


def read_inferencex(root: Path) -> list[Deployment]:
    out = []
    for f in sorted(root.rglob("*.yaml")):
        try:
            docs = list(yaml.safe_load_all(f.read_text()))
        except yaml.YAMLError:
            continue
        for doc in docs:
            specs = _specs(doc)
            target = _first(doc, {"served-model-name"})
            if not specs or not target:
                continue
            for spec in specs:
                out.append(Deployment(target, str(spec["method"]), spec.get("model"), str(f)))
    return out


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        self.models: dict[str, dict] = {}
        self._methods: dict[str, str | None] = {}
        self.targets_by_repo: dict[str, str] = {}
        self.drafters_by_repo: dict[str, str] = {}
        for m in sorted((root / "models").glob("*/model.yaml")):
            card = yaml.safe_load(m.read_text()) or {}
            self.models[m.parent.name] = card
            repo = ((card.get("source") or {}).get("repo") or "").casefold()
            if repo:
                # A drafter shipped in its target's repository (MiniMax-M3.1-Flash's dspark/)
                # shares the repository; it is found as a drafter, never as the target.
                book = self.drafters_by_repo if card.get("targets") else self.targets_by_repo
                book.setdefault(repo, m.parent.name)

    def entry(self, repo: str) -> str | None:
        """The target entry whose source repository is repo."""
        return self.targets_by_repo.get(repo.casefold())

    def drafter(self, repo: str) -> str | None:
        """The separate-drafter entry whose source repository is repo, else a target entry."""
        r = repo.casefold()
        return self.drafters_by_repo.get(r) or self.targets_by_repo.get(r)

    def speculator_method(self, name: str) -> str | None:
        """The method a model's graph states for its speculator, as a deployment states it
        (None when the graph has no speculator or no graph is committed)."""
        if name not in self._methods:
            self._methods[name] = None
            graph = self.root / "models" / name / "graph.yaml"
            if graph.is_file():
                spec = (yaml.load(graph.read_text(), Loader=_Loader) or {}).get("speculator") or {}
                if spec.get("method"):
                    self._methods[name] = vllm_speculative_method(str(spec["method"]))
        return self._methods[name]

    def shipped_drafters(self, target: str) -> list[str]:
        """The drafters the target lists that are shipped inside the target's own repository
        and name the target among their `targets`."""
        repo = ((self.models[target].get("source") or {}).get("repo") or "").casefold()
        return [d for d in self.models[target].get("drafters") or []
                if d != "speculator" and d in self.models
                and (self.models[d].get("source") or {}).get("repo", "").casefold() == repo
                and target in (self.models[d].get("targets") or [])]

    def runs_method_without_a_draft_checkpoint(self, target: str, method: str) -> bool:
        """Whether the target's entry holds a draft that runs `method` and needs no draft
        checkpoint named: its own in-graph speculator, or a drafter shipped in its repository."""
        listed = self.models[target].get("drafters") or []
        if "speculator" in listed and self.speculator_method(target) == method:
            return True
        return any(self.speculator_method(d) == method for d in self.shipped_drafters(target))


_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def check(cat: Catalog, deployments: list[Deployment]) -> dict[str, list[str]]:
    """Gaps by kind, each a sorted list of unique lines."""
    gaps: dict[str, set[str]] = {"missing_model": set(), "missing_drafter": set(),
                                 "unlinked_drafter": set(), "missing_speculator": set(),
                                 "wrong_method": set()}
    for d in deployments:
        target = cat.entry(d.target)
        if target is None:
            gaps["missing_model"].add(d.target)
            continue
        listed = (cat.models[target].get("drafters") or [])
        method = vllm_speculative_method(d.method)
        if not d.draft:
            if not cat.runs_method_without_a_draft_checkpoint(target, method):
                held = [cat.speculator_method(target) if "speculator" in listed else None,
                        *[cat.speculator_method(x) for x in cat.shipped_drafters(target)]]
                have = sorted({m for m in held if m})
                gaps["missing_speculator"].add(
                    f"{d.target}: {d.method} without a draft checkpoint, and {target} lists no draft "
                    f"that runs {method} (it holds {have or 'none'})")
            continue
        drafter = cat.drafter(d.draft)
        if drafter is None:
            gaps["missing_drafter"].add(f"{d.draft} ({d.method}) drafts for {d.target}")
            continue
        names = cat.models[drafter].get("targets") or []
        if target not in names or drafter not in listed:
            gaps["unlinked_drafter"].add(
                f"{drafter} for {target}: targets {names}, {target} lists {listed}")
            continue
        if cat.speculator_method(drafter) != method:
            gaps["wrong_method"].add(
                f"{drafter} for {target}: the deployment runs {d.method} with {d.draft}, "
                f"and {drafter}'s graph states {cat.speculator_method(drafter) or 'no speculator'}")
    return {k: sorted(v) for k, v in gaps.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--vllm-recipes")
    ap.add_argument("--inferencex")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not (args.vllm_recipes or args.inferencex):
        ap.error("name at least one source of deployments")
    deployments: list[Deployment] = []
    if args.vllm_recipes:
        deployments += read_vllm_recipes(Path(args.vllm_recipes))
    if args.inferencex:
        deployments += read_inferencex(Path(args.inferencex))
    gaps = check(Catalog(Path(args.catalog)), deployments)
    if args.json:
        print(json.dumps({"deployments": len(deployments), "gaps": gaps}, indent=2, sort_keys=True))
    else:
        print(f"{len(deployments)} deployment(s) read")
        for kind, lines in gaps.items():
            for line in lines:
                print(f"{kind}: {line}")
    missing = any(gaps[k] for k in ("missing_drafter", "unlinked_drafter", "missing_speculator", "wrong_method"))
    return 1 if missing and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
