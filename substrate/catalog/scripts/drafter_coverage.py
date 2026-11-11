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
inside the checkpoint) the target's entry must list `speculator`. A target the catalog lacks is
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
        self.models: dict[str, dict] = {}
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

    def ships_a_draft(self, target: str) -> bool:
        """Whether a drafter the target lists is shipped inside the target's own repository."""
        repo = ((self.models[target].get("source") or {}).get("repo") or "").casefold()
        return any(d in self.models and (self.models[d].get("source") or {}).get("repo", "").casefold() == repo
                   for d in self.models[target].get("drafters") or [] if d != "speculator")


def check(cat: Catalog, deployments: list[Deployment]) -> dict[str, list[str]]:
    """Gaps by kind, each a sorted list of unique lines."""
    gaps: dict[str, set[str]] = {"missing_model": set(), "missing_drafter": set(),
                                 "unlinked_drafter": set(), "missing_speculator": set()}
    for d in deployments:
        target = cat.entry(d.target)
        if target is None:
            gaps["missing_model"].add(d.target)
            continue
        listed = (cat.models[target].get("drafters") or [])
        if not d.draft:
            if "speculator" not in listed and not cat.ships_a_draft(target):
                gaps["missing_speculator"].add(
                    f"{d.target}: {d.method} without a draft checkpoint, and {target} lists no speculator")
            continue
        drafter = cat.drafter(d.draft)
        if drafter is None:
            gaps["missing_drafter"].add(f"{d.draft} ({d.method}) drafts for {d.target}")
            continue
        names = cat.models[drafter].get("targets") or []
        if target not in names or drafter not in listed:
            gaps["unlinked_drafter"].add(
                f"{drafter} for {target}: targets {names}, {target} lists {listed}")
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
    missing = any(gaps[k] for k in ("missing_drafter", "unlinked_drafter", "missing_speculator"))
    return 1 if missing and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
