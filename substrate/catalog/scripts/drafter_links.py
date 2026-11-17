"""Which catalog drafter a run drafts with: the one rule every script that resolves a drafter uses.

    from drafter_links import Catalog, resolve, Unresolved

The catalog links a target and its drafters both ways (design D8, blis-substrate#109): a target's
model.yaml lists its drafters (`drafters: [speculator, <entry>, ...]`, where `speculator` is its
own in-graph head), and a drafter entry names its targets (`targets: [...]`). `resolve` turns a
run's stated draft checkpoint into the catalog entry it is, or "" for the model's own in-graph
speculator:

  - none stated, `native`, or the target's own repository: the model's own speculator when its
    graph speculates by the run's method; otherwise the one drafter entry the model lists that
    ships in the model's own repository (a DSpark head in a dspark/ subdirectory, which the
    engine loads from the target's repository);
  - a Hugging Face repository: the drafter entry with that `source.repo` (case-insensitive) that
    names the model as a target and is listed by it, and whose graph speculates by the run's
    method;
  - anything else (a local path) names no repository and is refused.

A drafter that does not resolve raises Unresolved with the reason. The catalog is read as text
(`key: value`, flow and block lists), so the module needs no YAML library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = re.compile(r"[\w.-]+/[\w.-]+")


class Unresolved(Exception):
    """A stated drafter that is no catalog drafter of the model, with the reason."""


def _flow_list(text: str, key: str) -> list[str]:
    """A top-level list, written as a flow list (`key: [a, b]`) or a block list (`key:` then
    `- a` lines)."""
    if m := re.search(rf"^{key}:\s*\[([^\]]*)\]\s*$", text, re.M):
        return [x.strip() for x in m.group(1).split(",") if x.strip()]
    if m := re.search(rf"^{key}:\s*\n((?:[ \t]*-[ \t]+\S+[ \t]*\n?)+)", text, re.M):
        return re.findall(r"^[ \t]*-[ \t]+(\S+)", m.group(1), re.M)
    return []


def _source_repo(text: str) -> str:
    m = re.search(r"^source:\s*\n(?:[ \t]+.*\n)*?[ \t]+repo:\s*(\S+)\s*$", text, re.M)
    return m.group(1) if m else ""


def _speculator_method(graph: str) -> str:
    """The method a graph's speculator states, as a deployment states it: every model-specific
    MTP name is mtp (derive_graph.vllm_speculative_method)."""
    m = re.search(r"^speculator:\s*\n(?:  .*\n)*?  method:\s*(\S+)\s*$", graph, re.M)
    if not m:
        return ""
    return "mtp" if m.group(1).endswith("_mtp") else m.group(1)


@dataclass
class Entry:
    name: str
    repo: str
    drafters: list[str]
    targets: list[str]
    speculator: str


class Catalog:
    def __init__(self, root: Path):
        self.entries: dict[str, Entry] = {}
        for p in sorted((Path(root) / "models").glob("*/model.yaml")):
            text = p.read_text()
            g = p.parent / "graph.yaml"
            self.entries[p.parent.name] = Entry(
                p.parent.name, _source_repo(text), _flow_list(text, "drafters"),
                _flow_list(text, "targets"), _speculator_method(g.read_text()) if g.is_file() else "")

    def entry(self, name: str) -> Entry:
        if name not in self.entries:
            raise Unresolved(f"{name} is not a catalog model")
        return self.entries[name]


def resolve(cat: Catalog, model: str, draft: str | None, method: str | None = None) -> str:
    """The catalog drafter entry of `model` that `draft` names, or "" for its own speculator.
    `method`, when given, is the run's speculative method, which the drafter must speculate by."""
    target = cat.entry(model)
    ok = lambda e: method is None or e.speculator == method  # noqa: E731
    if not draft or draft == "native" or draft.casefold() == target.repo.casefold():
        if "speculator" in target.drafters and (method is None or target.speculator == method):
            return ""
        own = [d for d in target.drafters if d in cat.entries
               and cat.entries[d].repo.casefold() == target.repo.casefold() and ok(cat.entries[d])]
        if len(own) == 1:
            return own[0]
        how = f"by {method}" if method else "by any method"
        raise Unresolved(f"{model} drafts {how} with no separate checkpoint, but its graph does not "
                         "speculate that way and it lists no single drafter entry in its own repository")
    if not REPO.fullmatch(draft):
        raise Unresolved(f"the draft checkpoint {draft} names no repository, so no catalog drafter "
                         "can be matched to it")
    hits = [e for e in cat.entries.values()
            if e.repo.casefold() == draft.casefold() and model in e.targets]
    if not hits:
        raise Unresolved(f"its drafter {draft} is not in the catalog as a drafter of {model}")
    if len(hits) > 1:
        raise Unresolved(f"drafter {draft} matches several catalog entries: {sorted(e.name for e in hits)}")
    e = hits[0]
    if e.name not in target.drafters:
        raise Unresolved(f"its drafter {draft} is the catalog's {e.name}, which {model} does not list "
                         "among its drafters")
    if not ok(e):
        raise Unresolved(f"its drafter {draft} is the catalog's {e.name}, whose graph speculates by "
                         f"{e.speculator or 'no method'}, not {method}")
    return e.name
