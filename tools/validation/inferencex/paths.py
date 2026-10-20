"""Locations the harness reads and writes. Every one is an environment variable with a default
relative to one user-chosen data directory (IX_DATA_DIR, default ./ix-data). Nothing here is shipped
with the repository: the user fetches the InferenceX data into IX_DATA_DIR (see README.md).

  IX_DATA_DIR      root of the fetched data (default ./ix-data)
  IX_ROOT          harness work root: work/, work-r7/, results/, kvm/, r7cmp/ (default $IX_DATA_DIR/coverage)
  IX_INVENTORY     inventory.jsonl from partition/build_inventory.py (default $IX_DATA_DIR/inventory.jsonl)
  IX_PARTITION     partition.json (default: partition/partition.v5.json beside this file, gunzipped by the user)
  IX_RECIPE_RECON  recipe_recon.jsonl from partition/reconstruct.py (default $IX_DATA_DIR/recipe_recon.jsonl)
  IX_RECIPES       InferenceX recipe trees by commit, <sha>/benchmarks/*.sh (default $IX_DATA_DIR/recipes)
  IX_API           InferenceX Data API per-result bodies (default $IX_DATA_DIR/inferencex-api/per-result)
  IX_CORPORA       aiperf trace corpora, <name>/traces.jsonl (default $IX_DATA_DIR/corpora)
  IX_BEHAVIOUR     behaviour_by_image.json from build_table.py (default $IX_DATA_DIR/behaviour_by_image.json)
  IX_PINS          pins JSON: blis_binary, registry_root, catalog_root, PR heads (default $IX_DATA_DIR/pins.json)
  IX_VLLM_REPO     a vllm-project/vllm clone with tags (default $IX_DATA_DIR/vllm)
  IX_PGDATABASE    the restored InferenceX db-dump database read by psql (default inferencex)
  IX_HEAVY         optional wrapper prefixed to each simulation command (default: none)
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _p(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


DATA = _p("IX_DATA_DIR", Path("ix-data").resolve())
ROOT = _p("IX_ROOT", DATA / "coverage")
INV = _p("IX_INVENTORY", DATA / "inventory.jsonl")
PART = _p("IX_PARTITION", HERE / "partition" / "partition.v5.json")
RECON = _p("IX_RECIPE_RECON", DATA / "recipe_recon.jsonl")
RECIPES = _p("IX_RECIPES", DATA / "recipes")
API = _p("IX_API", DATA / "inferencex-api" / "per-result")
CORPORA = _p("IX_CORPORA", DATA / "corpora")
BEHAVIOUR = _p("IX_BEHAVIOUR", DATA / "behaviour_by_image.json")
PINS = _p("IX_PINS", DATA / "pins.json")
VLLM_REPO = _p("IX_VLLM_REPO", DATA / "vllm")
PGDATABASE = os.environ.get("IX_PGDATABASE", "inferencex")
HEAVY = os.environ.get("IX_HEAVY", "")


def heavy(cmd: list[str]) -> list[str]:
    return ([HEAVY] if HEAVY else []) + cmd
