#!/usr/bin/env python3
"""Extract the seed measured run from the aiconfigurator survey excerpt.

    uv run --no-project --with pyyaml python -I scripts/measurements/aiconfigurator_example.py \
        --workspace <survey workspace> --out <run.yaml> [--check]

The survey workspace holds data/external/aiconfigurator/, with the CSV excerpt of
ai-dynamo/aiconfigurator's silicon_sample.csv and the PROVENANCE.txt recording where it
came from. This script reads one sweep from the excerpt, gpt-oss-120b on H100 at TP4 with
1024/1024 tokens under vLLM 0.18.0, and writes it as a MeasuredRun to --out. Every
number in the output is read from those two files or computed from their bytes; the
script states only which rows to select.

It is the catalog's example of the measurements/ format, not an adapter: the adapters
live in inference-sim's tools/validation/. Its output is not committed; the catalog
points at the source by URL and sha256 instead. With --check it re-extracts and fails if
the file at --out differs, without writing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import yaml

CATALOG = Path(__file__).resolve().parents[2]
NAME = "aiconfigurator-gpt-oss-120b-h100-tp4-1024x1024-vllm-0.18.0"
EXCERPT = "data/external/aiconfigurator/silicon_sample_vllm_hopper_excerpt.csv"
PROVENANCE = "data/external/aiconfigurator/PROVENANCE.txt"

# The sweep: one configuration from one source file of the excerpt, every batch size.
SELECT = {
    "backend": "vllm", "backend_version": "0.18.0", "model_path": "openai/gpt-oss-120b",
    "system": "h100_sxm", "mode": "agg", "isl": "1024", "osl": "1024", "tp_size": "4",
    "pp_size": "1", "attention_dp_size": "1", "moe_tp_size": "4", "moe_ep_size": "1",
}
SOURCE_FILE = "source_file=2026-04-06.csv"
# aiconfigurator's system names for the catalog chips they are.
HARDWARE = {"h100_sxm": "h100", "h200_sxm": "h200"}
CONFIG_COLUMNS = ["tp_size", "pp_size", "attention_dp_size", "moe_tp_size", "moe_ep_size",
                  "isl", "osl", "precision", "gemm_quant_mode", "moe_quant_mode"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(path: Path) -> dict[str, str]:
    """Reads the fields the run cites from the survey's PROVENANCE.txt."""
    text = path.read_text()

    def field(pattern: str) -> str:
        m = re.search(pattern, text, re.M)
        if not m:
            raise SystemExit(f"{path}: no match for {pattern!r}")
        return m.group(1)

    return {
        "repo": field(r"^Source: (\S+) \(NVIDIA\)"),
        "path": field(r"^Source: \S+ \(NVIDIA\), file (\S+)$"),
        "commit": field(r"^Repo commit: ([0-9a-f]{40})$"),
        "url": field(r"^Full-file URL: (\S+)$"),
        "retrieved": field(r"^Retrieved \(UTC\): (\S+)$"),
        "size": field(r"^Full file:\s+(\d+) bytes"),
        "sha256": field(r"^Full file:.*sha256 ([0-9a-f]{64})"),
        "excerpt_sha256": field(r"^Excerpt sha256: ([0-9a-f]{64})$"),
        "excerpt_note": field(r"^(Stored excerpt: .*)$"),
        "statistic_note": field(r"(Columns silicon_ttft_ms/silicon_tpot_ms: .*)$"),
    }


def catalog_model(repo: str) -> str:
    """Names the catalog model whose model.yaml records repo as its source."""
    names = [p.parent.name for p in sorted(CATALOG.glob("models/*/model.yaml"))
             if (yaml.safe_load(p.read_text()).get("source") or {}).get("repo") == repo]
    if len(names) != 1:
        raise SystemExit(f"expected one catalog model with source.repo {repo}, found {names}")
    return names[0]


def rows(excerpt: Path) -> list[dict[str, str]]:
    with excerpt.open(newline="") as f:
        picked = [r for r in csv.DictReader(f)
                  if all(r[k] == v for k, v in SELECT.items())
                  and r["silicon_data_source"].startswith(SOURCE_FILE + ";")]
    if not picked:
        raise SystemExit(f"{excerpt}: no rows match the selection")
    for r in picked:
        m = re.search(r"silicon_concurrency=(\d+);", r["silicon_data_source"])
        if not m or m.group(1) != r["batch_size"]:
            raise SystemExit(f"{r['id']}: batch_size {r['batch_size']} is not the recorded concurrency")
    picked.sort(key=lambda r: int(r["batch_size"]))
    if len({r["batch_size"] for r in picked}) != len(picked):
        raise SystemExit("two rows share a batch size")
    return picked


def unstated(setting: str, value: str | None, note: str, material: str = "yes",
             column: str | None = None) -> dict:
    row = {"setting": setting}
    if value is not None:
        row["report"] = {"value": value, "ref": f"{EXCERPT} column {column or setting}"}
    row.update({"verdict": "UNSTATED", "material": material, "note": note})
    return row


def build(workspace: Path) -> str:
    excerpt, prov_path = workspace / EXCERPT, workspace / PROVENANCE
    prov = provenance(prov_path)
    if sha256(excerpt) != prov["excerpt_sha256"]:
        raise SystemExit(f"{excerpt}: sha256 differs from the one {prov_path} records")
    picked = rows(excerpt)
    first = picked[0]
    image = re.search(r"image=(\S+)$", first["silicon_data_source"]).group(1)
    model = catalog_model(first["model_path"])

    provenance_rows = [
        {"setting": "model", "report": {"value": first["model_path"], "ref": f"{EXCERPT} column model_path"},
         "blis": {"value": model, "ref": f"models/{model}/model.yaml source.repo"},
         "verdict": "MATCH", "material": "yes"},
        {"setting": "hardware", "report": {"value": first["system"], "ref": f"{EXCERPT} column system"},
         "blis": {"value": HARDWARE[first["system"]]}, "verdict": "MATCH", "material": "yes",
         "note": "aiconfigurator system name mapped to the catalog chip"},
        unstated("server image", image, "a tag, not a digest", material="unknown",
                 column="silicon_data_source"),
    ]
    for col in CONFIG_COLUMNS:
        provenance_rows.append(unstated(col, first[col], "stated by the report only; no run artifact"))
    provenance_rows.append(unstated(
        "engine flags", None,
        "max_num_batched_tokens, chunked prefill, gpu_memory_utilization and CUDA graphs are not recorded",
        material="unknown"))
    provenance_rows.append(unstated("metric statistic", None, prov["statistic_note"]))

    run = {
        "kind": "MeasuredRun",
        "name": NAME,
        "schema_version": 1,
        "source": {
            "kind": "aiconfigurator",
            "adapter": {"name": "blis-catalog/scripts/measurements/aiconfigurator_example.py",
                        "version": "1", "sha256": sha256(Path(__file__))},
            "inputs": [
                {"role": "report", "url": prov["url"], "repo": prov["repo"], "commit": prov["commit"],
                 "path": prov["path"], "size": int(prov["size"]), "sha256": prov["sha256"],
                 "fetched_utc": prov["retrieved"],
                 "note": "the full file; not stored in the survey, only its digest"},
                {"role": "raw_artifact", "path": EXCERPT, "size": excerpt.stat().st_size,
                 "sha256": sha256(excerpt), "note": prov["excerpt_note"]},
            ],
            "extraction": {"method": "csv", "selectors": [r["id"] for r in picked]},
            "retrieved_utc": prov["retrieved"],
            "license_ref": "aiconfigurator",
        },
        "model": model,
        "engine_version": first["backend_version"],
        "client": {},
        "metric_defs": {
            "tpot": {"unit": "ms", "cite": f"{EXCERPT} column silicon_tpot_ms; {prov['statistic_note']}"},
            "ttft": {"formula": "ttft", "unit": "ms",
                     "cite": f"{EXCERPT} column silicon_ttft_ms; {prov['statistic_note']}"},
        },
        "points": [
            {"id": f"c{r['batch_size']}", "load": {"concurrency": int(r["batch_size"])},
             "metrics": {
                 "tpot": {"value": float(r["silicon_tpot_ms"]), "report_label": "silicon_tpot_ms", "raw_key": r["id"]},
                 "ttft": {"value": float(r["silicon_ttft_ms"]), "report_label": "silicon_ttft_ms", "raw_key": r["id"]},
             }}
            for r in picked
        ],
        "provenance": provenance_rows,
        "verdict": "NOT_REPRODUCIBLE",
    }
    header = ("# Generated by scripts/measurements/aiconfigurator_example.py. Do not edit.\n"
              "# aiconfigurator records parallelism and lengths only, so no cluster, deployment\n"
              "# or request stream is stated; validate-catalog recomputes the verdict.\n")
    return header + yaml.safe_dump(run, sort_keys=False, allow_unicode=True, width=100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True,
                    help="the survey workspace holding data/external/aiconfigurator/")
    ap.add_argument("--out", type=Path, required=True, help="where to write the MeasuredRun")
    ap.add_argument("--check", action="store_true", help="fail if the file at --out differs; write nothing")
    args = ap.parse_args()
    text = build(args.workspace)
    out = args.out
    if args.check:
        if not out.exists() or out.read_text() != text:
            print(f"{out} differs from a fresh extraction", file=sys.stderr)
            return 1
        print(f"{out} is current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
