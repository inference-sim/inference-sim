# InferenceX validation harness

Replays InferenceX end-to-end serving results (NVIDIA, vLLM family) in BLIS and scores them with
InferenceX's own metric definitions over matched windows. It produces the HELD-out error tables
(MAPE and signed median per metric, post-stratified, run-clustered bootstrap CI), the distribution
distances (W1/median, KS, quantile errors) and the KV / server-state comparison quoted in the BLIS
PR bodies.

This directory holds code only. No InferenceX measurement is shipped. You fetch the data yourself,
into a directory you choose, and read it as untrusted input (`python -I`, never executed).

## Data (public; fetched by you; never redistribute)

| What | Source | Used by |
|---|---|---|
| benchmark rows, server logs | InferenceX DB dump `db-dump/2026-09-28`, https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-28, restored into Postgres (database `$IX_PGDATABASE`, default `inferencex`) | `partition/build_inventory.py`, `cov.py`, `kvmeas.py`, `logfacts.py`, `parity.py` |
| request timelines, 1 s server series, post-dump server logs | InferenceX Data API https://inferencex.semianalysis.com/api/v1/{request-timeline,trace-server-metrics,server-log-files,server-log} (contract `/api/openapi.json`) | `distance.py`, `perreq*.py`, `kvmeas.py` |
| launch recipes for runs without a log | `SemiAnalysisAI/InferenceX` at each run's `head_sha` (`benchmarks/*.sh`, `.github/`) into `$IX_RECIPES/<sha>/` | `partition/reconstruct.py`, `cov.py` |
| agentic trace corpora | HF `semianalysisai/cc-traces-weka-062126` @ `23f152f6`, `semianalysisai/cc-traces-weka-062126-256k` @ `8fecd2fc` (resolved after download by the LFS oid; the generated workload still says `revision: unknown`) | `cov.py` |
| vLLM source with tags | https://github.com/vllm-project/vllm into `$IX_VLLM_REPO` | `build_table.py`, `spec.py` |

Fetch scripts (all write under `IX_DATA_DIR`; downloads are untrusted data, verified by sha256 where the source
publishes one, parsed as text or JSON and never executed or sourced):

| Script | Fetches | Pin |
|---|---|---|
| `fetch/db_dump.sh` | the 17-part DB dump, then `pg_restore` into `$IX_PGDATABASE` (`--no-restore` to skip) | release `db-dump/2026-09-28`, per-part sha256 in the script |
| `fetch/db_export.sh` | `db/{rows,logs,logs2,heads}.tsv` from the restored dump (the exact queries behind the inventory) | the dump |
| `fetch/gh_bmk.sh` | post-dump `bmk_*` artifact JSON of InferenceX "Run Sweep" runs 2026-09-27..10-16 into `gh-bmk-postdump/` (needs `gh auth`) | run/artifact ids; zip sha256 in `index.tsv` |
| `fetch/recipes.sh inputs/nolog_shas.txt` | launcher and workflow files at the 41 InferenceX commits behind the no-log runs into `$IX_RECIPES` | commit sha |
| `fetch/corpora.sh` | both trace corpora into `$IX_CORPORA` | HF commit + sha256 |
| `fetch/fetch.sh`, `fetch/chunk.sh series\|logs <idfile>` | Data API per-result bodies (timelines, server series, post-dump logs) into `$IX_API` | none (live API; `FETCHLOG.tsv` records URL + sha256) |

`inputs/` holds identifiers only, no measurements: `dsv41_binding_ids.json` (57 DSV4.1 Hopper DB row ids, each with the
GitHub artifact path and the sha256 of the `profile_export.jsonl` bound to it; the binding itself recomputed each row's
metrics from the artifact with InferenceX's own code), `inventory_extra.jsonl` (run descriptors from the Data API, plus
ml-energy/benchmark-v3 descriptors kept as `gated` and never partitioned, plus 6 vLLM-PR descriptors that are blocked)
and `nolog_shas.txt`. The binding only marks those rows as request-level; the inventory sha needs it, scoring does not.

Every location is an environment variable with a default under `IX_DATA_DIR` (default `./ix-data`); see
`paths.py`. `IX_PINS` is a JSON file `{"blis_binary", "registry_root", "catalog_root", "inference_sim",
"blis_latency_kernel", "blis_schemas", "blis_catalog", "blis_registry"}` naming the binary, snapshot
directories and PR heads; `cov.py` stamps it into every result.

## Pre-registered splits

- **FIT / HELD partition** (`partition/draw_partition.py`, seed `20261016`, FIT share 0.40): hash per
  configuration group plus stratification repair over model, gpu, engine image, layout, P/D, spec,
  offload, frontend, scenario and concurrency bin. The rules are in the script's docstring. The frozen
  v5 assignment is `partition/partition.v5.json.gz` (gunzip to `partition.v5.json`, sha256
  `2365af27e95f34ed7e9bdf1efd5cd5cb812bb5a640837876c66a908fed7578d8`). It holds InferenceX point ids,
  group keys (configuration fields only), the hash draw `u`, set assignments and coverage counts. It
  holds no measurement. A fresh draw differs from v5 by 3 of 56,332 points, because v5 was frozen from
  earlier versions under the EXTENSION RULE; use the shipped file. Its `frozen_from.path` is historical (the path of v4 on the
  machine that froze it); the v4 sha256 beside it is what identifies the parent.
- **HELD order** (`cov.py plan`, `HELD_SEED = 20261017`): the DSV4.1 Hopper anchors first, then a seeded
  stratified order.
- **r6 vs r7 subsample** (`subsample.py`, seed `7017`, 25% per plan stratum plus the anchors in full),
  scored by `compare.py`.

## Reproducing the HELD table

Rebuild the inventory (sha256 678f244a... with catalog a2a8e47 and registry 2e7070c, the trees it was built at):

```bash
fetch/db_dump.sh && fetch/db_export.sh && fetch/gh_bmk.sh && fetch/recipes.sh inputs/nolog_shas.txt
partition/rebuild_inventory.sh <blis-catalog@a2a8e47> <blis-registry@2e7070c>   # inventory.jsonl + recipe_recon.jsonl
```

Build BLIS at the inference-sim PR head (`go build -o blis .`). Export `blis-registry` and `blis-catalog` at
their PR heads (`git archive <sha> | tar -x -C <dir>`). Write `$IX_PINS`. Then:

```bash
export IX_DATA_DIR=$PWD/ix-data PY="uv run --no-project --with numpy python -I"
gunzip -k partition/partition.v5.json.gz
fetch/corpora.sh
$PY build_table.py                       # version-dependent vLLM defaults -> $IX_BEHAVIOUR
$PY cov.py plan                          # -> $IX_ROOT/work-r7/plan.jsonl
$PY cov.py run --set HELD --width 2      # simulate + score (resumable)
$PY kv_units.py $IX_ROOT/work-r7        # pool-size consistency per log (kv_units_differ)
$PY dist_backfill.py $IX_ROOT/work-r7
$PY heldout_report.py                    # post-stratified HELD MAPE + CIs
$PY subsample.py && $PY compare.py       # r6 -> r7 on the pre-registered subsample
$PY kvmeas.py && $PY kvrun.py --blis <blis> && $PY kvcompare.py && $PY kvreport.py   # KV metrics
```

`heldout_report.py` and `aggregate.py` read `$IX_ROOT/work`, and `compare.py` reads both `work` (r6 build) and
`work-r7`. Run `cov.py` once per build, setting `cov.WORK` accordingly (see `run_fit_agentic.py`).
`kv_units_differ` reads `<work>/kv_units.json`, built by `kv_units.py` from the logs of the scored points. Run it
after `cov.py run` and re-score (`cov.py run` reuses the simulations), or the KV-pool rows are scored without it.

## Tests

```bash
uv run --no-project --with pytest --with numpy python -I -m pytest -q tools/validation/inferencex
```

The tests use synthetic inputs. Seven resolver tests need `$IX_BEHAVIOUR` and skip without it. One
anchor-reproduction test needs `IX_R6_DIR` and skips without it. inference-sim CI runs Go only, so these
tests are not run in CI. Run them by hand.

## See also

- blis-registry `docs/fitting-principles.md`: how coefficients are fitted and held out.
- `docs/contributing/cross-repo-changes.md` (here and in each BLIS repo): the cross-repository change process.
