# Model Compatibility

A model runs in BLIS when three things exist for it:

1. **A catalog entry** in [`blis-catalog`](https://github.com/inference-sim/blis-catalog) at
   `<catalog>/models/<name>/`: the vendor's HuggingFace `config.json` (committed verbatim),
   `model.yaml` (identity and provenance), and `graph.yaml`, the model graph
   blis-latency-kernel prices.
2. **A scenario** naming the model, a catalog chip (`cluster.hardware`) and, when a handoff
   crosses nodes, a catalog fabric, plus the coefficient sets it needs.
3. **Those coefficient sets** in [`blis-registry`](https://github.com/inference-sim/blis-registry).

Anything else is refused at startup, naming what is missing. BLIS never fetches a model at
run time, and adding a model, chip or fabric is a commit to blis-catalog (and, when new
coefficients are needed, to blis-registry), not a change to this repo. How the kernel prices
a model is described in [Latency Models](../guide/latency-models.md).

The catalog is located by `--catalog <path>` or the `BLIS_CATALOG` environment variable;
there is no default and no search path, so a run with neither is refused naming both forms
(#1731). See [Catalog compatibility](../getting-started/installation.md#catalog-compatibility)
for the pinned release.

## Scored deployments

blis-latency-kernel `v0.1.0` ships the scenarios it was scored on in
`testdata/aisimulate` of its module, covering deepseek-v3, deepseek-v4-pro, glm-5,
gpt-oss-120b, kimi-k2.5, llama-3.1-70b-instruct, minimax-m2.5, minimax-m3 and
qwen3.5-397b-a17b on H100, H200, B200 and B300 under several TP degrees and precisions. The
accuracy figures are in [Latency Models](../guide/latency-models.md#the-published-accuracy-figures).
A model or chip outside that set runs if the catalog and registry cover it, but its numbers
are unscored: treat absolute latencies as estimates.

## Catalog validation -- there is no `validate` command

BLIS validates whatever it reads and fails naming the file and the problem; there is
deliberately no `blis validate` subcommand, because a separate validator would be free to
accept a catalog a run rejects. The catalog is kept loadable by a whole-catalog load through
the same code path a run uses (#1750), in two layers:

- **Unconditional** -- `go test ./cmd/...` loads the committed fixture catalog
  (`testdata/catalog`, a vendored subset of blis-catalog `0.2.1`) on every test run.
- **Against the authoritative catalog** -- `scripts/catalog-load-gate.sh` clones
  `blis-catalog` at a pinned revision and runs the same load over every entry. Wiring it into
  CI as a `catalog-load` job is a pending human step, tracked by
  [#1823](https://github.com/inference-sim/inference-sim/issues/1823); until then it runs on
  demand only.

What the load requires of a catalog:

| Rule | Where it applies |
|---|---|
| Every `models/<name>/` entry has its vendor `config.json` **and** `model.yaml`, whose `name` matches the directory | `models/` |
| Strict parsing: an **unknown key is a hard error naming the file and the key** | catalog-authored YAML |
| **One YAML document per file** (an empty trailing `---` is fine) | catalog-authored YAML |
| No catalog-authored file states a **GPU** or a **tensor-parallel degree**: those are deployment choices, stated in the scenario | catalog-authored YAML |

The deployment-fact rule is scoped to the catalog's own YAML, never the vendor
`config.json`: many vendor configs state `pretraining_tp`, an architectural fact of the
checkpoint rather than a serving choice. The file formats themselves are defined by
[`blis-schemas`](https://github.com/inference-sim/blis-schemas).
