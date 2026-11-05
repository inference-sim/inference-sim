# Vendored catalog subset

A verbatim subset of blis-substrate's `catalog/`, shaped as a catalog root so the Go
tests can resolve models, presets, chips and devices offline. Nothing here is edited: a file is
either copied unchanged from that release or absent. Operators do not use this copy; they clone
the release and point `--catalog` / `BLIS_CATALOG` at it
([installation](../../docs/getting-started/installation.md#catalog-compatibility)).

| path | what | read by |
|---|---|---|
| `models/<name>/` | `config.json`, `graph.yaml`, `model.yaml` for 24 models | the HF-config backends, the kernel, the catalog gate |
| `hardware/`, `networks/` | chip and fabric descriptors | the kernel, the catalog gate |
| `devices/storage.yaml` | KV-offload storage tiers | `--kv-offload-config` device classes, the legacy CPU tier |
| `workloads/` | the named presets, and the `agentx-mvp` replay workload (blis-catalog#34) | `run --workload`, `observe --workload`, `convert preset`; the replay through `replay --workload-spec` |

The models are those the cmd tests name, the nine the cost model's scenario fixtures
use, and deepseek-v4.1-flash for `testdata/scenarios/deepseek-v4.1-flash-h100-tp8-dspark5.yaml`. Every file format is blis-substrate's (contracts/spec), and BLIS reads each through that package's own loaders.

The release is not chosen here. These files are byte-identical to blis-substrate's `catalog/` at
the release `go.mod` pins (blis-substrate 4eb16d8 copied blis-catalog 0.5.0 unchanged). This
copy, and `testdata/registry/`, are removed when blis-substrate's loader (`load/`) is released
and `compat/kernelmodel` reads the substrate's embedded data instead.
