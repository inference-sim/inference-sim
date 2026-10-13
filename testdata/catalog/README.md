# Vendored blis-catalog subset

A verbatim subset of [`blis-catalog`](https://github.com/inference-sim/blis-catalog) release
**`0.2.1`** (commit recorded in `.upstream-commit`), shaped as a catalog clone root so the Go
tests can resolve models, presets, chips and devices offline. Nothing here is edited: a file is
either copied unchanged from that release or absent. Operators do not use this copy; they clone
the release and point `--catalog` / `BLIS_CATALOG` at it
([installation](../../docs/getting-started/installation.md#catalog-compatibility)).

| path | what | read by |
|---|---|---|
| `models/<name>/` | `config.json`, `graph.yaml`, `model.yaml` for 20 models | the HF-config backends, the kernel, the catalog gate |
| `hardware/`, `networks/` | chip and fabric descriptors | the kernel, the catalog gate |
| `devices/storage.yaml` | KV-offload storage tiers | `--kv-offload-config` device classes, the legacy CPU tier |
| `workloads/` | the four named presets | `run --workload`, `observe --workload`, `convert preset` |

The models are those the cmd tests name plus the nine the blis-latency-kernel scenario fixtures
use. Every file format is blis-schemas', and BLIS reads each through that package's own loaders.

The release is not chosen here. blis-latency-kernel's `testdata/upstream.lock` pins the catalog
and registry its fixtures resolve against, and
`TestTheVendoredCatalogAndRegistryAreTheLockedReleases` (`sim/kernelmodel`) fails when this
copy's `.upstream-commit` differs from it. `TestCatalogStrictLoad_CommittedFixtureCatalog`
(`cmd`) requires the copy to load with no problems through the production readers.

## Updating

When `go.mod` moves blis-latency-kernel to a release whose lock names a newer catalog:

```bash
git clone --branch <tag> --depth 1 https://github.com/inference-sim/blis-catalog.git /tmp/c
for m in $(ls testdata/catalog/models); do rm -rf testdata/catalog/models/$m; cp -R /tmp/c/models/$m testdata/catalog/models/; done
rm -rf testdata/catalog/{hardware,networks,devices}; cp -R /tmp/c/{hardware,networks,devices} testdata/catalog/
cp /tmp/c/workloads/*.yaml testdata/catalog/workloads/
git -C /tmp/c rev-parse HEAD > testdata/catalog/.upstream-commit
go test ./...
```

`testdata/registry/` follows the same rule for blis-registry (`coefficients/` only), with its
own `.upstream-commit`.
