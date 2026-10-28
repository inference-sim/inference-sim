# Changes across the BLIS repositories

The process for a change that spans blis-schemas, blis-catalog, blis-registry, blis-vllm,
blis-latency-kernel and inference-sim is described once, in blis-schemas:
[Changes across the repositories](https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md). This page lists only what is specific to inference-sim,
which is the last repository in the dependency order: it is pushed and merged after the others
(blis-schemas, then blis-vllm, then blis-latency-kernel, #2021).
The per-PR workflow ([PR Workflow](pr-workflow.md)) and the standards
([rules](standards/rules.md), [invariants](standards/invariants.md)) apply unchanged; the canonical
page adds the cross-repository steps around them.

## Repinning

1. `go.mod`: the kernel, blis-schemas and the two blis-vllm modules (`blis-vllm` and
   `blis-vllm/rules`) at their pushed heads' pseudo-versions (release tags before merge), then
   the llm-d router fork's `replace` at its `vX.Y.Z-blis.N` tag, after blis-schemas because the
   fork implements blis-schemas `embed` ([The llm-d Router Fork](llm-d-router-fork.md)). Never
   commit a `replace` for a blis-* module (the llm-d router's `replace` to the inference-sim fork
   is the one exception); run `go mod tidy`.
2. `testdata/registry` and `testdata/catalog`: re-vendor from the registry and catalog heads,
   updating each `.upstream-commit` stamp.
3. `testdata/upstream-prerelease.lock`: the same two heads (TEMPORARY). Empty it once the
   kernel's `testdata/upstream.lock` names their releases;
   `TestThisRepositorysVendoredCatalogAndRegistryAreTheLockedReleases` (`sim/vendored`) holds this
   repository's vendored copies to it, as `TestTheVendoredCatalogAndRegistryAreTheLockedReleases`
   (blis-vllm `compat/kernelmodel`, moved from `sim/kernelmodel` by #2021) holds blis-vllm's own.
4. Documentation that pins a catalog release (`docs/getting-started/installation.md`) moves with
   it; `TestDocExamplesPinTheCatalogVersion` checks it.
5. `go test ./...` and `golangci-lint run ./...`.

## The integrated build

Gates and published numbers come from one binary built from the integrated head with its
revision injected:

```bash
go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRevision=$(git rev-parse HEAD)" -o blis main.go
```

It reads the registry and catalog from read-only snapshots of the pinned commits, never from a
working tree. The pull request body records the binary's revision and the five head SHAs; the
held-out and gate evidence the canonical page requires goes under *Held-out and gate evidence*
in the template.

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
