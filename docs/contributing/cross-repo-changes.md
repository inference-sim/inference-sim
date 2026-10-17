# Changes across the BLIS repositories

The process for a change that spans blis-schemas, blis-catalog, blis-registry,
blis-latency-kernel and inference-sim is described once, in blis-schemas:
[Changes across the repositories](https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md). This page lists only what is specific to inference-sim,
which is the last repository in the dependency order: it is pushed and merged after the other four.
The per-PR workflow ([PR Workflow](pr-workflow.md)) and the standards
([rules](standards/rules.md), [invariants](standards/invariants.md)) apply unchanged; the canonical
page adds the cross-repository steps around them.

## Repinning

1. `go.mod`: the kernel and blis-schemas at their pushed heads' pseudo-versions (release tags
   before merge). Never commit a `replace`; run `go mod tidy`, and tidy the
   `llmd-conformance` modules too.
2. `testdata/registry` and `testdata/catalog`: re-vendor from the registry and catalog heads,
   updating each `.upstream-commit` stamp.
3. `testdata/upstream-prerelease.lock`: the same two heads (TEMPORARY). Empty it once the
   kernel's `testdata/upstream.lock` names their releases;
   `TestTheVendoredCatalogAndRegistryAreTheLockedReleases` then holds the vendored copies to it.
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
