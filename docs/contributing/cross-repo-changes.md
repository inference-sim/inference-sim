# Changes across the BLIS repositories

A change that spans blis-substrate, blis-vllm, the llm-d router fork and inference-sim is one
review set, released bottom-up (design X10). This page lists only what is specific to
inference-sim, which is pushed and merged after the repositories below it: blis-substrate, then
blis-vllm and the router fork.
The per-PR workflow ([PR Workflow](pr-workflow.md)) and the standards
([rules](standards/rules.md), [invariants](standards/invariants.md)) apply unchanged; the canonical
page adds the cross-repository steps around them.

## Repinning

1. `go.mod`: blis-substrate and blis-vllm at their pushed heads' pseudo-versions (release tags
   before merge), then the llm-d router fork's `replace` at its `vX.Y.Z-blis.N` tag
   ([The llm-d Router Fork](llm-d-router-fork.md)). The fork's `replace` is the only one
   (design X11), and `exclude github.com/inference-sim/blis-vllm v0.32.0` stays. Run
   `go mod tidy`.
2. The catalog and registry need no step: the tests read the `catalog/` and `registry/`
   directories of the blis-substrate module that `go.mod` selects
   (`internal/substratedata`), so the substrate pin is the one place that names them.
3. `go test ./...` and `golangci-lint run ./...`.

## The integrated build

Gates and published numbers come from one binary built from the integrated head with its
revision injected:

```bash
go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRevision=$(git rev-parse HEAD)" -o blis main.go
```

It reads the catalog and registry embedded in, or exported from, the pinned blis-substrate
commit, never a working tree. The pull request body records the binary's revision and the five head SHAs; the
held-out and gate evidence the canonical page requires goes under *Held-out and gate evidence*
in the template.

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
