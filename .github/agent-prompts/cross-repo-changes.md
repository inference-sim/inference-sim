# Changes that touch the llm-d router fork

The substrate (`substrate/`), the engines (`engines/vllm`, `engines/sglang`) and the simulator
(`sim/`, `cmd/`) are directories of one Go module, so a change to any of them is one pull request
and needs no repin. Only the llm-d router fork, `inference-sim/llm-d-router`, is a separate
repository, so only a change that needs a fork change has a release order: the fork first, then
this repository. The per-PR workflow ([PR Workflow](pr-workflow.md)) and the standards
([rules](standards/rules.md), [invariants](standards/invariants.md)) apply unchanged.

## Repinning the router fork

1. `go.mod`: the fork's `replace` at its `vX.Y.Z-blis.N` tag, or, while the fork pull request is
   open, at its exact pushed head marked `TEMPORARY` with the pull request
   ([The llm-d Router Fork](llm-d-router-fork.md)). It is the only `replace` (design X11). Run
   `go mod tidy`.
2. `stack.json` names the pinned engine releases and the router; a change to a pin updates it in the
   same pull request (`blis version` prints the pins).
3. The catalog and registry need no step: they are `substrate/catalog` and `substrate/registry`,
   embedded in the binary (`internal/substratedata` for tests).
4. `go test ./...` and `golangci-lint run ./...`.

## The integrated build

Gates and published numbers come from one binary built from the integrated head with its
revision injected:

```bash
go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRevision=$(git rev-parse HEAD)" -o blis main.go
```

It reads the catalog and registry embedded in the binary, never a working tree. The pull request
body records the binary's revision; the held-out and gate evidence the canonical page requires
goes under *Held-out and gate evidence* in the template.

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
