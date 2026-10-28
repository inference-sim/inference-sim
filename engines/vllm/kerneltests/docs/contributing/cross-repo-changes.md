# Changes across the repositories

<p class="lede">The process for a change that spans blis-schemas, blis-catalog, blis-registry,
this kernel and inference-sim is described once, in blis-schemas:
[Changes across the repositories](https://github.com/inference-sim/blis-schemas/blob/main/docs/contributing/cross-repo-changes.md). This page lists only what is specific to the kernel.</p>

The kernel depends on blis-schemas and is consumed by inference-sim, so it is pushed after
blis-schemas and before inference-sim.

## When blis-schemas changes

1. Move the `go.mod` requirement to the pushed schemas head's pseudo-version
   (`GOPROXY=direct go list -m github.com/inference-sim/blis-schemas@<sha>`), or to its release
   tag before merge. Never commit a `replace`.
2. Run `go mod tidy`, then blis-vllm's `go run ./kerneltests/internal/docgen` against this checkout: `docs/generated/sources.md` names the
   pin, and `TestGeneratedDocsAreCurrent` fails until it is regenerated.
3. Run `go test ./...`.

## When the tests need a newer catalog or registry

Edit the line in `testdata/upstream.lock` (repository, tag, full commit), run
`scripts/fetch-testdata.sh`, and run `go test ./...`. A renamed chip or coefficient is meant to
break the fixtures that name it. While the catalog or registry is unreleased, inference-sim
overlays its own `testdata/upstream-prerelease.lock` instead; the kernel's lock moves to the
release tags before the kernel merges.

## Evidence a kernel pull request carries

- the vLLM lines each pricing statement rests on, at the modelled release;
- the `cmd/shape` / `cmd/score` figures before and after a pricing change;
- for a change that moves a published estimate, the held-out and gate evidence the canonical
  page requires, from a build of inference-sim against this kernel's pushed head;
- for a coefficient the change reads or adds, the registry's
  [fitting principles](https://github.com/inference-sim/blis-registry/blob/main/docs/fitting-principles.md).

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
