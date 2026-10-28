# Sources of truth

<p class="lede">Every claim the kernel makes about the engine is stated against one release
of vLLM, and every input it reads is pinned to one commit. This table is generated from the
pins themselves.</p>

--8<-- "docs/generated/sources.md"

## Two sources of engine behavior

Most of what the kernel knows about vLLM is encoded here, at v0.31.0: which kernels run, how
work is split, what an unstated setting defaults to, which layouts start. Some version-scoped
facts are read instead from the engine rules the caller supplies for the release the scenario
declares (for vLLM, the packs of `github.com/inference-sim/blis-vllm/rules`),
among them which tensor-parallel widths the custom all-reduce supports, when the MoE input is
made sequence-parallel, which decode-context combine backends the release accepts, how many
pages a window cache holds, how stated caches group into the shared KV block pool, and the
CUDA graph capture sizes a deployment's stated knobs resolve to (`resolve.EngineRules` lists
them).

## Reading a citation

A comment in the source that cites `vllm/<path>:<lines>` means those lines at the commit
above. To read them in a vLLM checkout:

```sh
git show v0.31.0:vllm/config/parallel.py | sed -n '571,577p'
```

A comment that cites a vLLM line without a path, as `:571-577`, continues the path of the
citation before it.

## Changing a pin

- **vLLM.** Moving to a newer release is a review of every behavior the kernel encodes, not a
  version bump: each citation must be checked against the new release, and the constants
  `EngineBehaviourVersion` and `EngineBehaviourCommit` in `new.go` changed with them.
- **blis-schemas.** A Go module dependency: change `go.mod`.
- **blis-catalog and blis-registry.** Edit `testdata/upstream.lock`, run
  `scripts/fetch-testdata.sh`, and run `go test ./...`. A renamed chip or coefficient is
  supposed to break the scenarios that name it.

After any of these, run `go run ./internal/docgen` so this table and the figures follow.
