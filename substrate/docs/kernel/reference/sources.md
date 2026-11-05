# Sources of truth

<p class="lede">Every claim the kernel makes about the engine is stated against one release
of vLLM. The contracts, the catalog and the registry it reads are in the same repository, at
the same commit. This table is generated from the pins themselves.</p>

--8<-- "docs/kernel/generated/sources.md"

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
- **The contracts, the catalog and the registry.** These are `contracts/`, `catalog/` and
  `registry/` of this repository, so there is no pin to move: a change to any of them is
  tested against the kernel in the same pull request. Run `go test ./kernel/...`. A renamed
  chip or coefficient is meant to break the tests that name it.

After a change to a pin or to pricing, regenerate the generated files with blis-vllm's
`go run ./kerneltests/internal/docgen`, so this table and the figures follow.
