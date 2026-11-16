# Kernel-implementation table

<p class="lede">An engine release's list of every kernel implementation it can emit, for the
primitives it covers: the library, the library's own kernel name, the target GPU architecture and
the code hash the coefficient registry keys its entries by (design §6.2, D5, VAL-8).</p>

Each engine repository generates the table from its upstream's own kernel-selection code and
publishes it as `kernel_impls.json` at its root, so the release tag carries it. Three readers use
it. The engine's lowering states the implementation of each lowered operation from it, so the
cost model's lookup matches a coefficient only for the build that runs. The registry's fitters key
a fitted entry by the implementation its data exercised (`kernel_impls`, and `measured_on` for a
transferred value). The scoring harness maps each library version a run recorded to the hash
BLIS priced (design §11.3, G1).

`kernelimpls.Parse` (and `kernelimpls.Load`, `contracts.LoadKernelImplTable`) reads a table
strictly, as every BLIS document is read, and validates it: a key the schema does not declare, a
key that appears twice, or a second document is an error. `kernelimpls.Marshal` writes the
canonical bytes, and an engine repository checks its committed file by comparing the two.

```json
--8<-- "docs/contracts/examples/kernel-impl-table.json"
```

## Keys

<!-- fields: kernelimpls.Table -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `KernelImplTable`. |
| `engine` | mapping | yes | The engine release whose kernels the table lists, below. |
| `primitives` | list of strings | yes | The lowered primitives whose implementations the table lists, sorted, each once: every kernel the release can emit for one of them is a row, except where `gaps` names a selection not yet listed. A primitive not named here is not covered, and a missing row for it says nothing. |
| `archs` | list of strings | yes | The device architectures the table lists builds for, as the compiler names them (`sm_90a`, `sm_100a`), sorted, each once. A build for another architecture is not covered, and a missing row for it says nothing. |
| `gaps` | list | no | The upstream selections under a covered primitive whose kernels the table does not yet list, below. |
| `libraries` | list | yes | Each kernel library at each version the table covers, below. |
| `implementations` | list | yes | One row per primitive, library, library version, variant and target architecture, below. |

### `gaps`

<!-- fields: kernelimpls.Gap -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `primitive` | string | yes | One of `primitives`. |
| `selection` | string | yes | The upstream choice, in the upstream's own words: a backend, a kernel path or a setting's value. |
| `issue` | string | yes | The issue that lists its kernels, as `<owner>/<repo>#<n>`. |

Sorted by primitive and then selection, each once. A gap is how a table says what it does not
know, as a field list's `not_modelled` does: a reader that finds no row for a build of a gap's
selection learns nothing from the missing row, and an engine's lowering reports such a build with
no code hash, which matches no keyed entry.

### `engine`

<!-- fields: kernelimpls.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The engine kind, for example `vllm`. |
| `version` | string | yes | The upstream release, for example `0.32.0`. |
| `commit` | string | yes | The upstream release's commit, 40 lower-case hexadecimal digits. |

### `libraries`

<!-- fields: kernelimpls.Library -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The library's name, as the registry's `kernel_impls` name it. |
| `version` | string | yes | The library's own version, or the commit the engine builds it from. |
| `pinned` | boolean | yes | `true` for the version the release pins. Each library has exactly one pinned version. |
| `pin` | string | yes | Where the version is stated: the release's lock or build file, as `path:line@commit8`, for a pinned version; the runs it was seen in otherwise. |
| `origin` | string | yes | The immutable identity of what was hashed: a repository URL and commit, or a distribution file and its SHA-256. |

The list is sorted by name and then version, and a version appears once. Beside the pinned
version, a table lists any version that runs of the release were seen with (design §6.2), so a
scoring harness can map the version a run recorded to a hash.

### `implementations`

<!-- fields: kernelimpls.Implementation -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `primitive` | string | yes | The lowered primitive the kernel executes, one of `primitives`. |
| `library` | string | yes | With `library_version`, a row of `libraries`. |
| `library_version` | string | yes | The library's version. |
| `variant` | string | yes | The library's own exported kernel name. |
| `target_arch` | string | yes | The architecture of the device the build runs on, one of `archs`. A lowering finds it from the chip's compute capability. |
| `hash_basis` | string | yes | What the hash covers, below. |
| `hash_inputs` | list of strings | yes | The content identities the hash covers, sorted, each once, none with a line break: for a source build, each source path with its git object id; for a binary, each shipped file or manifest with its SHA-256. |
| `code_hash` | string | yes | `kernelimpls.CodeHash` of the row, 64 lower-case hexadecimal digits. The validator recomputes it. |
| `toolkit` | string | yes | The CUDA toolkit version of the release's build. |
| `selected_by` | list of strings | yes | The upstream code that can select the kernel, as `path:line@commit8`, sorted, each once. |

Hash bases:

--8<-- "docs/contracts/generated/enum/kernelimpls.HashBasis.md"

`source`
:   The kernel is built from source; the hash covers its source files and the target architecture.
`binary`
:   The kernel ships only as a binary; the hash covers the shipped file's SHA-256 and the target architecture.

Rows are sorted by primitive, library, library version, variant and target architecture, and
each appears once. One build of a kernel has one hash whichever primitive it serves, so two rows
that differ only in their primitive must agree on it.

## The code hash

`kernelimpls.CodeHash` is the one rule every engine uses. It is the SHA-256, in lower-case
hexadecimal, of these lines, each ending in a newline:

```text
blis kernel hash v1
basis <hash_basis>
library <library>
variant <variant>
arch <target_arch>
input <hash_inputs[0]>
input <hash_inputs[1]>
...
```

The inputs are written in byte order. The library's version is not hashed: a release that bumps a
library and leaves a kernel's inputs alone leaves the kernel's hash alone, so an entry keyed by it
stays valid with no registry change (design §6.2). A kernel whose inputs changed has a new hash,
and an entry keyed by the old one no longer matches it (VAL-8): the registry supplies a value for
the new key, or the lookup takes the most trusted entry it does offer and the price records that
method (VAL-3).

## Which entries the tables account for

Design VAL-8 asks of every registry entry that a lookup on the conformance scenarios selects
it, or that its kernel hash appears in the table of an engine release some supported simulator
release pins. `go run ./validate/cmd/unselected-entries` lists the entries that meet neither:

```sh
go run ./validate/cmd/unselected-entries \
  -selected run-1/result.json -selected run-2/result.json \
  -kernel-impls kernel_impls.json \
  registry
```

Each `-selected` file is the Result of a conformance scenario's run; its `selected_entries`
names an entry by set, name, scope, method and kernel implementations. Each `-kernel-impls`
file is a release's table; an entry's hash appears in it when the table lists one of the entry's
`kernel_impls` by library, variant and code hash together. An entry that states no
`kernel_impls` has no hash to appear, so only a selection accounts for it. The command exits 1
when it lists an entry; the next substrate release removes what it lists.
