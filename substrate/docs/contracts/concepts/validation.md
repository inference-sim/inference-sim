# Validation

<p class="lede">Validation has two layers, and every report keeps them apart. The
<strong>field layer</strong> asks whether each document is well formed; its answer never
depends on an engine release. The <strong>rule layer</strong> asks whether the release the
scenario names would run the documents as written; its answer always does.</p>

--8<-- "docs/contracts/figures/validation.html"

## Field problems

A field check asks whether a document is what it claims to be. Are the counts positive?
Does the arithmetic close? Is an enumerated value one the schema knows? Is the model's
graph acyclic? Does the deployment's parallel layout fit on the GPUs its pool owns?

These questions have the same answer under every engine release, so they live with the
types in `spec/` and change only when a schema changes. A field problem always means the
document is wrong, and the author fixes it.

Most field checks read one document; a few compare two. A deployment is checked against
the cluster in its scenario: the pools must account for every node, each engine must fit
in the GPUs its pool owns, its local replicas must fit on a node, and, if the cluster lists
storage classes, each offload tier must use one of them. A bundle (the set of documents handed to `Validate`) that holds a deployment but no
scenario is itself a field problem, since checking the deployment alone would skip half of
its checks and report success.

## Rule problems

A rule asks whether a document describes something a *particular* engine release would
run, and run as the document says. Backend names come and go between releases, defaults
change, and a feature gains a new condition that disables it. Each of these is a fact
about one release.

Rules are therefore grouped into a **rules pack** for each release. The engine model that
owns the release owns its pack; for vLLM that is blis-vllm. The caller passes the rules in
(`Bundle.Rules`), and a scenario's `engine_version` selects a release by exact match.
[Engine rules](../reference/rules.md) describes the interface.

A rule problem does not always mean the document is wrong. It may mean the document is
right and names the wrong engine version, or that a rule needs extending for a release it
has not seen. That is why the layers are reported separately:

```go
--8<-- "contracts/example_test.go:rules"
```

```text title="Output"
field layer ok: true
error [enum-values-known]: : pools[0].engine.cache_dtype: "fp8_e4m3fn" is not accepted by engine 0.29.0
```

A rule problem carries its rule's name in brackets. Rules state the location inside the
message, so the path field between the two colons is empty.

The rule layer runs only if the field layer reports no error. A rule reads a document on
the assumption that it is well formed, and over a malformed one it reports consequences of
the malformation, which a reader cannot tell apart from real problems.

!!! note "An unknown engine version passes, with a warning"
    If the rules given know no release of the scenario's `engine_version`, the rule layer reports one
    warning, `engine-version-known`, and `RulesApplied` is empty. Warnings do not fail a
    report, so `OK()` is still true. A caller that requires rules to have run should also
    check that `RulesApplied` is not empty.

Rules that answer for a release other than the scenario's are an error,
`engine-version-matches`, and nothing is applied. A bundle with a scenario and no rules is an
error, `engine-rules-given`. [The Go guide](../guides/go.md#pass-the-engines-own-rules) shows
how a caller passes them.

## Reports

`Validate` returns a `Report`:

`Field`
:   The field layer's problems.

`Rule`
:   The rule layer's problems, each carrying the name of the rule that reported it.

`RulesApplied`
:   The engine version whose rules ran. Empty if the field layer failed, if the bundle has
    no scenario, if no rules matched, or if the caller's rules answered for another release.

`OK()` is true when neither layer holds an error. `Problems()` returns everything, field
problems first. Each problem has a severity, a dotted path to the field and a message:

```text
error: deployment.pools[0].parallel.dcp: decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp 8 must be divisible by dcp 3
```

An **error** means the document cannot be used as written. A **warning** means it can, but
someone should look; a model graph that declares a layer it never uses is a typical case.
Only errors make `OK()` false.

Validators do not stop at the first problem. They report every problem in one pass,
sorted by path, so a document with five mistakes takes one round of fixes rather than five.
Loaders work differently. A file that fails to decode yields an error and no document, so
nothing in it is validated until it decodes.

## Do the names resolve?

`Validate` reads nothing from disk. It checks the documents it is given. A scenario that
says `hardware: nvidia-h100` passes, because the field is a well-formed string, and fails
later when something tries to open the file.

`ValidateAgainstCatalog` answers that question. Given a catalog root and a registry root,
it checks that each name the scenario uses refers to an entry that exists:

| Scenario key | Must exist |
|---|---|
| `model` | the directory `models/<name>/` in the catalog |
| `cluster.hardware` | the file `hardware/<name>.yaml` in the catalog |
| `cluster.fabric` | the file `networks/<name>.yaml` in the catalog, if a fabric is named |
| `cluster.storage` | each name as a key in the catalog's `devices/storage.yaml` |
| `workload.shape` | the file `workloads/<name>.yaml` in the catalog, if `workload` names a shape |
| `coefficients` | each name as the file `coefficients/<name>.yaml` in the registry |

It checks presence, not validity; validating what the names point to is `Validate`'s job.
A failure names the path it looked for and lists what the catalog has, so a wrong guess is
corrected in one step:

```text
error: references.hardware: "nvidia-h100": looked for hardware/nvidia-h100.yaml; the catalog has [a100-80 a100-sxm b200 b300 gb200-nvl72 gb300 h100 h200 l40s]
```

The two entry points are separate so that `Validate` does not depend on where, or whether,
a catalog is checked out.

## Who runs what

| Caller | Runs |
|---|---|
| `validate-catalog`, `validate-registry`, `validate-measurements` | The field layer, for each committed document, and for measured runs the licence check and the verdict. These documents name no engine version, so no rules run. |
| The cost model (`kernel/`), when it constructs a kernel | `Validate` over its inputs. A field-layer error stops construction; rule-layer problems do not. |
| Your own Go program | Whatever it needs; see [Validate documents from Go](../guides/go.md). |
