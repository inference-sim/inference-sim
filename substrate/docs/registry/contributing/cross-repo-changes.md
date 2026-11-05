# Changes across the repositories

<p class="lede">The process for a change that spans blis-schemas, blis-catalog, this registry,
blis-latency-kernel and inference-sim is described once, in blis-schemas:
[Changes across the repositories](https://github.com/inference-sim/blis-substrate/blob/main/docs/contracts/contributing/cross-repo-changes.md). This page lists only what is specific to the registry.</p>

The registry depends on blis-schemas and on the catalog's part names; the kernel's tests and
inference-sim read it. It is pushed after blis-schemas and before inference-sim.

## The standard for a coefficient or pricing change

Every change to a coefficient, or to how a value is fitted, follows
[Fitting BLIS to the real world](https://github.com/inference-sim/blis-substrate/blob/main/docs/registry/fitting-principles.md): a fit / held-out split declared before the held-out
numbers are read, distributions rather than point errors, agreement across latency, KV, queueing
and throughput, the version rule, and the landing gate. [Contributing a change](index.md) gives
the workflow.

## When blis-schemas changes

Move the blis-schemas `ref` in `.github/workflows/validate.yml` to the pushed head's full SHA,
marked `TEMPORARY` with its pull request, or to the release tag before merge; see
[Moving the schema pin](releasing.md#moving-the-schema-pin).

## When the catalog changes

If the change needs a newer catalog, move `CATALOG_SHA` in `.github/workflows/rederive.yml`
and the validator's vendored catalog files together, and run the derivation tests with the
data present.

## When the registry changes

The kernel's tests pin the registry in `testdata/upstream.lock`, and inference-sim vendors it
into `testdata/registry` with an `.upstream-commit` stamp. The vendored copy must equal this
pull request's head, file for file, before any published number is computed from it.

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
