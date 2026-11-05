# Changes across the repositories

<p class="lede">The process for a change that spans blis-schemas, this catalog, blis-registry,
blis-latency-kernel and inference-sim is described once, in blis-schemas:
[Changes across the repositories](https://github.com/inference-sim/blis-substrate/blob/main/docs/contracts/contributing/cross-repo-changes.md). This page lists only what is specific to the catalog.</p>

The catalog depends on blis-schemas, and the registry, the kernel's tests and inference-sim read
it. It is pushed after blis-schemas and before inference-sim.

## When blis-schemas changes

Move the `validate-catalog@<version>` pin in `.github/workflows/ci.yml` and in the README to the
pushed schemas head's pseudo-version, marked `TEMPORARY` with its pull request, or to the release
tag before merge. Run the [checks](checks.md).

## When the catalog changes

The consumers pin the catalog by commit, so a catalog change reaches them only when they repin:

- blis-registry: `CATALOG_SHA` in `rederive.yml`, and its validator's vendored catalog files;
- blis-latency-kernel: the catalog line of `testdata/upstream.lock`;
- inference-sim: `testdata/catalog` re-vendored with its `.upstream-commit` stamp, and the
  catalog line of `testdata/upstream-prerelease.lock`.

Name each of those in the pull request's dependency table. A changed figure that alters a
published estimate (a memory size, a bandwidth, a derived graph) needs the held-out evidence
the canonical page requires, measured on a build that vendors this catalog's pushed head.

A private-names check (`.github/workflows/private-names.yml`) reads its pattern from a repository secret and fails a pull request whose files, paths, commit messages, title or body match it.
