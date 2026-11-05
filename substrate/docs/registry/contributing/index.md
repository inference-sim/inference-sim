# Contributing a change

Most changes to the registry change numbers that every estimate using the affected sets
will read. The process below exists so that each such change is visible, reproducible
and checked at the level the numbers are used at.

## Rules that apply to every change

**Never edit a generated value by hand.** Most sets are written by a script, and the
header of each file says which. Change the script or its input, re-run it, and commit
what it prints. A hand edit is indistinguishable from a typo, and the derivation tests
will reject it when they run against the data.

**A value and its citation move together.** An entry's `sources` must name the exact
collection the value was fitted on. Changing one without the other is how a registry
comes to describe a fit nobody ran.

**One collection per part, per family.** Fit on a single engine release. Mixing
collections changes the row set silently and makes a difference between two parts
reflect software rather than silicon.

**Fitting data never includes evaluation data.** A fitter must not read InferenceX or
SemiAnalysis results; `scripts/select_overlap_band.py` holds the guard to import.

**No per-primitive fit ships without an end-to-end check.** A better fit to one
primitive has, six times in this project, made the overall estimate worse. Score the
change with `scripts/compare_registries.py` (see
[methodology §6.1](../methodology.md#61-reproducing-the-end-to-end-evaluation-tables))
and report the before and after figures in the pull request.

**A coefficient with no reader is not live.** Before adding a family, confirm that
blis-latency-kernel looks the key up. Search the kernel for the name, not the registry.

## The usual workflow

1. **Set up the data.** Clone AISimulate and a blis-catalog checkout, and export their
   paths:

    ```sh
    git clone https://github.com/ai-dynamo/aisimulate
    export AISIMULATE_DATA=$PWD/aisimulate/python/aisimulate/src/aisimulate_core/systems/data
    export BLIS_CATALOG=/path/to/blis-catalog
    pip install -r requirements-fit.txt
    ```

    `requirements.txt` covers the property tests. `requirements-fit.txt` adds what the
    fitters need to read the Parquet sweeps: `pyarrow`, `pandas` and `numpy`.

2. **Run the family's writer.** The
   [writer table](../reproducing-coefficients.md#every-committed-value-and-the-one-command-that-regenerates-it)
   names the script that owns each family. Each takes `--check`, which exits non-zero if
   the committed file would change, and most take `--insert CHIP` to add a part.

3. **Run the derivation tests with the data present.** CI does not run them on pull
   requests (the `rederive` workflow takes ~20 minutes and is manual-dispatch only), so
   run them locally before any pull request that touches a coefficient, a writer or a
   test. With `COLLECTIVEX_DATA` also set to the CollectiveX run bodies:

    ```sh
    scripts/rederive.sh
    ```

    It runs `python -m pytest validator/ -q -rs` and fails on any skip other than the
    by-design ramp-form ones, because a skip with the data present is a test that did
    not run.

4. **Run the schema validator.** CI runs it on every pull request; running it first
   saves a round trip. From a blis-schemas checkout at the tag pinned in
   `.github/workflows/validate.yml`:

    ```sh
    go run ./cmd/validate-registry /path/to/blis-registry
    ```

5. **Score the change end to end**, as above, if it changes a value a step time depends on.

6. **Preview the documentation.** The reference pages are generated from
   `coefficients/`, so they show your change:

    ```sh
    pip install -r requirements-docs.txt
    mkdocs serve
    ```

7. **Open a pull request.** Three checks run: `schema-validate`, `derivation-tests` and
   `docs`. All three must pass.

## Adding a part

A part must exist in blis-catalog before the registry can scope an entry to it, because
`scope.hardware` uses catalog chip names. The order of the writers matters in one place
(prefill attention reads the GEMM ramp from the registry), so follow the
[GB300 worked example](../reproducing-coefficients.md#adding-a-part-the-gb300-worked-example).

The host-overhead and memory sets list every catalog part in their scope, and a property
test checks that they do. Adding a part is therefore also a visible edit to those sets,
which is intended: it is the reminder that their values were never measured on the new
part.

## Adding a set

Add the file under `coefficients/`, then add a reference page for it at
`docs/reference/sets/<name>.md` and list that page in the `nav` of `mkdocs.yml`. The
documentation build fails until both exist, so a set cannot be committed without a page.
Copy an existing page; the table on it is generated, so the page needs only a short
description of what the set prices and where its values come from.

## Writing documentation

The pages under `docs/` are Markdown, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). A number that
describes the registry's contents (a count, a value, a census) must not be typed into a
page. Use a `registry:` directive instead, which the build computes from the committed
files; [`docs_hooks/registry.py`](https://github.com/inference-sim/blis-substrate/blob/main/docs/hooks/registry/registry.py)
lists them. A retyped count is correct on the day it is written and wrong after the next
refit.

## Changes across the repositories

A change that also touches blis-schemas, the catalog, the kernel or inference-sim follows
[Changes across the repositories](cross-repo-changes.md), which links the canonical process and
lists the registry's own steps.
