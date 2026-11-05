# Releases and these docs

## Cutting a release

1. Make sure `main` is green: `schema-validate`, `derivation-tests` and `docs` all pass.
2. Run the derivation tests locally with the AISimulate data present (step 3 of
   [the usual workflow](index.md#the-usual-workflow)). CI cannot run them.
3. Publish the release from `main`:

    ```sh
    gh release create vX.Y.Z --target main --title vX.Y.Z --generate-notes
    ```

4. Edit the notes to say which families changed, on which parts, and why, with links to
   the pull requests. A reader deciding whether to move to the release needs to know
   which estimates it affects.

## What publishing does

Publishing a release runs the `docs` workflow, which builds this site from the tag and
publishes it as version `X.Y.Z`. If the tag is the highest release, `latest` moves to it;
a patch to an older line is published under its own version and leaves `latest` alone. A
pre-release (marked as such on GitHub, or tagged with a suffix such as `-rc.1`) is
published under its own version and never becomes `latest`.

Every merge to `main` republishes `dev (main)`. Every pull request builds the site with
`--strict` and publishes nothing.

## Republishing a release's pages

To rebuild a release's documentation after fixing a page, run the workflow by hand:

```sh
gh workflow run docs.yml -f tag=vX.Y.Z
```

This builds the pages from `main`, so a fix lands without a new release, and generates the
tables from the data. It first checks that `coefficients/` on `main` is identical to the
tag, and stops if it is not: the pages would describe numbers the release does not
contain. Use the same command for a release that predates the documentation site.

## How the site is built

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) renders `docs/`, and
[mike](https://github.com/jimporter/mike) keeps one copy per version on the `gh-pages`
branch. The tables, counts and quoted entries are produced at build time by
[`docs_hooks/registry.py`](https://github.com/inference-sim/blis-substrate/blob/main/docs/hooks/registry/registry.py),
which reads `coefficients/` and fails the build rather than render anything it cannot
represent exactly. Its tests are in `docs_hooks/test_registry.py`.

To preview locally:

```sh
pip install -r requirements-docs.txt
mkdocs serve
```

## Moving the schema pin

The schema gate is pinned to a blis-schemas release in `.github/workflows/validate.yml`. A
new blis-schemas release changes nothing here until the pin is moved, which is a one-line
pull request whose CI run shows whether every committed set still loads. Update the tag
named on [File format](../reference/format.md) in the same pull request.
