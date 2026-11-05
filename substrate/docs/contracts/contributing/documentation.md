# Documentation and releases

<p class="lede">These pages are built with Material for MkDocs and published to GitHub
Pages, one copy per release line. The vocabulary and rules listings, the set of keys in each
table, the examples and the Go snippets are generated from the code or tested against it,
so they cannot drift from it. The prose, and the type, requirement and meaning given for
each key, are kept right by review.</p>

## Preview locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r docs/requirements.txt
.venv/bin/mkdocs serve
```

The site is then at <http://127.0.0.1:8000>. CI builds with `mkdocs build --strict`, which
fails on a page missing from the navigation, a link to a missing page or heading, or a
missing included file.

## Where things live

| Path | Contents |
|---|---|
| `mkdocs.yml` | Site configuration and the navigation. A new page must be added here. |
| `docs/` | The pages, one directory per section. |
| `docs/examples/` | Example documents. Pages include them; tests validate them. |
| `docs/figures/` | The figures, as inline SVG that takes its colors from the page. |
| `docs/generated/` | Generated lists. Never edit these by hand. |
| `docs/stylesheets/extra.css` | Typography and figure styling. |
| `docs/requirements.txt` | The pinned Python packages the site is built with. |
| `internal/docgen/` | The generator for `docs/generated/`, and the test that it is current. |
| `docs_test.go`, `example_test.go` | The tests that tie the pages to the code. |

A page includes a file with the snippets syntax, `--8<-- "path"`, where the path is relative
to the repository root. That is how a page shows a real catalog file, a tested example, or a
Go example whose output is checked.

The pages use American spelling.

## What keeps the pages correct

Four checks, all run by `go test ./...`:

**Generated lists.** `go run ./internal/docgen` writes the members of every closed
vocabulary, and each registered rules pack's rules and accepted values, into
`docs/generated/`. `TestGeneratedDocsAreCurrent` fails if a committed file differs from
what the generator would write, or if the generator no longer writes it. After changing a
vocabulary or a pack, run the generator and commit its output.

**Key tables.** Each key table on a reference page follows a marker naming its Go type:

```markdown
<!-- fields: deployment.Pool -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | ... |
```

`TestDocFieldTablesMatchTypes` compares the keys in the first column with the YAML keys the
type declares. It does not check the other columns. A key missing from the table, or a key the type does not declare, fails the
test. One type may be split over several tables, whose keys are combined.
`TestDocEveryDocumentTypeHasATable` requires a table for every type reachable from a
document, so a nested type added later cannot go undocumented.

**Example documents.** `TestDocExamplesValidate` validates every file in `docs/examples/`
and requires no problems at all, not even warnings. It also requires the catalog names in each example scenario to resolve against the pinned
copy of the catalog in `testdata/`. The scenarios' coefficient-set names are not resolved,
since there is no pinned copy of the registry.

**Go examples.** The Go snippets on these pages are sections of `example_test.go`. Go runs
them as example tests and compares their output with the `// Output:` comment, so a snippet
cannot show a call that no longer compiles or a message the code no longer prints.

No test can check that the prose is right. When a change alters behavior, search these
pages for the behavior and update them in the same pull request.

## Publishing

`.github/workflows/docs.yml` builds and publishes the site with
[mike](https://github.com/jimporter/mike), which keeps one directory per version on the
`gh-pages` branch.

| Event | What is published |
|---|---|
| A pull request touching the docs | Nothing. The site is built with `--strict`, and the job fails on any warning. |
| A push to `main` | Version `dev`, rebuilt from `main`. Until a release is published, the site root serves `dev`. |
| A release `vX.Y.Z` is published on GitHub | Version `X.Y`, rebuilt from the tag. If `X.Y` is the newest stable release line, the alias `latest` moves to it and the site root serves it. |
| A pre-release, or a tag not of the form `vX.Y.Z` | Nothing. |
| A manual run from the Actions tab | The version given, built from the chosen branch or tag. It becomes `latest` only if asked. |

Docs are versioned by release line, not by patch: `v0.2.0`, `v0.2.1` and `v0.2.2` all
publish to `0.2`, each replacing the last. The version selector in the header lists the
lines.

### Making a release

1. Merge the changes to `main`, including any regenerated files in `docs/generated/`.
2. Publish a release on GitHub with a new tag, `vX.Y.Z`.

The workflow then publishes `X.Y`.

A release created by a workflow using the default `GITHUB_TOKEN` starts no other workflow,
so it would not publish the docs. Release automation, if it is ever added, must publish with
another token or start this workflow itself.

### Backfilling or repairing a version

Run the **docs** workflow by hand from the Actions tab. Choose the branch or tag to build
from, give the release line as `MAJOR.MINOR`, and choose whether it becomes `latest`. The
same run repairs a version whose build failed. A tag older than the documentation has no
`mkdocs.yml` and cannot be built; build such a line from a later commit whose code it
matches.

GitHub Pages serves the `gh-pages` branch at
<https://inference-sim.github.io/blis-schemas/>.
