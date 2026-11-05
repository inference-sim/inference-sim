# Documentation and releases

<p class="lede">This site is built from blis-substrate with Material for MkDocs and
published to GitHub Pages, one copy per release line. It has a section for each part of the
repository. Tables, lists and examples that the code or the data already state are
generated from them or tested against them, so they cannot drift. The prose is kept right by
review.</p>

## Preview locally

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r docs/requirements.txt
DOCS_RELEASE=main .venv/bin/mkdocs serve
```

The site is then at <http://127.0.0.1:8000>, rebuilt whenever a page, the data or a hook
changes. CI builds it with `mkdocs build --strict`, which fails on a page missing from the
navigation, a link to a missing page or heading, or a missing included file.

## Where things live

| Path | Contents |
|---|---|
| `mkdocs.yml` | Site configuration and the navigation. A new page must be added here. |
| `docs/contracts/`, `docs/catalog/`, `docs/registry/`, `docs/kernel/` | One section per directory of the repository. |
| `docs/contributing/` | Pages that apply to the whole repository, such as this one. |
| `docs/contracts/examples/` | Example documents. Pages include them; tests validate them. |
| `docs/contracts/figures/`, `docs/kernel/figures/` | Drawn figures, as inline SVG that takes its colors from the page. |
| `docs/contracts/generated/` | The vocabulary lists, written by `go run ./tools/docgen`. Never edit by hand. |
| `docs/kernel/generated/` | The cost model's step figure, its known divergences and its sources, written by blis-vllm's `kerneltests/internal/docgen`. Never edit by hand. |
| `docs/hooks/` | The two build hooks that render the catalog's and the registry's reference pages from the data, and their tests. |
| `docs/stylesheets/extra.css` | Typography and figure styles. Rules for one section are scoped under `.section-<name>`. |
| `docs/requirements.txt` | The pinned Python packages the site is built with. |

A page includes a file with the snippets syntax, `--8<-- "path"`, or a marked section of
one with `--8<-- "path:name"`. The path is relative to the repository root. That is how a
page shows a real catalog file, a tested example, or a Go example whose output is checked.

The pages use American spelling. Each opens with a paragraph that stands alone, defines a
term before it uses it, and points to the source for detail rather than reproducing it.

## What keeps the pages correct

### Contracts

All run by `go test ./contracts/ ./tools/...`:

**Generated lists.** `go run ./tools/docgen` writes the members of every closed vocabulary
into `docs/contracts/generated/`. `TestGeneratedDocsAreCurrent` fails if a committed file
differs from what the generator would write, or if the generator no longer writes it. After
changing a vocabulary, run the generator and commit its output.

**Key tables.** Each key table on a reference page follows a marker naming its Go type:

```markdown
<!-- fields: deployment.Pool -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | ... |
```

`TestDocFieldTablesMatchTypes` compares the keys in the first column with the YAML keys the
type declares. It does not check the other columns. A key missing from the table, or a key
the type does not declare, fails the test. One type may be split over several tables, whose
keys are combined. `TestDocEveryDocumentTypeHasATable` requires a table for every type
reachable from a document, so a nested type added later cannot go undocumented.

**Example documents.** `TestDocExamplesValidate` validates every file in
`docs/contracts/examples/` and requires no problems at all, not even warnings. It also
requires the catalog names in each example scenario to resolve against the repository's
`catalog/`.

**Go examples.** The Go snippets on the contracts pages are sections of
`contracts/example_test.go`. Go runs them as example tests and compares their output with
the `// Output:` comment, so a snippet cannot show a call that no longer compiles or a
message the code no longer prints.

### Catalog

The hook `docs/hooks/catalog_pages.py` runs during every build and reads `catalog/`. It
writes one page per model under `catalog/reference/models/`, and replaces each placeholder of
the form `[[!catalog.NAME]]` on a catalog page with a table, a figure or a value computed
from the data. A placeholder with an unknown name fails the build, so a misspelled one
cannot reach the published site. Adding an entry needs no change to the documentation.

### Registry

The hook `docs/hooks/registry.py` reads `registry/` and replaces each directive of the form
`<!-- registry:NAME -->` on a registry page with a table, a chart, a quoted entry or a
count. Its module docstring lists the directives. An unknown directive, a set with no
reference page, or data a table cannot represent exactly fails the build.

Both hooks render only their own section's pages. Their tests run with:

```sh
DOCS_RELEASE=main python3 -m pytest docs/hooks -q
```

### Cost model

`go test ./kernel/` runs three checks. `TestDocAssumptionsMatchTheCode` compares the table
of assumptions on [Provenance, Evidence, Resolved](../kernel/reference/provenance.md) with
the names the code can record, in both directions. `TestDocOptionalCoefficientsMatchTheCode`
does the same for the optional coefficients. `TestDocSourceLinksExist` checks that every
link from the cost model's pages into this repository's source names a path that exists.

The Go the cost model's pages describe is sections of blis-vllm's
`kerneltests/example_test.go`, because it opens a kernel from a vLLM engine block, which
blis-vllm decodes. blis-vllm runs those examples, and its `TestDocOutputsMatchTheirExamples`
checks that the output a page shows is the example's output. The files under
`docs/kernel/generated/` are written by blis-vllm's `kerneltests/internal/docgen`, whose
`TestGeneratedDocsAreCurrent` checks blis-vllm's own copy. No test in this repository checks
that the copy here is current.

### Prose

No test can check that the prose is right. When a change alters behaviour, search these
pages for the behaviour and update them in the same pull request.

## The release a page describes

Each hook names the release the pages describe. `DOCS_RELEASE` gives the registry hook its
release tag, and `SUBSTRATE_RELEASE` gives the catalog hook its own; the docs workflow sets
both from the release tag. A local build sets `DOCS_RELEASE=main`. Without
`SUBSTRATE_RELEASE`, the catalog hook uses the tag of a clean checkout at a tagged commit,
and describes `main` otherwise.

## Releases

blis-substrate is released as one Go module, tagged `vX.Y.Z` (design §14.1). One tag covers
the contracts, the catalog, the registry and the cost model, and a release embeds the
catalog and registry it was tested with.

| Part of the version | Raised when |
|---|---|
| PATCH | no contract, price or value changed |
| MINOR | a compatible contract addition, new data or pricing forms, or changed values, listed in the notes |
| MINOR, while the module is at v0; MAJOR from v1 | an incompatible contract change, made as one review set with every implementer |

The release bot cuts releases, from a commit where every check passed, and writes the
release notes from the release's changes (LIF-3, LIF-4). If the bot is unavailable, two
release owners follow the documented manual procedure, and each manual tag is signed and
listed in its release notes.

## Publishing

`.github/workflows/docs.yml` builds and publishes the site with
[mike](https://github.com/jimporter/mike), which keeps one directory per version on the
`gh-pages` branch.

| Event | What is published |
|---|---|
| A pull request that changes the docs, the data or the hooks | Nothing. The `docs` job of `ci.yml` runs the hook tests and builds the site with `--strict`. |
| A push to `main` | Version `dev`, rebuilt from `main`. Until a release is published, the site root serves `dev`. |
| A release `vX.Y.Z` is published on GitHub | Version `X.Y`, rebuilt from the tag. If `X.Y` is the newest stable release line, the alias `latest` moves to it and the site root serves it. |
| A pre-release, or a tag not of the form `vX.Y.Z` | Nothing. |
| A manual run from the Actions tab | The release line given as `MAJOR.MINOR`, built from the chosen branch or tag. It becomes `latest` only if asked. |

Docs are versioned by release line, not by patch: `v0.11.0` and `v0.11.1` both publish to
`0.11`, each replacing the last. A patch to an older line updates that line and leaves
`latest` alone. Every version other than the newest shows a banner that links to it.

A release created by a workflow that uses the default `GITHUB_TOKEN` starts no other
workflow, so it would not publish the docs. Release automation must publish with another
token, or start the docs workflow itself.

To repair a version whose deploy failed, run the **docs** workflow by hand from the Actions
tab: choose the tag to build from, give the release line as `MAJOR.MINOR`, and choose
whether it becomes `latest`. The rebuild uses the pages, hooks and data in that tag.

GitHub Pages serves the `gh-pages` branch at
<https://inference-sim.github.io/blis-substrate/>.
