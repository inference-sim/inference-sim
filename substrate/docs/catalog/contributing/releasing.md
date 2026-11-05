# Releasing and the docs site

This site is built from the repository. The prose is in `docs/`; the reference tables, figures and model pages are generated from the data in the same commit. The site for a release therefore describes that release's files.

## How a release is published

1. A maintainer tags a commit on `main` as `MAJOR.MINOR.PATCH`, with no `v` prefix, and publishes a GitHub release with notes.
2. Publishing the release runs the `docs` workflow. It checks out the tag, runs the generator's tests and a strict build, and deploys the site with [mike](https://github.com/jimporter/mike) to the `gh-pages` branch, as the docs version `MAJOR.MINOR`.
3. If the tag is the highest release, the workflow also points the `latest` alias at it; the site's root URL serves `latest`.

Each `MAJOR.MINOR` line shows its newest full release, judged from the repository's GitHub releases. A patch on an older line updates that line and leaves `latest` alone. A tag older than the newest patch on its line is refused, so a rebuild cannot replace newer documentation with older. A pre-release is published when it is promoted to a full release, not before.

Every merge to `main` deploys the docs version `dev`.

To republish a version, for example after a failed deploy, run the `docs` workflow from the Actions tab and give it the release tag. The rebuild uses the generator and the pinned dependencies in the tag itself, so a fix on `main` reaches a published version only through a new patch release. Only tags that contain `mkdocs.yml` can be built.

## How the reference pages are generated

A [MkDocs hook](https://www.mkdocs.org/user-guide/configuration/#hooks), `scripts/docs/catalog_pages.py`, runs during every build. It reads the catalog and:

- writes one page per model under `reference/models/`, each linked from the [Models](../reference/models/index.md) table;
- replaces each placeholder of the form `[[!catalog.NAME]]` in the hand-written pages with a table, a figure or a value computed from the data, such as the number of models or the pinned validator version;
- names the release the site describes. The name comes from the `SUBSTRATE_RELEASE` environment variable, which the workflow sets. A local build without it uses the tag of a clean checkout, and describes `main` otherwise.

A placeholder with an unknown name fails the build, so a misspelled one cannot reach the published site. Because rows and model pages are generated, adding an entry needs no change to the documentation.

The hand-written pages, including [File formats](../reference/formats.md), are not generated. A change to a format in blis-schemas needs a matching edit there.

## Preview locally

```sh
pip install -r requirements-docs.txt
mkdocs serve
```

The site is served at <http://127.0.0.1:8000> and rebuilt whenever a file under `docs/`, a catalog directory or the generator changes. To see the pages labeled as a release, set the variable: `SUBSTRATE_RELEASE=v9.9.9 mkdocs serve`.
