# Checks and CI

Three checks decide whether a change can merge, and two more build this site. Run them from the repository root before pushing:

```sh
# Every entry is well formed.
go run ./validate/cmd/validate-catalog catalog

# Every graph is what the deriver produces from its config, and the deriver's tests pass.
pip install --require-hashes -r catalog/scripts/requirements-dev.txt
(cd catalog && python3 scripts/derive_graph.py --check && python3 -m pytest tests -q)

# The site's page generators pass their tests, and the site builds without warnings.
pip install --require-hashes -r docs/requirements.txt
DOCS_RELEASE=main python3 -m pytest docs/hooks -q
DOCS_RELEASE=main mkdocs build --strict
```

The first needs Go; the others need Python 3. CI runs the first three in the `catalog` job and the last two in the `docs` job of `.github/workflows/ci.yml`, with the Go version `go.mod` names and Python 3.12. Both jobs run when a file under `catalog/` changes.

## What each check catches

`validate-catalog`
:   Loads every file in `models/`, `hardware/`, `networks/`, `workloads/` and `devices/` through the document types in `contracts/`, which define the format of each. It reports an unknown or misspelled key, a missing required field, a non-finite or out-of-range number, a `model.yaml` whose `name` differs from its directory, and a model graph with a cycle. It prints one line per valid entry and each problem on standard error. It exits `0` when everything validates, `1` when anything fails, and `2` when it is given no catalog to check. The validator and the catalog are in one repository, so a change to a document type, the matching edit to [File formats](../reference/formats.md), and the migration of any catalog file that no longer validates land in one pull request.

`derive_graph.py --check`
:   Derives every graph again and compares it with the committed `graph.yaml`, writing nothing. It fails when a graph was edited by hand, or when a `config.json` changed and its graph was not derived again.

`catalog/tests/test_derive_graph.py`
:   Tests the deriver itself. It checks derived shapes against the upstream model implementations (attention and expert geometry, where each collective sits, how draft modules are built, how the stack is compressed) and checks that a config the deriver cannot translate is refused. `--check` shows that the committed graphs can be reproduced; these tests check that the derivation is right, for the cases they cover.

`docs/hooks/test_catalog_pages.py` and `mkdocs build --strict`
:   Test the generator of the reference pages, `docs/hooks/catalog_pages.py`, against the data, then build the site, failing on any warning, broken link or unknown placeholder. A data change can break the site, for example a model whose page links to a missing file; these checks catch it before merge.

## What no check catches

The checks decide whether an entry is well formed and consistent. They cannot decide whether a figure is true: a plausible but wrong peak rate passes all of them. That is why every figure cites its source, and why review checks each figure against its source.
