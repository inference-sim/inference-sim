# Experiment with a candidate catalog

To ask what BLIS predicts for a model or a chip that is not in a release, edit a copy of the
catalog and name the copy when you run. You need no fork of the simulator, no rebuild and no
pull request.

## Edit a copy

Start from a working tree of blis-substrate, or a copy of its `catalog/` directory, and edit
it. If you edit a model's `config.json`, derive its graph again, so that `graph.yaml` matches
the edited config; the cost model reads `graph.yaml`, not `config.json`. Then validate the
directory, so that a malformed edit fails here rather than in the middle of a run. From the
repository root:

```sh
pip install -r catalog/scripts/requirements-dev.txt
python3 catalog/scripts/derive_graph.py --model qwen3-14b
go run ./validate/cmd/validate-catalog catalog
```

Both take any directory with the catalog's layout, so a copy elsewhere is derived and checked
the same way:

```sh
python3 catalog/scripts/derive_graph.py ~/scratch/catalog-wide-heads --model qwen3-14b
go run ./validate/cmd/validate-catalog ~/scratch/catalog-wide-heads
```

## Run with it

Name the edited directory with `--catalog`; the run uses it instead of the embedded catalog:

```sh
blis run --scenario path/to/scenario.yaml --catalog ~/scratch/catalog-wide-heads
```

To run a private model against the released data, put only its entry in a directory, for
example `models/<name>/`, and add it with `--catalog-overlay <dir>`. The rest of the catalog
stays the embedded copy. [Point BLIS at the catalog](point-blis.md#where-a-runs-catalog-comes-from)
states how overlay entries replace or add to the data.

## An experiment is labelled

The Result records the catalog's source as `path`, with its content hash and the hash of any
overlay. Its data are not the release's embedded data, so the run is experimental, and its
figures cannot be mistaken for, or published as, those of a release
([What a result records](point-blis.md#what-a-result-records)). To let others reproduce the
experiment, commit the edits and publish the commit, for example on a fork; the content hash
then names data anyone can fetch and check.
