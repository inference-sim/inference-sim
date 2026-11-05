# Experiment with a scratch clone

To ask what BLIS predicts for a model or a chip that does not exist yet, edit a copy of the catalog and point BLIS at the copy. You need no fork of the simulator and no pull request.

```sh
git clone --branch <release> https://github.com/inference-sim/blis-catalog.git ~/scratch/catalog-wide-heads
cd ~/scratch/catalog-wide-heads
# edit models/qwen3-14b/config.json, hardware/h100.yaml, ...
export BLIS_CATALOG=~/scratch/catalog-wide-heads
```

If you edit a model's `config.json`, derive its graph again, so that `graph.yaml` matches the edited config; the kernel reads `graph.yaml`, not `config.json`. Then run the schema check, so that a malformed edit fails here rather than in the middle of a run:

```sh
pip install -r requirements-dev.txt
python3 scripts/derive_graph.py --model qwen3-14b
go run ./validate/cmd/validate-catalog catalog
```

## Keep the clone separate

Make the scratch catalog a git clone of its own, outside any other repository. Git reports edits only within its own repository. A plain copy placed inside another repository that ignores it looks unedited to git, even though its contents match no commit.

## Keep experiments attributable

Record the clone's state with each result, as [Point BLIS at the catalog](point-blis.md#record-which-catalog-a-result-came-from) describes. An edited clone shows its edits in `git status`, so an experiment cannot be mistaken for a run of a published release. To make the experiment reproducible by others, commit the edits and publish the commit, for example on a fork. A commit that exists only on your machine names data no one else can fetch.
