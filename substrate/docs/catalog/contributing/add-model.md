# Add a model

A model entry is a directory holding three files:

```
models/<name>/
├── config.json   the vendor's file, byte for byte, never edited
├── model.yaml    which model this is, and exactly where config.json came from
└── graph.yaml    the model graph, generated from config.json by a script
```

You copy `config.json`, write `model.yaml`, and generate `graph.yaml`. The steps use Qwen3-14B as the example. Read [Model graphs](../concepts/model-graphs.md) first if the term is new to you.

## 1. Choose the name

The directory name is the model's name everywhere in BLIS; a scenario's `model:` key uses it exactly. Use lowercase, and start from the Hugging Face repository name: `qwen3-14b` for `Qwen/Qwen3-14B`. Existing entries shorten long names, for example `nemotron-3-ultra-550b-a55b-bf16` for `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`. When several checkpoints of one model are catalogued, keep the part of the name that tells them apart, as in `glm-5.2` and `glm-5.2-fp8`.

## 2. Copy the vendor config at a pinned revision

Look up the commit the config will come from, then download the file at that commit:

```sh
repo=Qwen/Qwen3-14B
name=qwen3-14b
rev=$(curl -fsS https://huggingface.co/api/models/$repo/revision/main \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
mkdir -p models/$name
curl -fsSL -o models/$name/config.json \
     https://huggingface.co/$repo/resolve/$rev/config.json
```

For a gated repository, one that requires accepting its terms before download, add `-H "Authorization: Bearer $HF_TOKEN"` to both `curl` commands, and check that the model's license allows its configuration file to be redistributed.

Do not reformat the file. A pretty-printer, or an editor that trims whitespace, changes its bytes, and the graph records a digest of those bytes. The repository's `.gitattributes` turns off line-ending conversion for `config.json` for the same reason. A reviewer can repeat your download and compare the two files with `cmp`.

## 3. Write `model.yaml`

```yaml
name: qwen3-14b
source:
  provider: huggingface
  repo: Qwen/Qwen3-14B          # as on the Hub; case matters
  revision: 40c069824f4251a91eefaf281ebe4c544efd3e18
  retrieved: 2026-09-16
```

`name` must equal the directory name. `provider`, `repo` and `revision` are required. `revision` is the full commit hash from step 2, not a branch name, which could move after the graph is derived. `retrieved` is the date of the download; it is optional, and every current entry has it.

## 4. Derive the graph

```sh
pip install -r requirements-dev.txt
python3 scripts/derive_graph.py --model qwen3-14b
```

The script reads `config.json` and writes `graph.yaml` beside it. If it reports `1 written`, go to step 5.

### If the architecture is new

The deriver chooses a *handler*, a function that builds the layers of one family of architectures, by the first entry of the config's `architectures` list. When no handler is registered for that architecture, the deriver stops with an error that names it:

```
no handler for architecture 'Qwen9ForCausalLM'; add one rather than letting a default misprice the model
```

Add a handler to `scripts/derive_graph.py` and register it in the `HANDLERS` table. Many architectures differ from a supported one only in values such as depth, width or expert count, and can reuse its handler; the comments in `HANDLERS` record several such cases. Add tests for the new handler to `tests/test_derive_graph.py`.

If the model has a property that no field of the graph format can express, the change starts in [blis-schemas](https://github.com/inference-sim/blis-schemas), followed by the kernel. That has happened a few times. Kimi-K2.5 added the `int4` weight format. DeepSeek-V4-Pro added a per-node `weight_dtype`, for experts stored at a narrower width than the rest of the checkpoint, and `compress_ratio`, for attention that reads a compressed cache.

## 5. Review the graph

Read `graph.yaml` against `config.json`. The model's page in a [local preview](releasing.md#preview-locally) of this site is the quickest way: it draws the layer stack and tabulates every node. Check the layer count, the attention head counts, the expert count and top-k, and whether a draft module is present.

## 6. Run the checks and open a pull request

Run the [checks](checks.md). In the pull request, link the Hugging Face repository at the revision you used.
