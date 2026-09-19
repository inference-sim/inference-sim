# Test-catalog workload presets

The named workload presets (`chatbot`, `summarization`, `contentgen`, `multidoc`) that
`blis run --workload`, `blis convert preset --name` and `blis observe --workload` resolve.

`--catalog` / `BLIS_CATALOG` names the catalog **clone root** (#1774), and a preset is read
from `<catalog>/workloads/<name>.yaml` — the sibling of the `models/` namespace. These files
are byte-for-byte the definitions that used to be duplicated in the `workloads:` block of the
repository's `defaults.yaml`, which #1769 deleted: they now exist once.

They are the same four files the authoritative
[`blis-catalog`](https://github.com/inference-sim/blis-catalog) repository ships under
`workloads/`. This directory (`testdata/catalog/`) is a small, clone-root-shaped **test
fixture** mirroring that repository, so the Go tests can resolve presets and model configs
offline; #1771 deleted the former in-repo `model_configs/` tree, so operators clone
`blis-catalog` and point `--catalog` / `BLIS_CATALOG` at it.

A preset file is parsed strictly (an unrecognized key is a hard error, R10) and must declare a
positive `prompt_tokens` and `output_tokens`.
