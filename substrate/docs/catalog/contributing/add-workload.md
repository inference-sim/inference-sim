# Add a workload

A workload describes the lengths of requests: how long prompts are, how long outputs are, and how long a prefix the requests share. It does not say how many requests arrive. The arrival rate or concurrency is set by each run.

Create `workloads/<name>.yaml`. The file name is the workload's identity; the file has no `name` key.

```yaml
prefix_tokens: 0         # length of the shared prefix; 0 means none
prompt:
  tokens: 256            # mean
  tokens_stdev: 100
  tokens_min: 2
  tokens_max: 800
output:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 1
  tokens_max: 1024
```

All values are token counts per request. The validator enforces:

- each mean (`tokens`) is at least 1;
- no `tokens_stdev`, `tokens_min` or `prefix_tokens` is negative;
- where bounds are given, `tokens_min` ≤ `tokens` ≤ `tokens_max`;
- `prefix_tokens` does not exceed the mean prompt length.

Set the bounds with care. Two workloads with the same means but different tails can saturate a deployment at different loads.

To record where the numbers came from, such as a benchmark or a published trace, add a `_comment` key at the top level or inside `prompt:` or `output:`.

## A replay workload

A workload that a load generator replays from a published corpus, rather than draws from distributions, is a replay workload. Its file's top-level key is `replay:`, and it states what the harness runs: the scenario, the harness release, the corpus pinned by repository, revision and SHA-256, and the scenario's settings. It states no `concurrency`, which each run chooses. `workloads/agentx-mvp.yaml` is the example:

```yaml
replay:
  scenario: inferencex-agentx-mvp
  harness_version: 0.12.0
  corpus:
    loader: semianalysis_cc_traces_weka_062126_256k
    format: weka-trace-v7
    encoding: jsonl
    source:
      huggingface: semianalysisai/cc-traces-weka-062126-256k
      revision: 8fecd2fc56694469f758f0afbbb6335ad3043740   # a commit, never a branch
      file: traces.jsonl
    sha256: e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
  random_seed: 42
  benchmark_duration_s: 900
  max_context_length: max_model_len
```

Do not commit the corpus; a consumer fetches it at the revision and checks the digest. Record its license, the script that built it and its own published statistics in top-level `_comment` keys. The validator refuses any setting the harness's scenario validator would refuse; the [workload reference](../../contracts/reference/workload.md) lists the keys.

## Then

Run the [checks](checks.md) and open a pull request that cites the source of the numbers.
