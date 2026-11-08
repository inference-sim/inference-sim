# Recorded block sizes

`recorded_block_sizes.json` holds the KV block size that InferenceX runs used, read from each run's
vLLM server log, to check `core/vllm.ResolveBlockSize` against runs (design X19, VLM-1). It holds
derived values only: no log text. `rows.json` is the list of rows it was built from, and `fetch.py`
writes both.

## What the three kinds of row prove

| evidence | the log shows | proves |
|---|---|---|
| `engine_line` | `Setting kv cache block size to N for <backends>` | vLLM picked N. This validates the resolution. |
| `silent` | vLLM v0.18.0 or later, no `block_size` among the non-default arguments, no engine line | vLLM kept the default, 16 (`vllm/platforms/interface.py:757-781@5df63a72` logs the line only for another size). This validates a resolution of 16, with the caveat that a log cannot tell it from an explicit `--block-size 16`. |
| `args` | `block_size` among the non-default arguments | The operator passed a flag. This says nothing about what vLLM picks when none is passed, so it validates nothing. A difference from the resolution is named, by row id, in `TestResolutionAgainstTheRecordedRuns`. |

The 354 committed fixtures that say `# block_size N: stated by the run.` are operator flags too. They
are compared with the resolution by `TestBlockSizeResolutionOverTheCensusFixtures`, which names every
difference by fixture path; they are not evidence of vLLM's choice.

## The rows and their coverage

152 rows were walked (see `models` in the JSON for the slugs and display names): 108 have a block
size and 44 are listed under `unobtainable` with a reason, which is also the count the test pins:

- 15 rows have no server log in the public API (all `llama70b`, whose rows date from October 2025);
- 22 rows have no `Initializing a V1 LLM engine (v...)` line in their log, so the vLLM version is
  unknown (a custom image or an agentic client log): 10 of `dsv41flash`, 8 of `minimaxm3`, 4 of `dsv4`;
- 7 rows are vLLM 0.15.1 to 0.17.0, which logs no block-size line (it exists from v0.18.0).

Of the 108: 14 `engine_line` (all kimi-k2.5 on Blackwell, size 32), 31 `silent` (gpt-oss-120b 18,
minimax-m2.5 12, kimi-k2.5 1) and 63 `args`. `TestResolutionAgainstTheRecordedRuns` logs the coverage
by model and pins every count. The earlier statement that two rows plus thirteen or fifteen were
missing counted a different set of rows; these are the real numbers.

Models the table resolves with no run that shows what vLLM picked: llama-3.1-70b-instruct (the only
Llama InferenceX serves with vLLM is `Llama-3.3-70B-Instruct-FP8`, a different checkpoint, and its
rows have no server log), minimax-m2.7 (InferenceX has no such slug: `minimaxm2.7` is not in
`/api/v1/availability`, and `/benchmarks` rejects the name), deepseek-v3 (`dsr1` has no vLLM rows).
Models the table refuses with runs that show what vLLM picked: none with an engine line; deepseek-v4-pro
and minimax-m3 have operator flags only.

## Regenerating

    python3 -I fetch.py --models gptoss120b=gpt-oss-120b,llama70b=Llama-3.3-70B-Instruct-FP8,minimaxm2.5=MiniMax-M2.5,kimik2.5=Kimi-K2.5,minimaxm3=MiniMax-M3,dsv4=DeepSeek-V4-Pro,dsv41flash=DeepSeek-V4.1-Flash \
        --rows rows.json --out recorded_block_sizes.json
    python3 -I fetch.py --from-rows rows.json --out recorded_block_sizes.json   # same rows
    python3 -I fetch.py --check recorded_block_sizes.json                        # exit 1 if a log changed

Each run row records `log_sha256` (the sha256 of the server log as the API served it) and `fetched`
(the date). Re-running with `--from-rows` while the hashes still match writes the file byte for
byte. A nightly job that runs `--check` is blis-vllm#91; it is not built.

Source: InferenceX, https://inferencex.semianalysis.com/api/v1 (`/benchmarks?model=` for the rows,
`/server-log?id=` for the logs). Rows are identified by their InferenceX row id.
