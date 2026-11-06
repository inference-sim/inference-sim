#!/usr/bin/env python3
"""Behavioural tests for the args-line parser.

Both defects these cover were latent on the committed corpus, and were found by running
the parser against a real log rather than by reading it: none of the 68 committed sweeps
is a speculative arm, so neither produced a wrong committed value. Both would have fired
on the first MTP row, and 1,345 of the 5,090 vLLM rows on NVIDIA parts carry one.

Run: python3 scripts/test_extract_inferencex_engine_settings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_inferencex_engine_settings import args_body, parse_args_line  # noqa: E402

# A real MiniMax-M3 MTP args line, shortened to the fields that matter. The nesting is
# what the parser must survive: speculative_config is last, it holds its own `model`
# key, and its closing brace is immediately followed by the args dict's own.
MTP_LOG = (
    "non-default args: {'model_tag': 'MiniMaxAI/MiniMax-M3-MXFP8', "
    "'model': 'MiniMaxAI/MiniMax-M3-MXFP8', 'tensor_parallel_size': 8, "
    "'max_model_len': 2304, 'block_size': 128, 'enable_prefix_caching': False, "
    "'gpu_memory_utilization': 0.9, 'stream_interval': 20, "
    "'max_cudagraph_capture_size': 64, "
    "'speculative_config': {'method': 'eagle3', "
    "'model': 'Inferact/MiniMax-M3-EAGLE3', 'num_speculative_tokens': 3, "
    "'attention_backend': 'FLASH_ATTN'}}\n(APIServer pid=1)"
)

PLAIN_LOG = (
    "non-default args: {'model': 'MiniMaxAI/MiniMax-M3-MXFP8', "
    "'tensor_parallel_size': 4, 'block_size': 128, "
    "'max_cudagraph_capture_size': None}\n(APIServer pid=1)"
)

FAILURES = []


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    got = parse_args_line(MTP_LOG)

    # The defect: a depth-blind search for 'model' finds the one inside
    # speculative_config, so the served checkpoint is reported as the draft. Those are
    # different models with different graphs, so the prediction would be wrong.
    ck("the served model is returned, not the draft",
       got.get("model") == "MiniMaxAI/MiniMax-M3-MXFP8", repr(got.get("model")))
    ck("the draft model is recorded separately",
       got.get("draft_model") == "Inferact/MiniMax-M3-EAGLE3",
       repr(got.get("draft_model")))

    # A step that accepts drafts serves more than one token per request, so these change
    # what an ITL reading means and cannot be left unread.
    ck("num_speculative_tokens is read from inside the nested dict",
       got.get("num_speculative_tokens") == 3,
       repr(got.get("num_speculative_tokens")))
    ck("the speculative method is read", got.get("speculative_method") == "eagle3",
       repr(got.get("speculative_method")))

    # The second defect: a non-greedy brace match ends inside speculative_config, so
    # every field after it is lost. These sit before it and were among the ones dropped.
    ck("fields before the nested dict survive",
       (got.get("block_size"), got.get("max_model_len")) == (128, 2304),
       f"block_size={got.get('block_size')} "
       f"max_model_len={got.get('max_model_len')}")
    ck("a field after the nested dict is not required to be lost",
       got.get("max_cudagraph_capture_size") == 64,
       repr(got.get("max_cudagraph_capture_size")))
    ck("the body spans the whole args dict",
       (args_body(MTP_LOG) or "").rstrip().endswith("}"),
       repr((args_body(MTP_LOG) or "")[-24:]))

    # A literal None is the absence of a setting, not the string "None": recording the
    # string makes a run that passed no capture size look like one that passed a value.
    plain = parse_args_line(PLAIN_LOG)
    ck("a literal None is absent rather than the string 'None'",
       plain.get("max_cudagraph_capture_size") is None,
       repr(plain.get("max_cudagraph_capture_size")))
    ck("a log with no nested dict still parses",
       plain.get("model") == "MiniMaxAI/MiniMax-M3-MXFP8"
       and plain.get("block_size") == 128)
    ck("a log with no args line yields nothing",
       parse_args_line("INFO: starting up") == {})

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
