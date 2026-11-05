#!/usr/bin/env python3
"""Write the one-at-a-time attribution variants of price_change_report.py.

    scripts/kernel/attribution_variants.py NEW_AISIMULATE_DIR OUT_DIR

For five representative fixtures, writes OUT_DIR/<variant>/<fixture> for variant "new" (the
fixture as regenerated, with block_size 16 stated because the engine refuses an unstated one for
some models) and for each setting the old generator wrote, stated on top of it.
"""
import os
import sys

FIXTURES = ["llama-3.1-70b-instruct-h200-fp8-vllm-tp2.yaml", "gpt-oss-120b-h200-fp4-vllm-tp4.yaml",
            "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml", "llama-3.1-70b-instruct-b200-fp4-vllm-tp1.yaml",
            "deepseek-v3-b200-fp4-sglang-tp4.yaml"]
VARIANTS = {"new": {}, "cache_dtype_fp8": {"cache_dtype": "fp8"}, "max_num_seqs_256": {"max_num_seqs": "256"},
            "max_num_batched_tokens_8192": {"max_num_batched_tokens": "8192"},
            "cudagraph_mode_PIECEWISE": {"cudagraph_mode": "PIECEWISE"}, "block_size_16": {}}


def main(src, out):
    for name, settings in VARIANTS.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for f in FIXTURES:
            text = open(os.path.join(src, f)).read()
            lines = "".join(f"      {k}: {v}\n" for k, v in settings.items())
            if "      block_size:" not in text:
                lines += "      block_size: 16\n"
            marker = "      max_model_len:"
            if marker not in text:
                sys.exit(f"{f}: no max_model_len line to insert before")
            open(os.path.join(d, f), "w").write(text.replace(marker, lines + marker, 1))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
