# Original to blis-vllm.
"""Quantization oracle: what vLLM v0.32.0 resolves `quantization` to on a DeepSeek-V4 FP8 checkpoint.

A deployment states `quantization` (or none), and vLLM's ModelConfig replaces it when the
checkpoint's own quantization_config asks for another method:

  ModelConfig._verify_quantization        vllm/config/model.py:1370-1412
  DeepseekV4FP8Config.override_quantization_method
                                          vllm/models/deepseek_v4/quant_config.py:140-157

The override reads the checkpoint's quant_method (fp8) and model_type (deepseek_v4), so this runs
vLLM's own ModelConfig over the catalog's own config.json for deepseek-v4-pro, once for each
stated value, and records the method each resolves to. Only the GPU is absent: a CPU falls back
from bfloat16 to float16, which no quantization decision reads, and nothing is replaced (design
FID-1). The config.json is the catalog's (BLIS_ORACLE_CATALOG), or fetched at the pinned revision
and checked against its SHA-256 (oracle/blocksize/modelconfig.py).

Usage:
    python3 -I oracle/quantization/oracle.py <vllm checkout> json              print the resolutions as JSON
    python3 -I oracle/quantization/oracle.py <vllm checkout> check <file>      fail unless <file> equals what it prints

compat/kernelmodel/quantization_v4_test.go reads the committed file and holds the kernel's plan to it.
"""
import hashlib
import json
import os
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blocksize"))

COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
CATALOG = os.environ.get("BLIS_ORACLE_CATALOG", "")
MODEL = "deepseek-v4-pro"
# None is a deployment that states no quantization. deepseek_v4_fp8 is the value the dashboard's
# points state; the rest are values a user could name for the same checkpoint.
STATED = [None, "deepseek_v4_fp8", "fp8", "awq", "nvfp4"]


def run(vllm_dir):
    warnings.simplefilter("ignore")
    import modelconfig  # pyright: ignore[reportMissingImports]

    data = modelconfig.config_bytes(MODEL, os.path.join(CATALOG, MODEL))
    sys.path.insert(0, vllm_dir)
    from vllm.config import ModelConfig

    cfg = json.loads(data)

    def resolve(config_bytes, stated):
        rows = []
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "wb") as f:
                f.write(config_bytes)
            for q in stated:
                mc = ModelConfig(model=d, quantization=q, skip_tokenizer_init=True, dtype="auto", seed=0,
                                 enforce_eager=True)
                rows.append({"stated": q, "resolved": mc.quantization})
        return rows

    rows = resolve(data, STATED)
    # The same config without its quantization_config (an unquantized checkpoint): the override
    # does not apply, so a stated value stays as stated. It shows the answers above come from the
    # checkpoint's quant_method, not from the oracle.
    bare = {k: v for k, v in cfg.items() if k != "quantization_config"}
    control = resolve(json.dumps(bare).encode(), [None, "fp8"])
    return {
        "vllm": COMMIT,
        "model": MODEL,
        "config_sha256": hashlib.sha256(data).hexdigest(),
        "model_type": cfg["model_type"],
        "checkpoint_quant_method": cfg["quantization_config"]["quant_method"],
        "cases": rows,
        "control_without_quantization_config": control,
    }


def main(argv):
    if len(argv) < 3 or argv[2] not in ("json", "check") or (argv[2] == "check" and len(argv) < 4):
        print(__doc__)
        return 2
    out = json.dumps(run(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "json":
        sys.stdout.write(out)
        return 0
    with open(argv[3], encoding="utf-8") as f:
        if f.read() != out:
            print(f"{argv[3]} differs from what the oracle prints; regenerate it")
            return 1
    print(f"{argv[3]}: current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
