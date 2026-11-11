# Original to blis-vllm.
"""Synthetic-acceptance oracle: what vLLM v0.32.0 resolves speculative_config.synthetic_acceptance_length to.

With rejection_sample_method "synthetic", vLLM turns the stated mean acceptance length into
per-position unconditional acceptance rates before the rejection sampler is built:

  SpeculativeConfig._resolve_synthetic_acceptance_rates   vllm/config/speculative.py:566-598
  SpeculativeConfig._acceptance_length_to_rates           vllm/config/speculative.py:557-564

This runs vLLM's own function over a grid of (num_speculative_tokens, length), valid and not, and
records the rates it returns or the refusal it raises. Nothing is replaced (design FID-1): the
function is a static method and reads no GPU. What this does not cover is the Triton rejection
sampler that consumes the rates (vllm/v1/sample/rejection_sampler.py:734-761), which needs a GPU
and whose sampled acceptance distribution therefore cannot be compared on a CPU; the record also
states, from the source text, that the length is refused unless rejection_sample_method is
synthetic (vllm/config/speculative.py:1843-1847).

Usage:
    python3 -I oracle/synthetic_acceptance/oracle.py <vllm checkout> json              print the record as JSON
    python3 -I oracle/synthetic_acceptance/oracle.py <vllm checkout> check <file>      fail unless <file> equals what it prints

rules/engineconfig/synthetic_acceptance_test.go reads the committed file and holds the engine
block's validation and blis-substrate's acceptance.MinimumVarianceRates to it.
"""
import json
import os
import sys
import warnings

COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"


def lengths(k):
    """Valid lengths across [1, k+1], and values just outside it."""
    ok = {1.0, 1.1, 1.5, 2.0, 2.5, 2.84, 3.0, 10.0 / 3.0, float(k), k + 0.5, k + 0.9999, float(k + 1)}
    bad = {0.0, 0.99, -1.0, k + 1.0001, k + 2.0}
    return sorted(ok | bad)


def run(root):
    warnings.simplefilter("ignore")
    sys.path.insert(0, os.path.abspath(root))
    from vllm.config.speculative import SpeculativeConfig

    resolve = SpeculativeConfig._resolve_synthetic_acceptance_rates
    cases = []
    for k in range(1, 9):
        for length in lengths(k):
            case = {"num_speculative_tokens": k, "length": length}
            try:
                case["rates"] = resolve(k, None, length)
            except ValueError as e:
                case["error"] = str(e)
            cases.append(case)
    # Both given, neither given: the function requires exactly one.
    for rates, length in [([1.0, 0.5], 2.0), (None, None)]:
        case = {"num_speculative_tokens": 2, "rates": None if rates is None else rates, "length": length}
        try:
            resolve(2, rates, length)
            case["error"] = None
        except ValueError as e:
            case["error"] = str(e)
        cases.append(case)
    with open(os.path.join(root, "vllm/config/speculative.py"), encoding="utf-8") as f:
        text = f.read()
    return {
        "commit": COMMIT,
        "cases": cases,
        "length_needs_synthetic": "are only valid with rejection_sample_method='synthetic'" in text,
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
