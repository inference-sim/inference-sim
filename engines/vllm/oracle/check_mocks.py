# Original to blis-vllm.
"""Fail if oracle/mocks.json replaces decision code.

The oracle may replace the GPU, the clock, transports and external services, never a
scheduling, KV-cache or connector decision (#2021 B8). A mock entry whose symbol lies under a
decision module, or whose kind is not one of the allowed kinds, fails this check.
"""
import json
import sys

ALLOWED_KINDS = {"gpu", "clock", "transport", "environment", "external-service"}
DECISION_PREFIXES = (
    "vllm.v1.core.",
    "vllm.v1.engine.core",
    "vllm.distributed.kv_transfer.kv_connector.v1.",
    "vllm.v1.kv_offload.",
)


def main(path: str) -> int:
    with open(path) as f:
        spec = json.load(f)
    bad = []
    for m in spec["mocks"]:
        sym, kind = m["symbol"], m["kind"]
        if kind not in ALLOWED_KINDS:
            bad.append(f"{sym}: kind {kind!r} is not one of {sorted(ALLOWED_KINDS)}")
        # A clock patch inside a decision module (e.g. its time.monotonic) is allowed: the
        # clock is the environment even where a decision reads it.
        if kind != "clock" and sym.startswith(DECISION_PREFIXES):
            bad.append(f"{sym}: replaces decision code ({kind})")
    for b in bad:
        print("mocks.json:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "oracle/mocks.json"))
