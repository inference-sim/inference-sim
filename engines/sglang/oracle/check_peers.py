# Original to blis-sglang.
"""Fail when oracle/requirements.lock pins a package oracle/peers.json names at another version
than peers.json gives (FID-2, FID-12): a KV-transfer or KV-store client enters the oracle only at
the version SGLang's own requirements or CI resolve. A peer that is not in the lock is not
checked; peers.json says how a scenario that selects it is refused.

Usage: python3 -I oracle/check_peers.py [oracle/peers.json] [oracle/requirements.lock]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def problems(peers: dict, lock_lines: list) -> list:
    want = {canonical(p["name"]): p["version"] for p in peers["peers"]}
    bad = []
    for line in lock_lines:
        s = line.split("#", 1)[0].strip()
        if "==" not in s:
            continue
        name, version = (x.strip() for x in s.split("==", 1))
        if canonical(name) in want and version != want[canonical(name)]:
            bad.append(f"requirements.lock pins {name}=={version}; oracle/peers.json pins it at {want[canonical(name)]}")
    return bad


def main(argv: list) -> int:
    peers_path = argv[0] if argv else os.path.join(HERE, "peers.json")
    lock_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "requirements.lock")
    with open(peers_path) as f:
        peers = json.load(f)
    if not peers.get("peers"):
        print(f"{peers_path}: names no peer", file=sys.stderr)
        return 1
    with open(lock_path) as f:
        bad = problems(peers, f.readlines())
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
