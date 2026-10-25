# Original to blis-vllm.
"""Fail unless the installed oracle environment is exactly the committed lock.

oracle/requirements.lock pins every package of the oracle's environment. CI installs it with
--no-deps and then compares `pip freeze` with it, so the environment a trace records
(provenance lockfile_sha256) is the committed one, with nothing added and nothing resolved
differently.

Usage: python3 -I oracle/check_lock.py <oracle/requirements.lock> <pip freeze output>
"""
import sys


def pins(path: str) -> set[str]:
    with open(path) as f:
        return {l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")}


def main(lock: str, frozen: str) -> int:
    want, got = pins(lock), pins(frozen)
    for p in sorted(want - got):
        print(f"{lock}: {p} is not installed")
    for p in sorted(got - want):
        print(f"{frozen}: {p} is installed but not in the lock")
    return 1 if want != got else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
