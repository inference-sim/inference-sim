# Original to blis-vllm.
"""Fail unless the installed oracle environment is exactly the committed lock.

oracle/requirements.lock pins every package of the oracle's environment. CI installs it with
--no-deps and then compares `pip freeze` with it, so the environment a trace records
(provenance lockfile_sha256) is the committed one, with nothing added and nothing resolved
differently.

CI installs the lock from oracle/requirements-hashed.txt, which oracle/lock.sh writes from it with
the sha256 of every archive, under pip install --require-hashes (X21). --hashed checks that file
names exactly the packages and versions the lock does, so the hashed install and the lock a trace
records cannot drift apart.

Usage: python3 -I oracle/check_lock.py <oracle/requirements.lock> <pip freeze output>
       python3 -I oracle/check_lock.py --hashed <oracle/requirements.lock> <oracle/requirements-hashed.txt>
"""
import re
import sys


def pins(path: str) -> set[str]:
    with open(path) as f:
        return {l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")}


def canonical(pin: str) -> str:
    """A pin with its project name normalised as PEP 503 does (docstring_parser and
    docstring-parser are one project), so two spellings of one pin compare equal."""
    name, sep, rest = re.split(r"(==| @ )", pin, maxsplit=1) if re.search(r"==| @ ", pin) else (pin, "", "")
    return re.sub(r"[-_.]+", "-", name).lower() + sep + rest


def hashed_pins(path: str) -> set[str]:
    """The pins of a --generate-hashes file: each requirement line without its hash options."""
    out = set()
    with open(path) as f:
        for l in f:
            s = l.strip()
            if not s or s.startswith("#") or s.startswith("--hash"):
                continue
            out.add(s.split(" \\")[0].strip().removesuffix("\\").strip())
    return out


def main_hashed(lock: str, hashed: str) -> int:
    want = {canonical(p) for p in pins(lock)}
    got = {canonical(p) for p in hashed_pins(hashed)}
    for p in sorted(want - got):
        print(f"{hashed}: {p} is in the lock but not hashed")
    for p in sorted(got - want):
        print(f"{hashed}: {p} is hashed but not in the lock")
    return 1 if want != got else 0


def main(lock: str, frozen: str) -> int:
    want, got = pins(lock), pins(frozen)
    for p in sorted(want - got):
        print(f"{lock}: {p} is not installed")
    for p in sorted(got - want):
        print(f"{frozen}: {p} is installed but not in the lock")
    return 1 if want != got else 0


if __name__ == "__main__":
    if sys.argv[1] == "--hashed":
        sys.exit(main_hashed(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1], sys.argv[2]))
