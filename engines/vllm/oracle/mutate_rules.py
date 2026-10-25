# Original to blis-vllm.
"""Mutation sweep for the tier-1 rules (rules/v0_32): apply one mutant at a time to a scratch copy
of the module, run the rule's unit test and its MatchSource check, and report whether each is caught.

Each mutant breaks one clause of a rule a supported row of docs/feature-matrix.md names, so the
row's mutants exist and run in CI rather than only on a contributor's machine.

Usage: python3 oracle/mutate_rules.py <blis-vllm checkout> <scratch dir> [R1,...]
Exits non-zero if a mutant survives or breaks the build without a test failure.
"""
import os
import shutil
import subprocess
import sys

PACK = "rules/v0_32/pack.go"

# (id, the clause it breaks, file, exact old text, new text)
MUTANTS = [
    ("D1", "dcp-divides-tp: with pcp off, tp % dcp must be 0", PACK,
     "case pcp == 1 && tp%dcp != 0:", "case pcp == 1 && tp%dcp != 0 && false:"),
    ("D2", "dcp-divides-tp: the modulus is tp by dcp, not dcp by tp", PACK,
     "case pcp == 1 && tp%dcp != 0:", "case pcp == 1 && dcp%tp != 0:"),
    ("D3", "dcp-divides-tp: with pcp on, dcp may span the pcp axis", PACK,
     "case pcp > 1 && dcp != 1 && dcp != pcp && dcp != tp*pcp:",
     "case pcp > 1 && dcp != 1 && dcp != tp*pcp:"),
    ("D4", "dcp-divides-tp: with pcp on, dcp may span tp x pcp", PACK,
     "case pcp > 1 && dcp != 1 && dcp != pcp && dcp != tp*pcp:",
     "case pcp > 1 && dcp != 1 && dcp != pcp:"),
    ("D5", "dcp-divides-tp: an unstated dcp is one, not zero", PACK,
     "tp, pcp, dcp := pl.TP, max(pl.PCP, 1), max(pl.DCP, 1)",
     "tp, pcp, dcp := pl.TP, max(pl.PCP, 1), max(pl.DCP, 0)"),
]

TESTS = "TestDCPDividesTP"


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    bad = 0
    for mid, row, path, old, new in MUTANTS:
        if only and mid not in only:
            continue
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid} {row}: the mutation site is not unique in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new, 1))
        r = subprocess.run(["go", "test", "-count=1", "-run", TESTS, "./v0_32/"],
                           cwd=os.path.join(scratch, "rules"), capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"SURVIVED {mid} {row}")
            bad += 1
        elif "build failed" in out or "[setup failed]" in out:
            print(f"BROKEN   {mid} {row}: the mutant does not compile")
            bad += 1
        else:
            print(f"CAUGHT   {mid} {row}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
