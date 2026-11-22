# Original to blis-sglang.
"""Fail if the committed refusal answers are not what SGLang raises now.

sglangengine/testdata/sglang_refusals.json holds the message SGLang v0.5.23 raises for each case of
oracle/refusals/cases.json (oracle/run_refusals.py), which the Go test (sglangengine
TestRefusalsAreSGLangsOwn) holds the port's refusals to. CI runs the same cases again and this
compares them whole: the provenance (the SGLang commit and the case file's sha256), each case's name,
the quote it belongs to, and its answer, the exception type and message or "accepted". A missing
case, a foreign case, a changed message or an answer that turns from a refusal into an acceptance
fails.

Usage: python3 -I oracle/compare_refusals.py <committed answers> <fresh answers>
"""
import json
import sys


def diff(a: dict, b: dict) -> list:
    out = []
    if a.get("provenance") != b.get("provenance"):
        out.append("provenance differs: committed %s, fresh %s" % (a.get("provenance"), b.get("provenance")))
    ca = {c["name"]: c for c in a.get("cases", [])}
    cb = {c["name"]: c for c in b.get("cases", [])}
    for n in sorted(set(ca) - set(cb)):
        out.append("%s: committed but not run" % n)
    for n in sorted(set(cb) - set(ca)):
        out.append("%s: run but not committed" % n)
    for n in sorted(set(ca) & set(cb)):
        if ca[n] != cb[n]:
            out.append("%s: committed %s, fresh %s" % (n, json.dumps(ca[n], sort_keys=True), json.dumps(cb[n], sort_keys=True)))
    if [c["name"] for c in a.get("cases", [])] != [c["name"] for c in b.get("cases", [])] and not out:
        out.append("the cases are in another order")
    return out


def main(committed: str, fresh: str) -> int:
    with open(committed) as f:
        a = json.load(f)
    with open(fresh) as f:
        b = json.load(f)
    errs = diff(a, b)
    for e in errs:
        print("FAIL:", e)
    if errs:
        return 1
    print("%d refusal cases match SGLang's fresh answers" % len(a["cases"]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
