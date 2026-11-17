# Original to blis-sglang.
"""Fail if the committed ServerArgs answers are not what SGLang resolves now.

config/testdata/serverargs.json holds SGLang v0.5.23's own ServerArgs(...).resolve_once() answers
for oracle/serverargs/cases.json (oracle/run_serverargs.py), which the Go test
(config TestResolveAgainstSGLang) reads. CI runs the same cases again and this compares each
case: its inputs, the model facts, acceptance or SGLang's refusal, every resolved field and the
step that decided it. So the answers the Go test reads pass only while the pinned release still
gives them. The provenance must name the same SGLang commit, case file and model revisions.

Usage: python3 -I oracle/compare_serverargs.py <committed answers> <fresh answers>
"""
import json
import sys


def main(committed: str, fresh: str) -> int:
    with open(committed) as f:
        a = json.load(f)
    with open(fresh) as f:
        b = json.load(f)
    if a["provenance"] != b["provenance"]:
        keys = sorted(k for k in set(a["provenance"]) | set(b["provenance"])
                      if a["provenance"].get(k) != b["provenance"].get(k))
        print(f"the committed answers and the fresh ones differ in provenance: {keys}")
        return 1
    ca, cb = a["cases"], b["cases"]
    if [c["name"] for c in ca] != [c["name"] for c in cb]:
        print(f"committed cases {[c['name'] for c in ca]}, fresh {[c['name'] for c in cb]}")
        return 1
    bad = 0
    for x, y in zip(ca, cb):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            detail = ""
            if "resolved" in keys and "resolved" in x and "resolved" in y:
                fields = sorted(k for k in set(x["resolved"]) | set(y["resolved"])
                                if x["resolved"].get(k) != y["resolved"].get(k))
                detail = f"; resolved fields {fields}"
            print(f"{x['name']}: {keys} differ{detail}")
            bad += 1
    print(f"{len(ca)} cases, {bad} differ from SGLang's fresh answers")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
