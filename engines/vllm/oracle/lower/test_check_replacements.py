# Original to blis-vllm.
"""check_replacements.py fails on the violation it exists to catch (X18): an extractor that
patches a name the list does not mention, and an entry of a kind the oracle may not use."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "check_replacements.py")
EXTRACT = os.path.join(HERE, "extract.py")
LIST = os.path.join(HERE, "replacements.json")


def run(extract, lst):
    return subprocess.run([sys.executable, "-I", CHECK, extract, lst], capture_output=True, text=True)


def main():
    good = run(EXTRACT, LIST)
    assert good.returncode == 0, good.stderr
    with tempfile.TemporaryDirectory() as d:
        src = open(EXTRACT, encoding="utf-8").read()
        bad_src = os.path.join(d, "extract.py")
        open(bad_src, "w", encoding="utf-8").write(src + '\n\ndef _sneak(ns):\n    ns["_has_global_unfinished_reqs"] = lambda *a: True\n')
        r = run(bad_src, LIST)
        assert r.returncode != 0 and "_has_global_unfinished_reqs" in r.stderr, "an unlisted patch passed"
        spec = json.load(open(LIST, encoding="utf-8"))
        spec["replacements"][0]["kind"] = "decision"
        bad_list = os.path.join(d, "r.json")
        json.dump(spec, open(bad_list, "w"))
        r = run(EXTRACT, bad_list)
        assert r.returncode != 0 and "decision" in r.stderr, "a decision-kind entry passed"
    print("ok: check_replacements.py catches an unlisted patch and a disallowed kind")


if __name__ == "__main__":
    main()
