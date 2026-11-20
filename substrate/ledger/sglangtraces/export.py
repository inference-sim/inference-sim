#!/usr/bin/env python3
"""List the traces tree of a commit of sgl-project/ci-data for ledger/sglangtraces.

    export.py --repo sgl-project/ci-data --commit SHA > listing.jsonl

Prints a header line {repo, commit, tree}, then one {path, sha, size} line per file under
traces/, in increasing path order. It walks the tree one run directory at a time because the
recursive tree of the whole directory is truncated by the API. It reads only git tree entries
(blob SHA-1 and size), never a trace, and needs `gh` authenticated for the API.
"""
import argparse
import json
import subprocess
import sys


def api(path):
    out = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()

    root = api(f"repos/{a.repo}/git/trees/{a.commit}")
    traces = next(e for e in root["tree"] if e["path"] == "traces" and e["type"] == "tree")
    files = []
    for run in api(f"repos/{a.repo}/git/trees/{traces['sha']}")["tree"]:
        if run["type"] != "tree":
            files.append({"path": f"traces/{run['path']}", "sha": run["sha"], "size": run.get("size", 0)})
            continue
        sub = api(f"repos/{a.repo}/git/trees/{run['sha']}?recursive=1")
        if sub.get("truncated"):
            sys.exit(f"the tree of run {run['path']} is truncated")
        for e in sub["tree"]:
            if e["type"] == "blob":
                files.append({"path": f"traces/{run['path']}/{e['path']}", "sha": e["sha"], "size": e["size"]})
    files.sort(key=lambda f: f["path"])
    dump = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    print(dump({"repo": a.repo, "commit": a.commit, "tree": traces["sha"]}))
    for f in files:
        print(dump(f))


if __name__ == "__main__":
    main()
