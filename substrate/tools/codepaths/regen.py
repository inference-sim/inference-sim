# Original to blis-substrate.
"""Regenerate an engine's code_paths.json in one command.

    python3 -I regen.py --spec <engine>/code_paths/spec.json --out <engine>/code_paths.json
    python3 -I regen.py --spec <engine>/code_paths/spec.json --out <engine>/code_paths.json --check

where <engine> is the engine module's directory (the substrate names no path of one, X7).
After a change, run export_code_ids.py so the registry's keys follow.

It fetches the engine's source at the release tag the spec names (one commit, no history), checks
that the tag is the spec's commit, runs gen_code_paths.py over it, and deletes the source. --keep DIR
keeps it there. --check fails when --out is not what the source gives instead of writing it.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_code_paths as gen  # noqa: E402


def fetch(repo, tag, into, commit=None):
    """One commit of repo at the release tag into the new directory into. An engine pinned at a
    commit no release names has no tag to fetch: its spec's tag is then not a release tag, and the
    commit (40 hexadecimal digits) is fetched instead."""
    if os.path.exists(into) and os.listdir(into):
        raise gen.Refused(f"{into} is not empty; regen.py fetches into an empty directory")
    os.makedirs(into, exist_ok=True)
    if re.fullmatch(r"v[0-9][0-9A-Za-z.\-]*", tag):
        steps = (["init", "-q"], ["fetch", "-q", "--depth", "1", repo, f"+refs/tags/{tag}:refs/tags/{tag}"],
                 ["checkout", "-q", tag])
    elif commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit):
        steps = (["init", "-q"], ["fetch", "-q", "--depth", "1", repo, commit], ["checkout", "-q", "FETCH_HEAD"])
    else:
        raise gen.Refused(f"the spec's tag {tag!r} is not a release tag and its commit is not a full commit")
    for args in steps:
        r = subprocess.run(["git", "-C", into, *args], capture_output=True, text=True)
        if r.returncode != 0:
            raise gen.Refused(f"fetching {repo} at {tag}: {r.stderr.strip()}")
    return into


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="the engine's code_paths/spec.json")
    ap.add_argument("--out", required=True, help="the table to write or check")
    ap.add_argument("--check", action="store_true", help="fail when --out is not current instead of writing it")
    ap.add_argument("--keep", metavar="DIR", help="fetch into DIR and keep the source")
    args = ap.parse_args(argv)
    try:
        spec = gen.load_spec(args.spec)
        engine = spec["engine"]
        for key in ("tag", "repo"):
            if key not in engine:
                raise gen.Refused(f"{args.spec} names no engine.{key}")
        work = args.keep or tempfile.mkdtemp(prefix="code-paths-")
        try:
            src = fetch(engine["repo"], engine["tag"], os.path.join(work, engine["kind"]), engine.get("commit"))
            out = gen.render(gen.generate(src, spec))
        finally:
            if not args.keep:
                shutil.rmtree(work, ignore_errors=True)
    except gen.Refused as e:
        print(f"regen: {e}", file=sys.stderr)
        return 1
    if args.check:
        with open(args.out, encoding="utf-8") as f:
            if f.read() != out:
                print(f"regen: {args.out} is not current for {engine['kind']} {engine['version']}; "
                      f"run python3 -I regen.py --spec {args.spec} --out {args.out}", file=sys.stderr)
                return 1
        return 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"regen: wrote {args.out} for {engine['kind']} {engine['version']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
