# Original to blis-vllm.
"""Regenerate kernel_impls.json for a vLLM release in one command.

    python3 -I kernels/regen.py              # the release this build models (internal/upstream's Tag)
    python3 -I kernels/regen.py v0.33.0      # another release with a classification in kernels/
    python3 -I kernels/regen.py --check      # fail when kernel_impls.json is not current

It fetches vLLM at the tag, reads from vLLM's own build files the pin of every library the
release's classification (kernels/<tag>/attention.json) names, fetches each library's source at
that pin (a pip library at its release tag, a library vLLM's build fetches at the commit its
GIT_TAG names), and runs gen_kernel_impls.py over them. Each fetch is one commit, without history
or submodules: the generator reads a submodule's pinned commit from the tree. The fetched
sources are deleted afterwards; pass --keep DIR to keep them there.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_kernel_impls as gen  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_REPO = "https://github.com/vllm-project/vllm"


def modelled_tag():
    """The vLLM tag this build models: Tag in internal/upstream/upstream.go, the one place the
    release is named (provenance.go's Upstream reads it)."""
    rel = os.path.join("internal", "upstream", "upstream.go")
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        m = re.search(r'^\s*Tag\s*=\s*"([^"]+)"', f.read(), re.M)
    if not m:
        raise gen.Refused(f"{rel} names no Tag")
    return m.group(1)


def fetch(repo, ref, into):
    """One commit of repo at ref into the new directory into. A ref of 40 hexadecimal digits is a
    commit; any other is a release tag, fetched as a tag so the generator can check it."""
    if os.path.exists(into):
        raise gen.Refused(f"{into} exists; regen.py fetches into a new directory (choose another --keep)")
    os.makedirs(into)
    spec = ref if re.fullmatch(r"[0-9a-f]{40}", ref) else f"+refs/tags/{ref}:refs/tags/{ref}"
    for args in (["init", "-q"], ["fetch", "-q", "--depth", "1", repo, spec], ["checkout", "-q", "FETCH_HEAD"]):
        r = subprocess.run(["git", "-C", into, *args], capture_output=True, text=True)
        if r.returncode != 0:
            raise gen.Refused(f"fetching {repo} at {ref}: {r.stderr.strip()}")
    return into


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", nargs="?", help="the vLLM release tag (default: the Tag internal/upstream names)")
    ap.add_argument("--out", default=os.path.join(ROOT, "kernel_impls.json"), help="the table to write or check")
    ap.add_argument("--check", action="store_true", help="fail when --out is not current instead of writing it")
    ap.add_argument("--keep", metavar="DIR", help="fetch into DIR and keep the sources")
    args = ap.parse_args(argv)
    try:
        tag = args.tag or modelled_tag()
        cls = gen.load_classification(tag)
        if args.keep and os.path.exists(args.keep) and os.listdir(args.keep):
            raise gen.Refused(f"--keep {args.keep} is not empty; regen.py fetches into an empty directory")
        work = args.keep or tempfile.mkdtemp(prefix="kernel-impls-")
        try:
            vllm = fetch(VLLM_REPO, tag, os.path.join(work, "vllm"))
            sources = {}
            for name, (_, pin, repo, ref) in gen.library_pins(vllm, cls).items():
                if repo is not None:
                    print(f"regen: {name} at {ref} ({pin})", file=sys.stderr)
                    sources[name] = fetch(repo, ref, os.path.join(work, name))
            out = gen.render(gen.generate(vllm, cls, sources))
        finally:
            if not args.keep:
                shutil.rmtree(work, ignore_errors=True)
    except gen.Refused as e:
        print(f"regen: {e}", file=sys.stderr)
        return 1
    if args.check:
        with open(args.out, encoding="utf-8") as f:
            if f.read() != out:
                print(f"regen: {args.out} is not current for vLLM {tag}; run python3 -I kernels/regen.py {tag}",
                      file=sys.stderr)
                return 1
        return 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"regen: wrote {args.out} for vLLM {tag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
