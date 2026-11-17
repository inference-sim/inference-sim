# Original to blis-vllm.
"""Move the scenario files and test specs that state the release the engine executes to the next one.

A build models one vLLM release (provenance.go, internal/upstream). A scenario file, or a test,
that states that release by number must move with it, as commit 4ea320e1 moved the kernel tests'
fixtures to 0.32.0. This rewrites only the shapes that state the executing release:

    engine_version: "X"  /  engine_version: X       (a scenario file, or one embedded in a test)
    version: "X"  inside  engine: {kind: vllm, ...}   (the same, in the engine block's form)
    UpstreamVersion: "X"  /  EngineVersion: "X"       (an engine.Spec or a scenario in Go)
    (t, "X"                                           (a test helper that opens a pool at X)

With --header OLDCOMMIT NEWCOMMIT it also moves a ported Go file's provenance header, the
line `// Derived from vllm/<path> at vLLM v<old> (<old commit>).` in the file's first comment
block, which TestEveryFileStatesItsProvenance requires to name the release the build models
(X3). The header changes only when the line names exactly the old release and commit.

A file that exists to test another release (the compatibility engine, a refusal of the old
release) is listed with --keep and left as it is. The rewrite is textual and claims nothing
(PROCESS.md §9); it is re-run after each rebase.

Usage: python3 -I tools/reversion/reversion.py <old> <new> [--header <old commit> <new commit>]
           [--keep <path>]... [--write] <file>...
"""
import argparse
import re
import sys


def patterns(old):
    o = re.escape(old)
    return [
        re.compile(r'(\bengine_version:\s*)"' + o + r'"'),
        re.compile(r'(\bengine_version:\s*)' + o + r'\b'),
        re.compile(r'(\bengine:\s*\{\s*kind:\s*vllm,\s*version:\s*\\?")' + o + r'(\\?")'),
        re.compile(r'(\bUpstreamVersion:\s*)"' + o + r'"'),
        re.compile(r'(\bEngineVersion:\s*)"' + o + r'"'),
        re.compile(r'(\(t,\s*)"' + o + r'"'),
    ]


def rewrite(text, old, new):
    n = 0
    for i, p in enumerate(patterns(old)):
        if i == 1:
            text, k = p.subn(lambda m: m.group(1) + new, text)
        elif i == 2:
            text, k = p.subn(lambda m: m.group(1) + new + m.group(2), text)
        elif i == 0 or i >= 3:
            text, k = p.subn(lambda m: m.group(1) + '"' + new + '"', text)
        n += k
    return text, n


def rewrite_header(text, old, new, old_commit, new_commit):
    """The provenance header of a ported Go file, in its first comment block, moved to the new
    release; other lines are left as they are."""
    lines = text.split("\n")
    want = re.compile(r"^(// Derived from vllm/\S+ at vLLM v)" + re.escape(old) + r"( \()" + re.escape(old_commit) + r"(\)[.:,]?)$")
    n = 0
    for i, line in enumerate(lines):
        if not line.startswith("//"):
            break
        m = want.match(line)
        if m:
            lines[i] = m.group(1) + new + m.group(2) + new_commit + m.group(3)
            n += 1
    return "\n".join(lines), n


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--header", nargs=2, metavar=("OLD_COMMIT", "NEW_COMMIT"))
    a = ap.parse_intermixed_args(argv)
    total, changed = 0, 0
    for f in a.files:
        if f in a.keep:
            continue
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        out, n = rewrite(text, a.old, a.new)
        if a.header and f.endswith(".go"):
            out, h = rewrite_header(out, a.old, a.new, *a.header)
            n += h
        if n:
            total += n
            changed += 1
            if a.write:
                with open(f, "w", encoding="utf-8") as fh:
                    fh.write(out)
    print("rewrote %d statements in %d files" % (total, changed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
