# Original to blis-vllm.
"""Move the repository's vLLM citations from one release to the next, checking each one.

A citation is `<path>:<lines>@<commit8>` (AGENTS.md: every statement about vLLM cites
vllm/<path>:<lines>@<commit8> at the release Upstream names). Many are shortened inside a
paragraph: a basename (`kv_cache_utils.py:2173-2174`), a bare `:1799-1822` that continues the
last path, or line numbers with no tag that belong to a paragraph whose other citations carry
it. So the unit is the paragraph: a run of consecutive non-blank lines. A paragraph that carries
`@<old8>` and no other commit tag is a paragraph of old-release citations, and every
`<path>:<lines>` in it is one.

Each citation is mapped to the new release and verified:
  - the file is found at the new commit, at the same path, or, when the cited code moved, in the
    file that now defines the enclosing Python symbol (MOVED below lists where to look);
  - its lines are mapped through a line diff of the old and new file (difflib, no junk
    heuristic), or of the moved symbol's body;
  - every cited line's text is the same at the new lines (trailing space aside), and for a Python
    file the innermost enclosing def or class at the first cited line has the same qualified
    name at both releases (the cited symbol is still at the new line).
A paragraph is rewritten only if every citation in it verifies: its line numbers, paths and tag
change together, so a paragraph never mixes releases. Anything else (a cited line that changed,
a symbol that moved where MOVED does not look, a basename that names two files) is left as it
is and listed for a person to re-read; the old citation stays true of the old release until
then.

Usage:
    python3 -I tools/citations/recite.py <vllm git checkout> <old commit> <new commit> \\
        [--skip <path-prefix>]... [--write] [--report <file.tsv>] <file>...

Give it the files whose citations are statements about the code: Go, Python, documentation and
the registry. Leave out what a run produced at the old release, since its citations are that
run's evidence and move only when the run is repeated at the new one: the oracle's scenarios and
goldens (oracle/scenarios*/, */testdata/; a scenario's bytes are recorded in its golden), the
inertness proofs (oracle/inertness/*.json, oracle/inertness.py --write), dated reports
(docs/evidence/), and generated files (fields.json, surface.json, docs/generated/,
docs/metrics-manifest.json), which are regenerated. Leave out, too, every path another change
has claimed (--skip).

Without --write it only reports. The report has one row per citation: file, line, the cited
path and lines, the outcome (moved, unchanged, review: <why>) and the new path and lines.
"""

import argparse
import ast
import difflib
import os
import re
import subprocess
import sys

# Where code that left a file went, old path -> files to search for the cited symbol.
# vllm/v1/core/kv_cache_planning.py was split out of kv_cache_utils.py between v0.32.0 and v0.33.0.
MOVED = {
    "vllm/v1/core/kv_cache_utils.py": ["vllm/v1/core/kv_cache_planning.py"],
}

SPEC = r"\d+(?:-\d+)?(?:,\s?\d+(?:-\d+)?)*"
PATH = r"(?:(?:vllm|tests|csrc|rust|requirements|benchmarks|examples|docs|tools)/)?[A-Za-z0-9_./-]*?[A-Za-z0-9_-]+\.(?:py|rs|cu|cuh|cpp|cc|h|hpp|txt|toml|yaml|json|md|sh)"
# <path>:<spec> or a bare :<spec>, optionally followed by @tag.
CITE = re.compile(r"(?<![\w/.-])(?P<path>" + PATH + r")?:(?P<spec>" + SPEC + r")(?![\d-])(?P<tag>@[0-9a-f]{8,40}|@v\d+\.\d+\.\d+)?")
# A spec continued after a comma with no colon: "vllm/x.py:1-2, 5-9@abc" is matched by SPEC
# itself; "(…, 2276-2297@5df63a72)" after a path elsewhere in the paragraph is a bare number list.
BARE_TAGGED = re.compile(r"(?<![\w:./-])(?P<spec>" + SPEC + r")(?P<tag>@[0-9a-f]{8,40})")
TAG = re.compile(r"@([0-9a-f]{8,40}|v\d+\.\d+\.\d+)\b")


class Tree:
    """Files of one vLLM commit, read through git."""

    def __init__(self, repo, commit):
        self.repo, self.commit = repo, commit
        out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", commit],
                             capture_output=True, text=True, check=True).stdout
        self.files = set(out.split())
        self.cache = {}

    def lines(self, path):
        if path not in self.cache:
            if path not in self.files:
                self.cache[path] = None
            else:
                r = subprocess.run(["git", "-C", self.repo, "show", "%s:%s" % (self.commit, path)],
                                   capture_output=True, check=True)
                self.cache[path] = r.stdout.decode("utf-8", "replace").splitlines()
        return self.cache[path]

    def resolve(self, name):
        """The one file whose path is name or ends with /name; None if none or several."""
        if name in self.files:
            return name
        hits = [f for f in self.files if f.endswith("/" + name)]
        return hits[0] if len(hits) == 1 else None


def symbols(lines):
    """For a Python file, a list giving each line's innermost enclosing def/class qualname."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None
    owner = [""] * (len(lines) + 2)

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + ch.name
                start = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                for i in range(start, (ch.end_lineno or ch.lineno) + 1):
                    owner[i] = q
                walk(ch, q + ".")
            else:
                walk(ch, prefix)

    walk(tree, "")
    return owner


def symbol_span(lines, qual):
    syms = symbols(lines)
    if syms is None or not qual:
        return None
    idx = [i for i, q in enumerate(syms) if q == qual]
    return (idx[0], idx[-1]) if idx else None


def line_map(a, b):
    """old line number -> new line number, for lines in equal blocks of a line diff."""
    m = {}
    sm = difflib.SequenceMatcher(None, [x.rstrip() for x in a], [x.rstrip() for x in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
    return m


def parse_spec(spec):
    out = []
    for part in re.split(r",\s?", spec):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def fmt_spec(ranges):
    return ", ".join("%d" % lo if lo == hi else "%d-%d" % (lo, hi) for lo, hi in ranges)


def map_ranges(old_lines, new_lines, ranges, lmap, old_syms, new_syms):
    """Map each range; None and a reason if any line is not carried over unchanged."""
    out = []
    for lo, hi in ranges:
        if lo < 1 or hi < lo or hi > len(old_lines):
            return None, "lines %d-%d are outside the old file (%d lines)" % (lo, hi, len(old_lines))
        new = [lmap.get(i) for i in range(lo, hi + 1)]
        if None in new:
            return None, "lines %d-%d changed" % (lo, hi)
        if new != list(range(new[0], new[0] + hi - lo + 1)):
            return None, "lines %d-%d are no longer contiguous" % (lo, hi)
        if old_syms is not None and new_syms is not None and old_syms[lo] != new_syms[new[0]]:
            return None, "line %d is in %s, and its new line %d in %s" % (lo, old_syms[lo] or "module scope", new[0], new_syms[new[0]] or "module scope")
        out.append((new[0], new[-1]))
    return out, None


def relocate(old, new, path, ranges):
    """Find cited lines whose code moved: the new path and ranges, or None and why."""
    old_lines = old.lines(path)
    syms = symbols(old_lines) if path.endswith(".py") else None
    candidates = ([path] if new.lines(path) is not None else []) + MOVED.get(path, [])
    if syms is None:
        return None, None, "not a Python file, so its moved code cannot be followed by symbol"
    if any(lo < 1 or hi > len(old_lines) for lo, hi in ranges):
        return None, None, "the cited lines are outside the old file (%d lines)" % len(old_lines)
    quals = {syms[lo] for lo, _ in ranges}
    if len(quals) != 1 or "" in quals:
        return None, None, "the cited lines are not inside one def or class"
    qual = quals.pop()
    o = symbol_span(old_lines, qual)
    for cand in candidates:
        nl = new.lines(cand)
        if nl is None:
            continue
        n = symbol_span(nl, qual)
        if n is None:
            continue
        sub = line_map(old_lines[o[0] - 1:o[1]], nl[n[0] - 1:n[1]])
        lmap = {k + o[0] - 1: v + n[0] - 1 for k, v in sub.items()}
        got, why = map_ranges(old_lines, nl, ranges, lmap, syms, symbols(nl))
        if got is not None:
            return cand, got, None
        return None, None, "%s moved to %s, and %s" % (qual, cand, why)
    return None, None, "%s is not defined in %s at the new commit" % (qual, " or ".join(candidates) or "any file MOVED names")


def paragraphs(text):
    """(first line number, lines) for each run of consecutive non-blank lines."""
    out, cur, start = [], [], 0
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip():
            if not cur:
                start = i
            cur.append(line)
        elif cur:
            out.append((start, cur))
            cur = []
    if cur:
        out.append((start, cur))
    return out


FULL = re.compile(r"(?<![\w/.-])((?:vllm|tests|csrc|rust)/[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.(?:py|rs|cu|cuh|cpp|cc|h|hpp|txt|toml))")
DERIVED = re.compile(r"Derived from (vllm/\S+?)(?:,| at )")


def context_path(raw, para_paths, header, file_paths, old, all_paths=()):
    """A shortened path (a basename, or a path tail) resolved from its context: the last full path
    in the paragraph that ends with it, then the file's Derived-from header, then the nearest full
    path before it in the file, then the one file in the old tree that ends with it. When the file
    names two different full paths that end with it, it is ambiguous, and None."""
    named = {p for p in all_paths if p == raw or p.endswith("/" + raw)}
    if len(named) > 1:
        return None
    for group in (reversed(para_paths), header, reversed(file_paths)):
        for p in group:
            if p == raw or p.endswith("/" + raw):
                return p if p in old.files else None
    return old.resolve(raw)


def process(fname, text, old, new, old8, new8, report):
    """The file's text with every verified paragraph rewritten, and how many were."""
    lines = text.split("\n")
    changed = 0
    header = DERIVED.findall("\n".join(lines[:12]))
    offset_paths = [(m.start(), m.group(1)) for m in FULL.finditer(text)]
    all_paths = set(header) | {p for _, p in offset_paths}
    line_offsets = [0]
    for ln in lines:
        line_offsets.append(line_offsets[-1] + len(ln) + 1)
    for start, para in paragraphs(text):
        joined = "\n".join(para)
        tags = {t for t in TAG.findall(joined)}
        if not any(t.startswith(old8) or old8.startswith(t) for t in tags if not t.startswith("v")):
            continue
        if any(not (t.startswith(old8) or old8.startswith(t)) for t in tags):
            for t in sorted(tags):
                report.append((fname, start, "", "", "review: the paragraph also cites @" + t, ""))
            continue
        before = [p for o, p in offset_paths if o < line_offsets[start - 1]]
        edits, ok, last, last_new, para_paths = [], True, None, None, []
        rows = []
        for k, line in enumerate(para):
            for m in list(CITE.finditer(line)) + [mm for mm in BARE_TAGGED.finditer(line)
                                                   if not any(c.start() <= mm.start() < c.end() for c in CITE.finditer(line))]:
                raw = m.groupdict().get("path")
                if raw:
                    path = raw if raw.startswith(("vllm/", "tests/", "csrc/", "rust/")) else \
                        context_path(raw, para_paths, header, before, old, all_paths)
                    if path is None:
                        if os.path.splitext(raw)[1] in (".go",) or "/" not in raw and old.resolve(raw) is None and not any(f.endswith(raw) for f in old.files):
                            continue  # not a vLLM file (a Go file of this repository, a testdata name)
                        rows.append((fname, start + k, raw, m.group("spec"), "review: %s names no single file at the old commit, or the file names two that end with it" % raw, ""))
                        ok = False
                        continue
                    last = path
                    para_paths.append(path)
                else:
                    path = last or (header[0] if len(header) == 1 else None) or (before[-1] if before else None)
                    if path is None:
                        rows.append((fname, start + k, "", m.group("spec"), "review: a bare line number with no path before it", ""))
                        ok = False
                        continue
                if old.lines(path) is None:
                    rows.append((fname, start + k, path, m.group("spec"), "review: not a file at the old commit", ""))
                    ok = False
                    continue
                ranges = parse_spec(m.group("spec"))
                ol, nl = old.lines(path), new.lines(path)
                got, why, newpath = None, "the file is gone", path
                if nl is not None:
                    got, why = map_ranges(ol, nl, ranges, line_map(ol, nl),
                                          symbols(ol) if path.endswith(".py") else None,
                                          symbols(nl) if path.endswith(".py") else None)
                if got is None:
                    p2, g2, w2 = relocate(old, new, path, ranges)
                    if g2 is not None:
                        newpath, got = p2, g2
                    else:
                        why = w2 if nl is None or "not inside" not in (w2 or "") else why
                if got is None:
                    rows.append((fname, start + k, path, m.group("spec"), "review: " + why, ""))
                    ok = False
                    continue
                if not raw and last_new is not None and newpath != last_new:
                    # A bare citation reads its path from the one before it, which now names
                    # another file than the one this code moved to.
                    rows.append((fname, start + k, path, m.group("spec"),
                                 "review: a bare citation moved to %s, and the path it inherits to %s" % (newpath, last_new), ""))
                    ok = False
                    continue
                if raw:
                    last_new = newpath
                outcome = "unchanged" if got == ranges and newpath == path else "moved"
                rows.append((fname, start + k, path, m.group("spec"), outcome, "%s:%s" % (newpath, fmt_spec(got))))
                new_raw = raw
                if raw and newpath != path:
                    new_raw = newpath if raw == path else os.path.basename(newpath) if raw == os.path.basename(path) else newpath
                edits.append((k, m.start(), m.end(), (new_raw or "") + ("" if m.re is BARE_TAGGED else ":") + fmt_spec(got) + ("@" + new8 if m.group("tag") else "")))
        report.extend(rows)
        if not ok or not edits:
            continue
        for k, s, e, rep in sorted(edits, key=lambda x: (x[0], -x[1])):
            i = start - 1 + k
            lines[i] = lines[i][:s] + rep + lines[i][e:]
        # Any remaining old tag in the paragraph (a file-level "path@old8") moves too.
        for k in range(len(para)):
            i = start - 1 + k
            lines[i] = re.sub(r"@" + old8 + r"[0-9a-f]*\b", "@" + new8, lines[i])
        changed += 1
    return "\n".join(lines), changed


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vllm")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--skip", action="append", default=[], help="a path prefix another change holds; left as it is")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_intermixed_args(argv)
    old, new = Tree(a.vllm, a.old), Tree(a.vllm, a.new)
    old8, new8 = a.old[:8], a.new[:8]
    report, total = [], 0
    for f in a.files:
        if any(f == p or f.startswith(p.rstrip("/") + "/") or f.startswith(p) and p.endswith("/") for p in a.skip):
            report.append((f, 0, "", "", "skipped: held by another change", ""))
            continue
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        if "@" + old8 not in text:
            continue
        out, n = process(f, text, old, new, old8, new8, report)
        total += n
        if a.write and out != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(out)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write("file\tline\tcited\tlines\toutcome\tnew\n")
            for r in report:
                fh.write("\t".join(str(x) for x in r) + "\n")
    kinds = {}
    for r in report:
        kinds[r[4].split(":")[0]] = kinds.get(r[4].split(":")[0], 0) + 1
    print("paragraphs rewritten: %d; citations: %s" % (total, ", ".join("%s %d" % kv for kv in sorted(kinds.items()))), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
