# Original to blis-sglang.
"""The mutation sweep (T7): apply one deliberate one-line break at a time to a scratch copy of the
repository, run the tests that cover it, and fail if any break goes uncaught.

Two kinds of mutant:

- Listed mutants, in oracle/mutants.json: an id, what it guards (a row, a check or an invariant),
  a file, the exact old text (it must occur once), the new text, and the suite that must catch it.
  The oracle's own checks are covered this way (X18), as blis-vllm's oracle/mutate*.py cover its
  ported rows.
- Generated mutants, for every Go file ported from SGLang (a file whose header says "Derived from
  python/sglang/"): each relational, equality and logical operator outside comments and strings is
  flipped (< and <=, > and >=, == and !=, && and ||), one occurrence at a time, and the file's
  package tests must catch it. So every ported function is swept as it lands, with no list to keep.

A mutant that survives fails the sweep, unless oracle/mutants.json lists it in `known_gaps` (with
the open issue that closes the gap, X28) or, for a generated mutant, in `equivalent` (with why no
test can tell it from the original). A listed gap or equivalent that is now caught, or that no
longer exists, fails too, so the lists stay true; so does a listed mutant whose old text is not
found exactly once.

A package may also list mutants in <package>/testdata/pending_mutants.json, beside the tests whose
inputs it describes: `pending` entries (id, needs, stage, issue) are known gaps whose reason is the
stage and the issue that closes the gap, and `equivalent` entries (id, why) are equivalents. The
package's own test fails a pending entry whose input has become accepted; this sweep fails one that
a test now catches, so the lists only shrink.

Modes:
  full (the default): every mutant. Nightly, on release tags and on demand.
  --changed <base>: on a pull request. The listed mutants of each changed file, and every listed
    mutant whose suite's test file changed; the generated mutants of each ported Go function the
    range base..HEAD changes, and of each package whose tests changed.
  --list: print the mutants the mode selects, and run none.
  --shard I/N: of the mutants the mode selects, in their listed order, run the I-th of every N
    (I from 0). The N shards partition the selection, so CI runs them side by side.

Usage: python3 -I oracle/mutate.py [--changed <base>] [--list] [--only ID,...] [--shard I/N] [--root <repository>] <sglang checkout> <scratch dir>
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IGNORE = shutil.ignore_patterns(".git", "sglang-source", "oracle-model", "oracle-traces", "__pycache__")
# One mutant's suite gets this long. A mutant that does not terminate (a loop that never advances) is
# caught by the limit, as a failing test is. The suite runs in its own process group and the whole group
# is killed, so a hung `go test` and the test binary it started cannot outlive the mutant and hold the
# runner. The limit is far above a healthy suite (a few seconds); BLIS_MUTANT_TIMEOUT_S overrides it.
TIMEOUT_S = int(os.environ.get("BLIS_MUTANT_TIMEOUT_S", "240"))
PORTED = re.compile(r"^// Derived from python/sglang/", re.M)
FLIPS = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==", "&&": "||", "||": "&&"}
OPERATOR = re.compile(r"<<=|>>=|<<|>>|<-|:=|&&|\|\||==|!=|<=|>=|<|>")


def go_code_spans(line: str, in_block: bool):
    """The parts of one Go source line outside comments, strings and runes, as (start, text)
    pairs, and whether a /* comment is still open at the end of the line."""
    spans, i, start, n = [], 0, 0, len(line)
    if in_block:
        end = line.find("*/")
        if end < 0:
            return [], True
        i = start = end + 2
    while i < n:
        c = line[i]
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            spans.append((start, line[start:i]))
            end = line.find("*/", i + 2)
            if end < 0:
                return spans, True
            i = start = end + 2
            continue
        if c in "\"'`":
            spans.append((start, line[start:i]))
            j = i + 1
            while j < n and line[j] != c:
                j += 2 if line[j] == "\\" and c != "`" else 1
            i = start = j + 1
            continue
        i += 1
    spans.append((start, line[start:i]))
    return [(s, t) for s, t in spans if t], False


def go_functions(text: str):
    """(name, first line, last line) of each top-level func in a gofmt-formatted file, 1-based."""
    out, lines, cur = [], text.split("\n"), None
    for no, line in enumerate(lines, 1):
        m = re.match(r"func (?:\([^)]*\) )?(\w+)", line)
        if m and cur is None:
            cur = (m.group(1), no)
            if line.rstrip().endswith("}"):
                out.append((cur[0], no, no))
                cur = None
        elif cur and line == "}":
            out.append((cur[0], cur[1], no))
            cur = None
    return out


def generated_mutants(rel: str, text: str) -> list:
    """The generated mutants of one ported Go file."""
    funcs = go_functions(text)
    out, in_block = [], False
    lines = text.split("\n")
    for no, line in enumerate(lines, 1):
        spans, in_block = go_code_spans(line, in_block)
        fn = next((f for f, a, b in funcs if a <= no <= b), None)
        if fn is None:
            continue
        seen: dict = {}
        for start, chunk in spans:
            for m in OPERATOR.finditer(chunk):
                op = m.group(0)
                if op not in FLIPS:
                    continue
                k = seen[op] = seen.get(op, 0) + 1
                col = start + m.start()
                new_line = line[:col] + FLIPS[op] + line[col + len(op):]
                out.append({
                    "id": f"{rel}:{fn}:{op}->{FLIPS[op]}#{k}:{line.strip()}",
                    "guards": f"{fn} ({rel})", "file": rel, "line": no, "function": fn,
                    "old": line, "new": new_line,
                    "suite": "go", "package": "./" + (os.path.dirname(rel) or "."),
                })
    return out


def git(*args) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def ported_go_files(root: str) -> list:
    out = []
    for rel in git("ls-files", "*.go").split():
        if rel.endswith("_test.go"):
            continue
        with open(os.path.join(root, rel)) as f:
            if PORTED.search(f.read(2048)):
                out.append(rel)
    return out


def changed_lines(base: str) -> dict:
    """file -> set of line numbers at HEAD that base..HEAD added or changed."""
    out: dict = {}
    cur = None
    for line in git("diff", "--unified=0", f"{base}...HEAD").splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur:
            first, count = int(m.group(1)), int(m.group(2) or 1)
            out.setdefault(cur, set()).update(range(first, first + max(count, 1)))
    return out


def select(cfg: dict, generated: list, base) -> list:
    listed = [dict(m, kind="listed") for m in cfg["mutants"]]
    gen = [dict(m, kind="generated") for m in generated]
    if base is None:
        return listed + gen
    changed = changed_lines(base)
    suite_tests = {name: s.get("tests", []) for name, s in cfg["suites"].items()}
    test_changed = {name for name, tests in suite_tests.items() if any(t in changed for t in tests)}
    pick = [m for m in listed if m["file"] in changed or m["suite"] in test_changed]
    changed_test_dirs = {os.path.dirname(f) for f in changed if f.endswith("_test.go")}
    for m in gen:
        lines = changed.get(m["file"], set())
        if os.path.dirname(m["file"]) in changed_test_dirs:
            pick.append(m)
            continue
        with open(os.path.join(ROOT, m["file"])) as f:
            text = f.read()
        span = next(((a, b) for f, a, b in go_functions(text) if f == m["function"] and a <= m["line"] <= b), None)
        if span and any(span[0] <= n <= span[1] for n in lines):
            pick.append(m)
    return pick


def command(cfg: dict, m: dict, sglang: str) -> list:
    return [a.replace("{sglang}", sglang).replace("{package}", m.get("package", "./..."))
            for a in cfg["suites"][m["suite"]]["command"]]


def run_suite(cmd: list, cwd: str) -> bool:
    """Run one mutant's suite; True when it passes (the mutant survived). A suite that outlives TIMEOUT_S is
    killed with everything it started and counts as failing, so the mutant is caught."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return False
    return proc.returncode == 0


def apply(scratch: str, m: dict) -> str:
    """Write the mutant into scratch; return why it could not be, or ""."""
    path = os.path.join(scratch, m["file"])
    if not os.path.exists(path):
        return f"{m['file']} does not exist"
    with open(path) as f:
        text = f.read()
    if m["kind"] == "generated":
        lines = text.split("\n")
        if lines[m["line"] - 1] != m["old"]:
            return f"line {m['line']} of {m['file']} is not the line the mutant was made from"
        lines[m["line"] - 1] = m["new"]
        new = "\n".join(lines)
    else:
        n = text.count(m["old"])
        if n != 1:
            return f"its old text occurs {n} times in {m['file']}, not once"
        new = text.replace(m["old"], m["new"])
    with open(path, "w") as f:
        f.write(new)
    return ""


def config_problems(cfg: dict) -> list:
    bad, ids = [], set()
    for m in cfg.get("mutants", []):
        for k in ("id", "guards", "file", "old", "new", "suite"):
            if not isinstance(m.get(k), str) or (k != "new" and not m[k].strip()):
                bad.append(f"mutant {m.get('id')!r}: {k} must be a non-empty string")
        if m.get("id") in ids:
            bad.append(f"mutant {m['id']}: listed twice")
        ids.add(m.get("id"))
        if m.get("suite") not in cfg.get("suites", {}):
            bad.append(f"mutant {m.get('id')}: suite {m.get('suite')!r} is not in suites")
        if m.get("old") == m.get("new"):
            bad.append(f"mutant {m.get('id')}: old and new are the same text")
    for k, why in list(cfg.get("known_gaps", {}).items()):
        if not isinstance(why, str) or not re.search(r"#\d+", why):
            bad.append(f"known gap {k}: its reason cites the open issue that closes it (X28)")
    for k, why in cfg.get("equivalent", {}).items():
        if not isinstance(why, str) or not why.strip():
            bad.append(f"equivalent {k}: no reason")
    if "go" not in cfg.get("suites", {}):
        bad.append("suites: no go suite, which the generated mutants run")
    return bad


def package_lists(root: str, cfg: dict) -> None:
    """Add to cfg the known gaps and equivalents the packages list in testdata/pending_mutants.json."""
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "sglang-source", "oracle-model", "oracle-traces", "__pycache__"))
        if os.path.basename(dirpath) != "testdata" or "pending_mutants.json" not in files:
            continue
        with open(os.path.join(dirpath, "pending_mutants.json")) as f:
            doc = json.load(f)
        for e in doc.get("pending", []):
            cfg.setdefault("known_gaps", {})[e["id"]] = f"{e.get('stage', '')}: {e.get('issue', '')}"
        for e in doc.get("equivalent", []):
            cfg.setdefault("equivalent", {})[e["id"]] = e.get("why", "")


def main(argv: list) -> int:
    global ROOT
    base = only = None
    list_only = False
    shard = None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--changed":
            base = next(it)
        elif a == "--only":
            only = set(next(it).split(","))
        elif a == "--shard":
            i, _, n = next(it).partition("/")
            if not (i.isdigit() and n.isdigit() and 0 <= int(i) < int(n)):
                print("mutate: --shard wants I/N with 0 <= I < N", file=sys.stderr)
                return 2
            shard = (int(i), int(n))
        elif a == "--list":
            list_only = True
        elif a == "--root":
            ROOT = os.path.abspath(next(it))
        else:
            args.append(a)
    if len(args) != 2:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    sglang, scratch = os.path.abspath(args[0]), os.path.abspath(args[1])
    with open(os.path.join(ROOT, "oracle", "mutants.json")) as f:
        cfg = json.load(f)
    package_lists(ROOT, cfg)
    bad = config_problems(cfg)
    if bad:
        for b in bad:
            print(b)
        return 1
    generated = []
    for rel in ported_go_files(ROOT):
        with open(os.path.join(ROOT, rel)) as f:
            generated += generated_mutants(rel, f.read())
    gen_ids = {m["id"] for m in generated}
    unexpected = 0
    if base is None and only is None:
        for k in sorted(set(cfg.get("equivalent", {})) - gen_ids):
            print(f"STALE      {k}: listed as equivalent, but no such generated mutant exists; remove it")
            unexpected += 1
        listed_ids = {m["id"] for m in cfg["mutants"]}
        for k in sorted(set(cfg.get("known_gaps", {})) - gen_ids - listed_ids):
            print(f"STALE      {k}: listed as a known gap, but no such mutant exists; remove it")
            unexpected += 1
    picked = select(cfg, generated, base)
    if only:
        picked = [m for m in picked if m["id"] in only]
    if shard:
        picked = [m for k, m in enumerate(picked) if k % shard[1] == shard[0]]
    if list_only:
        for m in picked:
            print(f"{m['kind']:9} {m['id']}  ({m['guards']})")
        print(f"mutate: {len(picked)} mutant(s) selected")
        return 1 if unexpected else 0
    allowed = {**cfg.get("known_gaps", {}), **cfg.get("equivalent", {})}
    caught = 0
    for m in picked:
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(ROOT, scratch, ignore=IGNORE)
        why = apply(scratch, m)
        if why:
            print(f"BROKEN     {m['id']}: {why}")
            unexpected += 1
            continue
        survived = run_suite(command(cfg, m, sglang), scratch)  # a mutant that hangs the suite is caught by its limit
        if survived and m["id"] in allowed:
            print(f"SURVIVED   {m['id']} (listed: {allowed[m['id']]})")
        elif survived:
            print(f"SURVIVED   {m['id']}  guards {m['guards']}")
            unexpected += 1
        elif m["id"] in allowed:
            print(f"NOW CAUGHT {m['id']}: remove it from known_gaps or equivalent")
            unexpected += 1
        else:
            print(f"CAUGHT     {m['id']}")
            caught += 1
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    print(f"mutate: {len(picked)} mutant(s), {caught} caught, {unexpected} finding(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
