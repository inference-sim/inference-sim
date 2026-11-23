# Original to blis-sglang.
"""Fail unless every SGLang refusal the port quotes is where oracle/refusal_quotes.json says it is
in SGLang's source at the pinned commit, and the port raises and tests the same words (SGL-6, SIM-9).

For each quote: the string literals SGLang's code has on the cited lines (adjacent literals joined,
as Python joins them) contain the message; the code lines contain the statements the quote
depends on; the port's raising code and its test contain the message's fragments verbatim. A
message that changed upstream, moved out of the cited lines, or was reworded in the port fails.

The scan is static (ast): SGLang is read, not imported. Standard library only.

Usage: python3 -I oracle/check_refusal_quotes.py <sglang checkout> [<refusal_quotes.json>]
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
LIST = os.path.join(HERE, "refusal_quotes.json")


def render(node) -> str:
    """A string literal as its text; an f-string keeps each replacement field as {expression}."""
    if isinstance(node, ast.Constant):
        return node.value
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(part.value)
        else:
            conv = {115: "!s", 114: "!r", 97: "!a"}.get(part.conversion, "")
            out.append("{" + ast.unparse(part.value) + conv + "}")
    return "".join(out)


def literals(src: str, first: int, last: int) -> list:
    """The string literals (adjacent ones are one, as Python joins them) that start on lines
    first..last, outermost only."""
    tree = ast.parse(src)
    inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            inner.update(id(n) for n in ast.walk(node) if n is not node)
    out = []
    for node in ast.walk(tree):
        if id(node) in inner:
            continue
        is_str = isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))
        if is_str and first <= node.lineno <= last:
            out.append(render(node))
    return out


def check(root: str, quotes: dict, engine: str = ENGINE) -> list:
    errs = []
    srt = os.path.join(root, "python", "sglang", "srt")
    for q in quotes["quotes"]:
        name = q["name"]
        a, b = q["lines"]
        path = os.path.join(srt, q["file"])
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except OSError as e:
            errs.append(f"{name}: {e}")
            continue
        lines = src.splitlines()
        window = "\n".join(lines[a - 1:b])
        found = literals(src, a, b)
        for lit in q["source_literals"]:
            if lit not in found:
                errs.append(f"{name}: {q['file']}:{a}-{b} has no string literal {lit!r}")
        for code in q["source_code"]:
            if code not in window:
                errs.append(f"{name}: {q['file']}:{a}-{b} has no line with {code!r}")
        for side in ("go", "test"):
            try:
                with open(os.path.join(engine, q[side]["file"]), encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as e:
                errs.append(f"{name}: {e}")
                continue
            for frag in q[side]["fragments"]:
                if frag not in text:
                    errs.append(f"{name}: {q[side]['file']} does not contain {frag!r}")
    return errs


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    quotes = json.load(open(argv[2] if len(argv) > 2 else LIST, encoding="utf-8"))
    errs = check(argv[1], quotes)
    for e in errs:
        print("FAIL:", e)
    if errs:
        return 1
    print(f"{len(quotes['quotes'])} refusal quotes match SGLang's source at {quotes['release']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
