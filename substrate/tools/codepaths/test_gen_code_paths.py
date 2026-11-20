# Original to blis-substrate.
"""Tests for gen_code_paths.py: what a function's hash follows, and what the generator refuses."""

import ast
import json
import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_code_paths as gen  # noqa: E402


def h(src, name="f"):
    tree = ast.parse(textwrap.dedent(src))
    return gen.function_hash(gen.find_function(tree, name, "test"))


BASE = '''\
def f(self, x, y=2):
    """Does a thing."""
    total = x + y
    for i in range(x):
        total += i
    return total
'''


def test_hash_ignores_what_is_not_the_code():
    base = h(BASE)
    same = {
        "a comment": BASE.replace("total = x + y", "total = x + y  # the sum"),
        "formatting": BASE.replace("total = x + y", "total   =   (x +\n   y)"),
        "a docstring": BASE.replace('"""Does a thing."""', '"""Does another thing."""'),
        "no docstring": BASE.replace('"""Does a thing."""', ""),
        "a parameter annotation": BASE.replace("(self, x, y=2)", "(self, x: int, y: int = 2)"),
        "a return annotation": BASE.replace("y=2):", "y=2) -> int:"),
        "an annotated assignment": BASE.replace("total = x + y", "total: int = x + y"),
        "a declaration without a value": BASE.replace("total = x + y", "seen: bool\n    total = x + y"),
        "a logging statement": BASE.replace("total = x + y", "logger.info('adding %s', x)\n    total = x + y"),
        "a logging statement on self": BASE.replace("total = x + y", "self.logger.debug('adding')\n    total = x + y"),
        "a once-only logging statement": BASE.replace("total = x + y", "logger.warning_once('adding')\n    total = x + y"),
    }
    for name, src in same.items():
        assert h(src) == base, f"{name} changed the hash"


def test_hash_follows_the_code():
    base = h(BASE)
    different = {
        "a constant": BASE.replace("y=2", "y=3"),
        "an operator": BASE.replace("x + y", "x - y"),
        "a statement": BASE.replace("total += i", "total += 2 * i"),
        "a call": BASE.replace("return total", "return total + helper(total)"),
        "a name": BASE.replace("total", "tot"),
        "a decorator": "@cache\n" + BASE,
        "a raised error": BASE.replace("return total", "if total < 0:\n        raise ValueError()\n    return total"),
        "a call that is not logging": BASE.replace("total = x + y", "metrics.info('adding')\n    total = x + y"),
        "a logger used as a value": BASE.replace("total = x + y", "total = logger.info('adding')"),
        "a branch of an if": "def f(x):\n    if x:\n        return 1\n    return 2\n",
    }
    for name, src in different.items():
        assert h(src, "f") != base, f"{name} did not change the hash"
    assert h("def f(x):\n    if x:\n        return 1\n    return 2\n") != h("def f(x):\n    if x:\n        return 1\n    return 3\n")


def test_hash_ignores_logging_inside_blocks_but_not_other_calls_there():
    plain = "def f(x):\n    if x:\n        y = 1\n    return y\n"
    logged = "def f(x):\n    if x:\n        logger.info('x')\n        y = 1\n    return y\n"
    called = "def f(x):\n    if x:\n        notify('x')\n        y = 1\n    return y\n"
    assert h(plain) == h(logged)
    assert h(plain) != h(called)


def test_nested_names_and_refusals():
    src = textwrap.dedent('''
        class A:
            def m(self):
                return 1
            class B:
                def m(self):
                    return 2
        def twice(): return 1
        def twice(): return 2
        class NotAFunction: pass
        from typing import overload
        @overload
        def over(x: int) -> int: ...
        @overload
        def over(x: str) -> str: ...
        def over(x): return x
    ''')
    tree = ast.parse(src)
    a = gen.function_hash(gen.find_function(tree, "A.m", "t"))
    b = gen.function_hash(gen.find_function(tree, "A.B.m", "t"))
    assert a != b
    assert gen.find_function(tree, "over", "t").name == "over"
    for name in ("missing", "A.missing", "twice", "NotAFunction", "A"):
        with pytest.raises(gen.Refused):
            gen.find_function(tree, name, "t")


def test_code_id_rule():
    fs = [{"id": "b", "hash": "2" * 64}, {"id": "a", "hash": "1" * 64}]
    expect = gen.hashlib.sha256(
        ("blis code path hash v1\npath host\nfunction a " + "1" * 64 + "\nfunction b " + "2" * 64 + "\n").encode()
    ).hexdigest()
    assert gen.code_id("host", fs) == expect
    assert gen.code_id("graphs", fs) != expect


# ------------------------------------------------------------------------------------------------
# generate(): a checkout of a tiny release.


def git(root, *args):
    subprocess.run(["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


@pytest.fixture
def release(tmp_path):
    root = str(tmp_path / "engine")
    os.makedirs(os.path.join(root, "pkg"))
    with open(os.path.join(root, "pkg", "mod.py"), "w") as f:
        f.write("class S:\n    def admit(self, r):\n        return r + 1\n\ndef launch(g):\n    return g\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "release")
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    spec = {"engine": {"kind": "toy", "version": "1.0", "commit": commit},
            "paths": {"host": [{"id": "pkg/mod.py::S.admit", "why": "admission"}],
                      "graphs": [{"id": "pkg/mod.py::launch", "why": "launch"}]}}
    return root, spec, commit


def test_generate_states_each_path(release):
    root, spec, commit = release
    table = gen.generate(root, spec)
    assert [p["path"] for p in table["paths"]] == ["host", "graphs"]
    host = table["paths"][0]
    assert host["functions"][0]["source"] == f"pkg/mod.py:2@{commit[:8]}"
    assert host["code_id"] == gen.code_id("host", host["functions"])
    assert table["paths"][1]["code_id"] != host["code_id"]
    assert gen.render(table) == gen.render(gen.generate(root, spec)), "generation is not deterministic"


def test_generate_refuses_another_commit_or_a_missing_function(release):
    root, spec, _ = release
    bad = json.loads(json.dumps(spec))
    bad["engine"]["commit"] = "0" * 40
    with pytest.raises(gen.Refused, match="not the spec's release"):
        gen.generate(root, bad)
    for fid in ("pkg/mod.py::S.gone", "pkg/gone.py::x", "no-separator"):
        bad = json.loads(json.dumps(spec))
        bad["paths"]["host"] = [{"id": fid}]
        with pytest.raises(gen.Refused):
            gen.generate(root, bad)
    bad = json.loads(json.dumps(spec))
    bad["paths"]["host"] = spec["paths"]["host"] * 2
    with pytest.raises(gen.Refused, match="twice"):
        gen.generate(root, bad)


def test_a_change_to_a_function_changes_its_path_only(release):
    root, spec, _ = release
    before = gen.generate(root, spec)
    path = os.path.join(root, "pkg", "mod.py")
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(text.replace("r + 1", "r + 2"))
    git(root, "commit", "-qam", "change")
    spec2 = json.loads(json.dumps(spec))
    spec2["engine"]["commit"] = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    after = gen.generate(root, spec2)
    assert after["paths"][0]["code_id"] != before["paths"][0]["code_id"]
    assert after["paths"][1]["code_id"] == before["paths"][1]["code_id"]
