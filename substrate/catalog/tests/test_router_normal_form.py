"""Whether a router build's deciding files are the same at every commit it could be (scripts/router_normal_form.py).

The normal form hides logging and nothing else: two versions of a function that differ only in logging
have one digest, and a change to what the function computes has another.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import router_normal_form as nf  # noqa: E402

BEFORE = """\
pub async fn round_robin(&self, request: SingleIn<T>) -> Result {
    let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;

    let instance_id = {
        let routing_instances = self.client.routing_instances();
        let count = routing_instances.free_ids().len();
        if count == 0 {
            return Err(self.empty_free_pool_error(&routing_instances));
        }
        routing_instances.free_ids()[counter % count]
    };
    tracing::trace!("round robin router selected {instance_id}");

    self.generate_with_fault_detection(instance_id, request).await
}

fn pick(ids: &[u64]) -> u64 {
    if ids.len() == 1 {
        return ids[0];
    }
    ids[0]
}
"""

# The same functions with structured logging added, as Dynamo's 14 June 2026 commit did.
LOGGING_ONLY = """\
pub async fn round_robin(&self, request: SingleIn<T>) -> Result {
    let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;

    let (instance_id, candidate_count) = {
        let routing_instances = self.client.routing_instances();
        let count = routing_instances.free_ids().len();
        if count == 0 {
            return Err(self.empty_free_pool_error(&routing_instances));
        }
        (routing_instances.free_ids()[counter % count], count)
    };
    tracing::info!(
        router_mode = "round-robin",
        worker_id = instance_id,
        candidate_count,
        "Selected worker"
    );

    self.generate_with_fault_detection(instance_id, request).await
}

fn pick(ids: &[u64]) -> u64 {
    if ids.len() == 1 {
        let worker_id = ids[0];
        tracing::info!(worker_id, "Selected worker");
        return worker_id;
    }
    ids[0]
}
"""


def d(text: str, path: str = "x.rs") -> str:
    return nf.digest(path, text)


def test_logging_only_rewrite_has_the_same_normal_form():
    assert d(BEFORE) == d(LOGGING_ONLY)


@pytest.mark.parametrize("old,new", [
    ("routing_instances.free_ids()[counter % count]\n    };", "routing_instances.free_ids()[(counter + 1) % count]\n    };"),
    ("fetch_add(1,", "fetch_add(2,"),
    ("if count == 0", "if count <= 1"),
])
def test_a_change_to_what_the_function_computes_changes_the_normal_form(old, new):
    changed = LOGGING_ONLY.replace(old.replace("\n    };", ""), new.replace("\n    };", ""))
    assert changed != LOGGING_ONLY
    assert d(changed) != d(BEFORE)


def test_removing_logging_does_not_hide_a_changed_selection_inside_it():
    # The selection index is on the tuple's first element: changing it is a change.
    changed = LOGGING_ONLY.replace("free_ids()[counter % count], count)", "free_ids()[0], count)")
    assert d(changed) != d(BEFORE)


def test_a_tuple_element_that_is_read_is_kept():
    src = LOGGING_ONLY + "\nfn f() {\n    let (a, n) = {\n        (1, 2)\n    };\n    use_it(n);\n}\n"
    base = LOGGING_ONLY + "\nfn f() {\n    let a = {\n        1\n    };\n    use_it(n);\n}\n"
    assert d(src) != d(base)


def test_a_binding_read_after_the_return_site_is_kept():
    src = "fn f() -> u64 {\n    if c {\n        let w = ids[0];\n        return w;\n    }\n    ids[1]\n}\n"
    bare = "fn f() -> u64 {\n    if c {\n        return ids[0];\n    }\n    ids[1]\n}\n"
    assert d(src) == d(bare)  # w is read nowhere else: inlined
    used = "fn f() -> u64 {\n    let w = ids[0];\n    if c {\n        return w;\n    }\n    w\n}\n"
    usedbare = "fn f() -> u64 {\n    if c {\n        return ids[0];\n    }\n    ids[0]\n}\n"
    assert d(used) != d(usedbare)


def test_only_rust_and_python_files_are_rewritten():
    txt = "x = 1\n\ny = 2\n"
    assert d(txt, "a.txt") != d(txt.replace("\n\n", "\n"), "a.txt")


def test_a_logging_macro_that_is_not_a_statement_is_not_removed():
    assert d("fn f() { let v = info!(1); }\n") != d("fn f() { let v = ; }\n")


def git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "GIT_CONFIG_GLOBAL": "/dev/null"}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def commit(repo: Path, files: dict[str, str], date: str) -> str:
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c", date=date)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path):
    r = tmp_path / "r"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    return r


def test_a_nightly_window_is_the_commits_of_its_utc_day_and_the_head_before_it(repo):
    a = commit(repo, {"f.rs": BEFORE}, "2026-06-12T23:00:00Z")   # head at the start of 13 June
    b = commit(repo, {"f.rs": BEFORE + "\n"}, "2026-06-13T09:00:00Z")
    c = commit(repo, {"f.rs": BEFORE + "\n\n"}, "2026-06-13T20:00:00Z")
    commit(repo, {"f.rs": BEFORE + "\n\n\n"}, "2026-06-14T01:00:00Z")  # after the day
    assert nf.window(str(repo / ".git"), "1.0.0.dev20260613") == [c, b, a]


def test_a_release_is_the_commit_of_its_tag(repo):
    a = commit(repo, {"f.rs": BEFORE}, "2026-06-12T23:00:00Z")
    git(repo, "tag", "v1.2.1")
    commit(repo, {"f.rs": BEFORE + "// later\n"}, "2026-06-13T09:00:00Z")
    assert nf.window(str(repo / ".git"), "1.2.1") == [a]


def test_a_window_is_equivalent_when_only_logging_changed_and_differs_when_behaviour_changed(repo):
    a = commit(repo, {"f.rs": BEFORE, "g.py": "router_mode = 1\n"}, "2026-06-12T23:00:00Z")
    b = commit(repo, {"f.rs": LOGGING_ONLY, "g.py": "router_mode = 1\n"}, "2026-06-13T09:00:00Z")
    g = str(repo / ".git")
    r = nf.check(g, [b, a], ["f.rs", "g.py"])
    assert r["f.rs"]["verdict"] == "equivalent" and len(r["f.rs"]["blobs"]) == 2
    assert r["g.py"]["verdict"] == "identical"
    c = commit(repo, {"f.rs": LOGGING_ONLY.replace("counter % count", "(counter + 1) % count")}, "2026-06-13T10:00:00Z")
    assert nf.check(g, [c, b, a], ["f.rs"])["f.rs"]["verdict"] == "differs"
    d2 = commit(repo, {"g.py": "router_mode = 2\n"}, "2026-06-13T11:00:00Z")
    assert nf.check(g, [d2, b], ["g.py"])["g.py"]["verdict"] == "differs"  # a Python statement that names a router
    d3 = commit(repo, {"g.py": "router_mode = 2\nx = 3\n"}, "2026-06-13T12:00:00Z")
    assert nf.check(g, [d3, d2], ["g.py"])["g.py"]["verdict"] == "equivalent"  # one that names none


def test_a_file_missing_at_a_commit_is_a_difference(repo):
    a = commit(repo, {"f.rs": BEFORE}, "2026-06-12T23:00:00Z")
    b = commit(repo, {"h.rs": "fn h() {}\n"}, "2026-06-13T09:00:00Z")
    git(repo, "rm", "-q", "f.rs")
    git(repo, "commit", "-q", "-m", "rm", date="2026-06-13T10:00:00Z")
    c = git(repo, "rev-parse", "HEAD")
    assert nf.check(str(repo / ".git"), [c, a], ["f.rs"])["f.rs"]["verdict"] == "differs"


PY_BEFORE = '''\
import vllm.platforms
from dynamo.llm import KvRouterConfig, RouterMode

def parse_args():
    if chat_processor == "vllm":
        import vllm.platforms
        vllm.platforms.current_platform = CpuPlatform()
    args.router_mode = "round-robin"

async def async_main(config):
    if config.router_mode == "random":
        router_mode = RouterMode.Random
    else:
        router_mode = RouterMode.RoundRobin
'''


def _py(text: str) -> str:
    return nf.digest("frontend/main.py", text)


def test_python_code_that_names_no_router_does_not_change_the_digest():
    after = PY_BEFORE.replace('        vllm.platforms.current_platform = CpuPlatform()\n', "").replace(
        "from dynamo.llm import KvRouterConfig, RouterMode", "from dynamo.llm import FrontendRoute, KvRouterConfig, RouterMode")
    after += "\ndef serve_extra_routes():\n    return 1\n"
    assert _py(after) == _py(PY_BEFORE)


def test_a_changed_router_mode_changes_the_digest():
    for changed in (PY_BEFORE.replace("RouterMode.RoundRobin", "RouterMode.Random"),
                    PY_BEFORE.replace('config.router_mode == "random"', 'config.router_mode == "kv"'),
                    PY_BEFORE.replace('args.router_mode = "round-robin"', 'args.router_mode = "kv"')):
        assert _py(changed) != _py(PY_BEFORE)
