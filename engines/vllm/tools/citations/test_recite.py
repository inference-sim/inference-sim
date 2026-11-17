# Original to blis-vllm.
"""Tests for recite.py over a two-commit synthetic upstream. Run: python3 -I tools/citations/test_recite.py"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recite  # noqa: E402

OLD = {
    "vllm/other/sched.py": """\
        def a():
            return 1


        def b():
            x = 1
            return x
    """,
    "vllm/core/sched.py": """\
        import os


        def schedule(reqs):
            budget = 8
            for r in reqs:
                budget -= r
            return budget


        def preempt(r):
            return r - 1
    """,
    "vllm/core/utils.py": """\
        def hash_block(tokens):
            return hash(tuple(tokens))


        def group_layers(specs):
            groups = []
            for s in specs:
                groups.append([s])
            return groups
    """,
}
# The new release adds four lines above schedule, changes preempt's body, and moves
# group_layers into a new module, below one new function.
NEW = {
    "vllm/other/sched.py": """\
        def a():
            return 1


        def b():
            x = 1
            return x
    """,
    "vllm/core/sched.py": """\
        import os

        # A new comment.
        # And another.
        LIMIT = 3


        def schedule(reqs):
            budget = 8
            for r in reqs:
                budget -= r
            return budget


        def preempt(r):
            return r - 2
    """,
    "vllm/core/utils.py": """\
        def hash_block(tokens):
            return hash(tuple(tokens))
    """,
    "vllm/core/planning.py": """\
        def plan():
            return None


        def group_layers(specs):
            groups = []
            for s in specs:
                groups.append([s])
            return groups
    """,
}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, files):
    for path, body in files.items():
        p = os.path.join(repo, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(body))
    for path in set(git(repo, "ls-files").split()) - set(files):
        os.remove(os.path.join(repo, path))
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-qm", "c")
    return git(repo, "rev-parse", "HEAD")


class ReciteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = cls.tmp.name
        git(cls.repo, "init", "-q")
        cls.old = commit(cls.repo, OLD)
        cls.new = commit(cls.repo, NEW)
        recite.MOVED["vllm/core/utils.py"] = ["vllm/core/planning.py"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_on(self, text):
        old, new = recite.Tree(self.repo, self.old), recite.Tree(self.repo, self.new)
        report = []
        out, n = recite.process("x.go", text, old, new, self.old[:8], self.new[:8], report)
        return out, n, report

    def test_a_citation_whose_lines_shifted_is_moved_to_the_new_lines(self):
        out, n, _ = self.run_on("// schedule's loop (vllm/core/sched.py:6-7@%s).\n" % self.old[:8])
        self.assertEqual(n, 1)
        self.assertEqual(out, "// schedule's loop (vllm/core/sched.py:10-11@%s).\n" % self.new[:8])

    def test_a_citation_of_a_changed_line_is_left_for_review(self):
        text = "// preempt (vllm/core/sched.py:12@%s).\n" % self.old[:8]
        out, n, report = self.run_on(text)
        self.assertEqual((out, n), (text, 0))
        self.assertTrue(report[0][4].startswith("review:"), report)

    def test_one_unverified_citation_keeps_its_whole_paragraph_on_the_old_release(self):
        text = ("// schedule (vllm/core/sched.py:6-7@%s), and\n// preempt (sched.py:12).\n" % self.old[:8])
        out, n, _ = self.run_on(text)
        self.assertEqual((out, n), (text, 0))

    def test_moved_code_is_followed_by_its_symbol_into_the_new_module(self):
        out, n, _ = self.run_on("// groups (vllm/core/utils.py:7-8@%s).\n" % self.old[:8])
        self.assertEqual(n, 1)
        self.assertEqual(out, "// groups (vllm/core/planning.py:7-8@%s).\n" % self.new[:8])

    def test_a_basename_and_a_bare_continuation_take_their_path_from_the_paragraph(self):
        text = ("// See vllm/core/sched.py:4@%s, then the loop (sched.py:6-7)\n// and the return (:8).\n" % self.old[:8])
        out, n, _ = self.run_on(text)
        self.assertEqual(n, 1)
        self.assertEqual(out, "// See vllm/core/sched.py:8@%s, then the loop (sched.py:10-11)\n// and the return (:12).\n" % self.new[:8])

    def test_a_bare_citation_whose_code_moved_elsewhere_than_its_inherited_path_is_left_for_review(self):
        # hash_block stays in utils.py; the bare :7-8 (group_layers) moved to planning.py, so
        # after the rewrite it would read as utils.py's lines.
        text = "// hash (vllm/core/utils.py:1-2, then :7-8@%s).\n" % self.old[:8]
        out, n, report = self.run_on(text)
        self.assertEqual((out, n), (text, 0))
        self.assertTrue(any("bare citation moved" in r[4] for r in report), report)

    def test_a_paragraph_that_also_cites_another_release_is_left_for_review(self):
        text = "// (vllm/core/sched.py:6@%s; compare sched.py:6@v0.31.0).\n" % self.old[:8]
        out, n, report = self.run_on(text)
        self.assertEqual((out, n), (text, 0))
        self.assertTrue(any("also cites" in r[4] for r in report), report)

    def test_a_basename_the_file_names_two_full_paths_for_is_left_for_review(self):
        text = ("// vllm/core/sched.py:4@%s.\n\n// vllm/other/sched.py has its own b.\n\n"
                "// The loop (sched.py:6-7@%s).\n" % (self.old[:8], self.old[:8]))
        out, n, report = self.run_on(text)
        # Without the rule the nearest path, vllm/other/sched.py, would be taken, and its
        # unchanged lines 6-7 would "verify".
        self.assertEqual(n, 1)  # the first paragraph only
        self.assertIn("sched.py:6-7@%s" % self.old[:8], out)
        self.assertTrue(any("names two" in r[4] for r in report), report)

    def test_a_citation_outside_the_old_file_is_left_for_review(self):
        text = "// (vllm/core/sched.py:400@%s).\n" % self.old[:8]
        out, n, report = self.run_on(text)
        self.assertEqual((out, n), (text, 0))
        self.assertIn("outside the old file", report[0][4])


if __name__ == "__main__":
    unittest.main()
