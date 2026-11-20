# Original to blis-sglang.
"""Tests for check_replacements.py: each kind of unlisted replacement fails it.

Run: python3 -I oracle/lower/test_check_replacements.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_replacements  # noqa: E402


def real():
    with open(os.path.join(HERE, "extract.py")) as f:
        src = f.read()
    with open(os.path.join(HERE, "replacements.json")) as f:
        return src, json.load(f)


class T(unittest.TestCase):
    def test_the_committed_list_is_exact(self):
        src, listed = real()
        self.assertEqual(check_replacements.problems(src, listed), [])

    def test_an_unlisted_stand_in_class_fails(self):
        src, listed = real()
        src += "\nclass FakeAllocator:\n    pass\n"
        self.assertIn("extract.py replaces FakeAllocator, which oracle/lower/replacements.json does not list (FID-1)",
                      check_replacements.problems(src, listed))

    def test_an_unlisted_module_patch_fails(self):
        src, listed = real()
        src += "\nimport sglang\nsglang.srt.utils.common.is_hip = lambda: True\n"
        self.assertIn("extract.py replaces sglang.srt.utils.common.is_hip, which oracle/lower/replacements.json does not list (FID-1)",
                      check_replacements.problems(src, listed))

    def test_an_unlisted_patching_helper_fails(self):
        src, listed = real()
        src += "\nimport run_serverargs\nrun_serverargs.restore_rocm({})\n"
        self.assertTrue(any("run_serverargs.restore_rocm" in e for e in check_replacements.problems(src, listed)))

    def test_a_stale_entry_and_a_bad_kind_fail(self):
        src, listed = real()
        listed["replacements"].append({"symbol": "Gone", "kind": "transcription", "how": "x"})
        errs = check_replacements.problems(src, listed)
        self.assertIn("replacements.json lists Gone, which extract.py does not replace", errs)
        self.assertTrue(any("kind 'transcription'" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
