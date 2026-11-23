# Original to blis-vllm.
"""Tests for reversion.py. Run: python3 -I tools/reversion/test_reversion.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reversion  # noqa: E402


class RewriteTest(unittest.TestCase):
    def rw(self, text):
        return reversion.rewrite(text, "0.32.0", "0.33.0")

    def test_each_shape_that_states_the_executing_release_moves(self):
        for before, after in [
            ('engine_version: "0.32.0"\n', 'engine_version: "0.33.0"\n'),
            ("engine_version: 0.32.0\n", "engine_version: 0.33.0\n"),
            ('engine: {kind: vllm, version: "0.32.0"}', 'engine: {kind: vllm, version: "0.33.0"}'),
            ('"engine: {kind: vllm, version: \\"0.32.0\\"}\\n"', '"engine: {kind: vllm, version: \\"0.33.0\\"}\\n"'),
            ('engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: r}', 'engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r}'),
            ('&scenario.Scenario{EngineVersion: "0.32.0"}', '&scenario.Scenario{EngineVersion: "0.33.0"}'),
            ('mustReplay(t, "0.32.0", "")', 'mustReplay(t, "0.33.0", "")'),
        ]:
            with self.subTest(before=before):
                self.assertEqual(self.rw(before)[0], after)

    def test_other_mentions_of_the_release_stay(self):
        for text in ['{"0.32.0", true}, {"0.31.0", true}', "vLLM v0.32.0's default (cache.py:102)",
                     'engine_version: "0.31.0"', 'engine_version: "0.32.01"', "# engine 0.32.0 notes",
                     'UpstreamVersion: "0.29.0"']:
            with self.subTest(text=text):
                self.assertEqual(self.rw(text), (text, 0))


class HeaderTest(unittest.TestCase):
    OLD, NEW = "5" * 40, "4" * 40

    def rw(self, text):
        return reversion.rewrite_header(text, "0.32.0", "0.33.0", self.OLD, self.NEW)

    def test_the_header_of_a_ported_file_moves(self):
        text = ("// SPDX-License-Identifier: Apache-2.0\n"
                "// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (%s).\n\npackage sched\n" % self.OLD)
        self.assertEqual(self.rw(text), (text.replace("v0.32.0 (" + self.OLD, "v0.33.0 (" + self.NEW), 1))

    def test_a_header_that_introduces_a_list_moves(self):
        text = "// Derived from vllm/v1/core/kv_cache_utils.py at vLLM v0.32.0 (%s):\n//   - a\n" % self.OLD
        self.assertEqual(self.rw(text)[1], 1)

    def test_a_derived_line_below_the_first_comment_block_stays(self):
        text = "package sched\n\n// Derived from vllm/x.py at vLLM v0.32.0 (%s).\n" % self.OLD
        self.assertEqual(self.rw(text), (text, 0))

    def test_a_header_naming_another_commit_stays(self):
        text = "// Derived from vllm/x.py at vLLM v0.32.0 (%s).\n" % ("6" * 40)
        self.assertEqual(self.rw(text), (text, 0))


if __name__ == "__main__":
    unittest.main()
