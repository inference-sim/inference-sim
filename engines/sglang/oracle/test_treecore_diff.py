# Original to blis-sglang.
"""run_treecore_diff.py's comparison fails on what it guards (X18): a different answer, a
different call at the pools and an inconsistent node pairing are found; node ids numbered
differently but consistently are not.

Usage: python3 -I oracle/test_treecore_diff.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_treecore_diff as r  # noqa: E402


def trace(base: int) -> dict:
    root, leaf = base, base + 1
    return {"calls": [
        {"seq": 0, "i": "allocator", "m": "alloc", "args": {"need_size": 2}, "ret": [1, 2]},
        {"seq": 1, "i": "prefix_cache", "m": "insert", "args": {"params": {"value": [1, 2]}},
         "ret": {"_t": "InsertResult", "prefix_len": 0, "last_device_node": leaf}},
        {"seq": 2, "i": "tree_core", "m": "begin_insert", "in": 1, "args": {}, "ret": {"opaque": base}},
        {"seq": 3, "i": "prefix_cache", "m": "match_prefix", "args": {},
         "ret": {"_t": "MatchResult", "device_prefix_len": 2, "last_device_node": leaf, "best_match_node": leaf}},
        {"seq": 4, "i": "prefix_cache", "m": "evict", "args": {}, "ret": {"num_tokens_evicted": 2}},
        {"seq": 5, "i": "allocator", "m": "free_segment", "in": 4, "args": {"free_index": [1, 2], "start_pos": 0}},
        {"seq": 6, "i": "prefix_cache", "m": "match_prefix", "args": {},
         "ret": {"_t": "MatchResult", "device_prefix_len": 0, "last_device_node": root, "best_match_node": root}},
    ]}


class Compare(unittest.TestCase):
    def test_consistent_renumbering_is_no_divergence(self):
        # The tree core's own calls (seq 2) are not compared: the cores differ there by design.
        self.assertEqual(r.compare("s", trace(10), trace(500)), [])

    def test_a_different_answer_is_found(self):
        b = trace(10)
        b["calls"][4]["ret"]["num_tokens_evicted"] = 1
        self.assertEqual([d["what"] for d in r.compare("s", trace(10), b)], ["prefix_cache.evict answers differently"])

    def test_a_different_free_is_found(self):
        b = trace(10)
        b["calls"][5]["args"]["free_index"] = [2, 1]
        self.assertEqual([d["what"] for d in r.compare("s", trace(10), b)],
                         ["prefix_cache.evict makes different calls at the pools"])

    def test_an_inconsistent_pairing_is_found(self):
        b = trace(10)
        b["calls"][5 + 1]["ret"]["last_device_node"] = 11  # the root answered as the leaf
        b["calls"][5 + 1]["ret"]["best_match_node"] = 11
        self.assertEqual([d["call"] for d in r.compare("s", trace(10), b)], [4])  # the fifth driver call

    def test_a_missing_call_is_found(self):
        b = copy.deepcopy(trace(10))
        del b["calls"][6]
        self.assertEqual(len(r.compare("s", trace(10), b)), 1)


if __name__ == "__main__":
    unittest.main()
