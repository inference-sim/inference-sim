# Original to blis-vllm.
"""Tests for gen_kernel_impls.py over a small synthetic vLLM tree and library checkouts.

Run: python3 -I kernels/test_gen_kernel_impls.py
"""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_kernel_impls as gen  # noqa: E402

VLLM = {
    "vllm/v1/attention/backends/registry.py": """
        class AttentionBackendEnum(Enum):
            FAST = "vllm.v1.attention.backends.fast.FastBackend"
            ROCM_ONLY = "vllm.v1.attention.backends.rocm.RocmBackend"
            LATER = "vllm.v1.attention.backends.later.LaterBackend"
            CUSTOM = None
    """,
    "vllm/platforms/cuda.py": """
        def _get_backend_priorities(use_mla, device_capability):
            return [
                AttentionBackendEnum.FAST,
                AttentionBackendEnum.LATER,
            ]
    """,
    "vllm/v1/attention/backends/fast.py": """
        from fastlib import prefill_kernel, plan_helper
        from fastlib.decode import decode_kernel
        from vllm.v1.attention.ops.tri import tri_attention

        class FastBackend:
            def forward(self, q):
                plan_helper(q)
                if q:
                    return prefill_kernel(q)
                return decode_kernel(q) + tri_attention(q)
    """,
    "vllm/v1/attention/ops/tri.py": "from vllm.v1.attention.ops.tri_helpers import tile\n\n\ndef tri_attention(q):\n    return tile(q)\n",
    "vllm/v1/attention/ops/tri_helpers.py": "def tile(q):\n    return q\n",
    "vllm/utils/gate.py": "def supports_fast(cap):\n    return cap >= 100\n",
    "requirements/cuda.txt": "torch==2.0\nfastlib-python[cu13]==1.2.3\n",
    "cmake/external_projects/other.cmake": "FetchContent_Declare(\n    other\n    GIT_TAG {other_commit}\n)\n",
    "docker/Dockerfile": "ARG CUDA_VERSION=13.0.3\n",
}
FASTLIB = {
    "version.txt": "1.2.3\n",
    "csrc/prefill.cu": '#include "fastlib/tile.cuh"\n#include <cuda_runtime.h>\n// prefill\n',
    "include/fastlib/tile.cuh": "// tile\n",
    "csrc/decode.cu": "// decode\n",
    "fastlib/artifacts.py": """
        class CheckSumHash:
            FMHA: str = (
                "1111111111111111111111111111111111111111111111111111111111111111"
            )
    """,
}
OTHER = {"kernels/k.cu": "// other\n"}


GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def commit_all(root, tag=None):
    for cmd in (["add", "-A"], ["commit", "-q", "-m", "y", "--no-gpg-sign"]):
        subprocess.run(["git", "-C", root, *cmd], check=True, env=GIT_ENV)
    if tag:
        subprocess.run(["git", "-C", root, "tag", "-f", tag], check=True, env=GIT_ENV, capture_output=True)


def git_tree(root, files, tag=None):
    for rel, body in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(body).lstrip("\n"))
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x", "--no-gpg-sign"]):
        subprocess.run(["git", "-C", root, *cmd], check=True, env=env)
    if tag:
        subprocess.run(["git", "-C", root, "tag", tag], check=True, env=env)
    return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def classification(commit):
    return {
        "upstream": {"tag": "v9.9.9", "commit": commit},
        "archs": {"sm_100a": {}, "sm_90a": {}},
        "libraries": {
            "fastlib": {"pip": "fastlib-python", "requirements": "requirements/cuda.txt", "repo": "https://example.org/fastlib",
                        "version_file": "version.txt", "checksums": "fastlib/artifacts.py", "modules": ["fastlib"],
                        "include_search": ["include"], "unhashed_includes": {}},
            "otherlib": {"cmake": "cmake/external_projects/other.cmake", "repo": "https://example.org/other", "modules": ["otherlib"]},
            "vllm-csrc": {"engine": True, "repo": "https://example.org/vllm", "modules": ["vllm.v1.attention.ops"],
                          "python_closure": ["vllm/v1/attention/ops/"]},
        },
        "backends": {
            "FAST": {"status": "listed", "not_kernels": {"plan_helper": "plans"}},
            "ROCM_ONLY": {"status": "not_cuda", "reason": "ROCm"},
            "LATER": {"status": "gap", "issue": "org/repo#7"},
            "CUSTOM": {"status": "not_lowered", "reason": "placeholder"},
        },
        "kernels": [
            {"backend": "FAST", "library": "fastlib", "variant": "prefill_kernel", "call": "prefill_kernel",
             "primitives": ["attention.prefill"], "archs": ["sm_100a"], "gates": ["vllm/utils/gate.py:supports_fast"],
             "compiler": "nvcc+prebuilt", "hash": {"artifact": "FMHA", "source": ["csrc/prefill.cu"]}},
            {"backend": "FAST", "library": "fastlib", "variant": "decode_kernel", "call": "decode_kernel",
             "primitives": ["attention.decode"], "archs": ["sm_100a", "sm_90a"], "compiler": "nvcc",
             "hash": {"source": ["csrc"]}},
            {"backend": "FAST", "library": "vllm-csrc", "variant": "tri_attention", "call": "tri_attention",
             "primitives": ["attention.decode"], "archs": ["sm_90a"], "compiler": "triton",
             "hash": {"source": ["vllm/v1/attention/ops/tri.py"]}},
        ],
    }


class GenKernelImplsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = self.tmp.name
        self.fastlib = os.path.join(base, "fastlib")
        self.other = os.path.join(base, "other")
        self.vllm = os.path.join(base, "vllm")
        for d in (self.fastlib, self.other, self.vllm):
            os.makedirs(d)
        git_tree(self.fastlib, FASTLIB, tag="v1.2.3")
        self.other_commit = git_tree(self.other, OTHER)
        files = dict(VLLM)
        files["cmake/external_projects/other.cmake"] = files["cmake/external_projects/other.cmake"].format(
            other_commit=self.other_commit)
        self.commit = git_tree(self.vllm, files)
        self.cls = classification(self.commit)
        self.sources = {"fastlib": self.fastlib, "otherlib": self.other}

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self, cls=None):
        return gen.generate(self.vllm, cls or self.cls, self.sources)

    def refused(self, cls):
        with self.assertRaises(gen.Refused) as ctx:
            self.generate(cls)
        return str(ctx.exception)

    def git(self, root, *args):
        return subprocess.run(["git", "-C", root, *args], check=True, capture_output=True, text=True).stdout.strip()

    def row(self, table, prim, variant, arch):
        rows = [r for r in table["implementations"]
                if (r["primitive"], r["variant"], r["target_arch"]) == (prim, variant, arch)]
        self.assertEqual(len(rows), 1, f"{prim} {variant} {arch}: {len(rows)} rows")
        return rows[0]

    def test_each_kernel_of_a_listed_backend_is_a_row_per_primitive_and_arch(self):
        t = self.generate()
        got = sorted((r["primitive"], r["library"], r["variant"], r["target_arch"]) for r in t["implementations"])
        self.assertEqual(got, [
            ("attention.decode", "fastlib", "decode_kernel", "sm_100a"),
            ("attention.decode", "fastlib", "decode_kernel", "sm_90a"),
            ("attention.decode", "vllm-csrc", "tri_attention", "sm_90a"),
            ("attention.prefill", "fastlib", "prefill_kernel", "sm_100a"),
        ])
        self.assertEqual(t["engine"], {"kind": "vllm", "version": "9.9.9", "commit": self.commit})
        self.assertEqual(sorted(t["archs"]), ["sm_100a", "sm_90a"])

    def test_a_source_kernel_hashes_the_git_objects_of_its_source_and_its_arch(self):
        t = self.generate()
        a = self.row(t, "attention.decode", "decode_kernel", "sm_100a")
        b = self.row(t, "attention.decode", "decode_kernel", "sm_90a")
        tree = self.git(self.fastlib, "rev-parse", "HEAD:csrc")
        self.assertEqual(a["hash_inputs"], [f"csrc tree {tree}"])
        self.assertEqual(a["hash_basis"], "source")
        self.assertEqual(a["code_hash"], gen.code_hash("source", "fastlib", "decode_kernel", "sm_100a", a["hash_inputs"]))
        self.assertNotEqual(a["code_hash"], b["code_hash"], "two architectures share a hash")

    def test_a_binary_kernel_hashes_the_checksum_its_library_records(self):
        r = self.row(self.generate(), "attention.prefill", "prefill_kernel", "sm_100a")
        self.assertEqual(r["hash_basis"], "binary")
        self.assertIn("FMHA checksums.txt sha256:" + "1" * 64, r["hash_inputs"])
        self.assertTrue(any(i.startswith("csrc/prefill.cu blob ") for i in r["hash_inputs"]),
                        "the launcher a precompiled kernel is loaded by is part of its build")

    def test_a_change_to_a_kernels_source_changes_only_its_hash(self):
        before = self.generate()
        with open(os.path.join(self.fastlib, "csrc", "decode.cu"), "a") as f:
            f.write("// changed\n")
        commit_all(self.fastlib, tag="v1.2.3")
        after = self.generate()
        for prim, variant, arch, changed in (("attention.decode", "decode_kernel", "sm_100a", True),
                                             ("attention.prefill", "prefill_kernel", "sm_100a", False),
                                             ("attention.decode", "tri_attention", "sm_90a", False)):
            same = self.row(before, prim, variant, arch)["code_hash"] == self.row(after, prim, variant, arch)["code_hash"]
            self.assertEqual(same, not changed, f"{variant} on {arch}")

    def test_versions_and_pins_come_from_vllms_build_files(self):
        libs = {lib["name"]: lib for lib in self.generate()["libraries"]}
        self.assertEqual(libs["fastlib"]["version"], "1.2.3")
        self.assertEqual(libs["fastlib"]["pin"], f"requirements/cuda.txt:2@{self.commit[:8]}")
        self.assertEqual(libs["vllm-csrc"]["version"], self.commit)
        self.assertNotIn("otherlib", libs, "a library no listed kernel uses is not listed")

    def test_a_kernel_cites_its_call_its_gate_and_the_platforms_selection(self):
        r = self.row(self.generate(), "attention.prefill", "prefill_kernel", "sm_100a")
        c = self.commit[:8]
        self.assertEqual(r["selected_by"], sorted([f"vllm/platforms/cuda.py:3@{c}", f"vllm/utils/gate.py:1@{c}",
                                                   f"vllm/v1/attention/backends/fast.py:9@{c}"]))
        self.assertEqual(r["toolkit"], "cuda 13.0.3 launcher, precompiled kernels")

    def test_a_gap_backend_is_a_gap_under_each_primitive(self):
        gaps = self.generate()["gaps"]
        self.assertEqual([(g["primitive"], g["selection"], g["issue"]) for g in gaps], [
            ("attention.decode", "attention backend LATER", "org/repo#7"),
            ("attention.prefill", "attention backend LATER", "org/repo#7")])

    # Each way the classification and the source can disagree is refused, naming the backend.

    def test_a_backend_with_no_status_is_refused(self):
        cls = copy.deepcopy(self.cls)
        del cls["backends"]["ROCM_ONLY"]
        self.assertIn("ROCM_ONLY", self.refused(cls))

    def test_a_status_for_a_backend_vllm_lacks_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["backends"]["GONE"] = {"status": "not_cuda", "reason": "x"}
        self.assertIn("GONE", self.refused(cls))

    def test_a_backend_the_cuda_platform_selects_cannot_be_not_cuda(self):
        cls = copy.deepcopy(self.cls)
        cls["backends"]["LATER"] = {"status": "not_cuda", "reason": "x"}
        self.assertIn("vllm/platforms/cuda.py:4", self.refused(cls))

    def test_an_unclassified_kernel_library_import_is_refused(self):
        cls = copy.deepcopy(self.cls)
        del cls["backends"]["FAST"]["not_kernels"]["plan_helper"]
        msg = self.refused(cls)
        self.assertIn("plan_helper", msg)
        self.assertIn("fast.py:1", msg)

    def test_a_name_the_module_no_longer_imports_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["backends"]["FAST"]["not_kernels"]["removed_helper"] = "x"
        self.assertIn("removed_helper", self.refused(cls))

    def test_a_kernel_its_backend_never_calls_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["backends"]["FAST"]["not_kernels"].pop("plan_helper")
        cls["kernels"].append({"backend": "FAST", "library": "fastlib", "variant": "plan_helper", "call": "plan_helper",
                               "primitives": ["attention.decode"], "archs": ["sm_90a"], "hash": {"source": ["csrc"]}})
        with open(os.path.join(self.vllm, "vllm/v1/attention/backends/fast.py")) as f:
            body = f.read().replace("plan_helper(q)\n", "pass\n")
        with open(os.path.join(self.vllm, "vllm/v1/attention/backends/fast.py"), "w") as f:
            f.write(body)
        self.git(self.vllm, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "y", "--no-gpg-sign")
        cls["upstream"]["commit"] = self.git(self.vllm, "rev-parse", "HEAD")
        self.assertIn("never calls plan_helper", self.refused(cls))

    def test_a_gate_that_does_not_exist_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["kernels"][0]["gates"] = ["vllm/utils/gate.py:supports_slow"]
        self.assertIn("supports_slow", self.refused(cls))

    def test_a_checkout_at_another_version_than_the_pin_is_refused(self):
        with open(os.path.join(self.fastlib, "version.txt"), "w") as f:
            f.write("1.2.4\n")
        commit_all(self.fastlib, tag="v1.2.3")
        self.assertIn("1.2.4", self.refused(self.cls))

    def test_a_pip_library_at_a_commit_after_its_tag_is_refused(self):
        with open(os.path.join(self.fastlib, "csrc", "decode.cu"), "a") as f:
            f.write("// later\n")
        commit_all(self.fastlib)
        self.assertIn("not tag v1.2.3", self.refused(self.cls))

    def test_a_checkout_with_changes_is_refused(self):
        with open(os.path.join(self.fastlib, "csrc", "decode.cu"), "a") as f:
            f.write("// uncommitted\n")
        self.assertIn("has changes", self.refused(self.cls))

    def test_a_built_library_checked_out_at_another_commit_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["kernels"].append({"backend": "FAST", "library": "otherlib", "variant": "k", "call": "decode_kernel",
                               "primitives": ["attention.decode"], "archs": ["sm_100a"], "hash": {"source": ["kernels"]}})
        self.sources["otherlib"] = self.fastlib
        self.assertIn(self.other_commit, self.refused(cls))

    def test_a_vllm_tree_at_another_commit_is_refused(self):
        cls = copy.deepcopy(self.cls)
        cls["upstream"]["commit"] = "0" * 40
        self.assertIn("0" * 40, self.refused(cls))

    def test_a_classification_of_another_release_is_refused(self):
        path = os.path.join(self.tmp.name, "cls.json")
        with open(path, "w") as f:
            json.dump(self.cls, f)
        with self.assertRaises(gen.Refused) as ctx:
            gen.load_classification("v9.9.8", path)
        self.assertIn("v9.9.9", str(ctx.exception))
        self.assertEqual(gen.load_classification("v9.9.9", path)["upstream"]["commit"], self.commit)

    def test_a_release_with_no_classification_is_refused(self):
        with self.assertRaises(gen.Refused) as ctx:
            gen.load_classification("v0.0.1")
        self.assertIn("v0.0.1", str(ctx.exception))

    def test_each_releases_classification_is_found_by_its_tag(self):
        here = os.path.dirname(os.path.abspath(__file__))
        tags = sorted(d for d in os.listdir(here) if os.path.isfile(os.path.join(here, d, "attention.json")))
        self.assertTrue(tags, "no classification under kernels/<tag>/")
        for tag in tags:
            self.assertEqual(gen.load_classification(tag)["upstream"]["tag"], tag)

    def test_the_pins_regen_fetches_are_the_ones_vllm_states(self):
        pins = gen.library_pins(self.vllm, self.cls)
        self.assertEqual(pins["fastlib"][2:], ("https://example.org/fastlib", "v1.2.3"))
        self.assertEqual(pins["otherlib"][2:], ("https://example.org/other", self.other_commit))
        self.assertIsNone(pins["vllm-csrc"][2], "the engine's own kernels are read from vLLM itself")

    def seen_checkout(self, version, checksum, tag=None):
        files = dict(FASTLIB)
        files["version.txt"] = version + "\n"
        files["fastlib/artifacts.py"] = FASTLIB["fastlib/artifacts.py"].replace("1" * 64, checksum)
        root = os.path.join(self.tmp.name, "fastlib-" + version)
        os.makedirs(root)
        git_tree(root, files, tag=tag or "v" + version)
        return root

    def with_seen(self, version="1.1.0", seen_in="run r1 recorded fastlib 1.1.0"):
        cls = copy.deepcopy(self.cls)
        cls["libraries"]["fastlib"]["seen"] = [{"version": version, "seen_in": seen_in}]
        return cls

    def test_a_version_seen_in_runs_is_listed_unpinned_with_its_own_build(self):
        self.sources["fastlib@1.1.0"] = self.seen_checkout("1.1.0", "2" * 64)
        t = self.generate(self.with_seen())
        libs = {(l["name"], l["version"]): l for l in t["libraries"]}
        self.assertFalse(libs[("fastlib", "1.1.0")]["pinned"])
        self.assertEqual(libs[("fastlib", "1.1.0")]["pin"], "run r1 recorded fastlib 1.1.0")
        self.assertTrue(libs[("fastlib", "1.2.3")]["pinned"])
        rows = {r["library_version"]: r for r in t["implementations"] if r["variant"] == "prefill_kernel"}
        self.assertIn("FMHA checksums.txt sha256:" + "2" * 64, rows["1.1.0"]["hash_inputs"])
        self.assertNotEqual(rows["1.1.0"]["code_hash"], rows["1.2.3"]["code_hash"])

    def test_an_unchanged_kernel_keeps_its_hash_in_a_seen_version(self):
        self.sources["fastlib@1.1.0"] = self.seen_checkout("1.1.0", "1" * 64)
        rows = {r["library_version"]: r for r in self.generate(self.with_seen())["implementations"]
                if r["variant"] == "prefill_kernel"}
        self.assertEqual(rows["1.1.0"]["code_hash"], rows["1.2.3"]["code_hash"])

    def test_a_seen_version_without_its_run_record_is_refused(self):
        self.sources["fastlib@1.1.0"] = self.seen_checkout("1.1.0", "2" * 64)
        self.assertIn("seen_in", self.refused(self.with_seen(seen_in="")))

    def test_a_seen_version_checked_out_at_another_version_is_refused(self):
        self.sources["fastlib@1.1.0"] = self.seen_checkout("1.0.9", "2" * 64, tag="v1.1.0")
        self.assertIn("is not fastlib 1.1.0", self.refused(self.with_seen()))

    def test_the_pin_is_not_listed_again_as_seen(self):
        self.assertIn("is the pin", self.refused(self.with_seen(version="1.2.3")))

    def edit(self, root, rel, text, tag=None):
        with open(os.path.join(root, rel), "a") as f:
            f.write(text)
        commit_all(root, tag=tag)
        if root == self.vllm:
            self.cls["upstream"]["commit"] = self.git(self.vllm, "rev-parse", "HEAD")

    def hash_of(self, variant, arch):
        return next(r["code_hash"] for r in self.generate()["implementations"]
                    if r["variant"] == variant and r["target_arch"] == arch)

    def test_a_header_the_kernel_includes_moves_its_hash(self):
        before = self.hash_of("prefill_kernel", "sm_100a")
        self.edit(self.fastlib, "include/fastlib/tile.cuh", "// changed\n", tag="v1.2.3")
        self.assertNotEqual(self.hash_of("prefill_kernel", "sm_100a"), before)

    def test_a_triton_helper_the_kernel_imports_moves_its_hash(self):
        before = self.hash_of("tri_attention", "sm_90a")
        self.edit(self.vllm, "vllm/v1/attention/ops/tri_helpers.py", "# changed\n")
        self.assertNotEqual(self.hash_of("tri_attention", "sm_90a"), before)

    def test_an_include_the_tree_does_not_hold_is_refused_unless_named(self):
        self.edit(self.fastlib, "csrc/prefill.cu", '#include "gen/config.inc"\n', tag="v1.2.3")
        self.assertIn("gen/config.inc", self.refused(self.cls))
        self.cls["libraries"]["fastlib"]["unhashed_includes"] = {"gen/": "generated by the JIT spec"}
        self.generate()

    def test_each_kernel_records_its_own_compiler(self):
        tk = {r["variant"]: r["toolkit"] for r in self.generate()["implementations"]}
        self.assertEqual(tk["tri_attention"], "triton (torch==2.0)")
        self.assertEqual(tk["decode_kernel"], "cuda 13.0.3")

    def test_an_unclassified_jit_function_is_refused(self):
        self.edit(self.vllm, "vllm/v1/attention/backends/fast.py",
                  "\n\n@triton.jit\ndef _dequant(x):\n    pass\n")
        self.assertIn("_dequant", self.refused(self.cls))
        self.cls["backends"]["FAST"]["jit"] = {"_dequant": {"not_attention": "a test"}}
        self.generate()

    def test_a_jit_function_compiled_into_a_kernel_must_be_in_its_hash(self):
        self.edit(self.vllm, "vllm/v1/attention/backends/fast.py", "\n\n@cute.jit\ndef _mask(x):\n    pass\n")
        self.cls["backends"]["FAST"]["jit"] = {"_mask": {"part_of": "tri_attention"}}
        self.assertIn("hash.engine_source", self.refused(self.cls))
        self.cls["kernels"][2]["hash"]["engine_source"] = ["vllm/v1/attention/backends/fast.py"]
        r = next(r for r in self.generate()["implementations"] if r["variant"] == "tri_attention")
        self.assertTrue(any("vllm/v1/attention/backends/fast.py" in i for i in r["hash_inputs"]))

    def test_an_unclassified_registered_op_is_refused(self):
        self.edit(self.vllm, "vllm/v1/attention/backends/fast.py", "\n\ndef f(x):\n    torch.ops._C.paged_attention(x)\n")
        self.assertIn("torch.ops._C.paged_attention", self.refused(self.cls))

    def test_a_kernel_classified_under_another_library_is_refused(self):
        self.cls["kernels"][1]["library"] = "vllm-csrc"
        self.assertIn("a module of fastlib", self.refused(self.cls))

    def test_two_gaps_of_one_kernel_with_different_issues_are_refused(self):
        self.edit(self.vllm, "vllm/v1/attention/backends/fast.py", "\nfrom fastlib import sink_kernel\n")
        self.cls["backends"]["FAST"]["gaps"] = [
            {"call": "sink_kernel", "primitives": ["attention.decode"], "issue": "org/repo#1"},
            {"call": "sink_kernel", "primitives": ["attention.decode"], "issue": "org/repo#2"}]
        self.assertIn("named twice", self.refused(self.cls))

    def test_a_gap_inside_a_listed_backend_names_its_kernel(self):
        self.edit(self.vllm, "vllm/v1/attention/backends/fast.py", "\nfrom fastlib import sink_kernel\n")
        self.cls["backends"]["FAST"]["gaps"] = [{"call": "sink_kernel", "primitives": ["attention.decode"], "issue": "org/repo#1"}]
        g = [g for g in self.generate()["gaps"] if g.get("variant")]
        self.assertEqual(g, [{"primitive": "attention.decode", "selection": "attention backend FAST",
                              "library": "fastlib", "variant": "sink_kernel", "issue": "org/repo#1"}])

    def test_a_build_entry_a_seen_version_lacks_is_hashed_as_absent(self):
        files = dict(FASTLIB)
        files["version.txt"] = "1.1.0\n"
        del files["csrc/decode.cu"]
        files["csrc/other.cu"] = "// keeps csrc a tree\n"
        root = os.path.join(self.tmp.name, "fastlib-old")
        os.makedirs(root)
        git_tree(root, files, tag="v1.1.0")
        self.sources["fastlib@1.1.0"] = root
        self.cls["kernels"][1]["hash"]["source"] = ["csrc/decode.cu"]
        r = next(r for r in self.generate(self.with_seen())["implementations"]
                 if r["variant"] == "decode_kernel" and r["library_version"] == "1.1.0")
        self.assertIn("csrc/decode.cu absent", r["hash_inputs"])

    def test_regen_refuses_a_keep_directory_that_is_not_empty(self):
        import regen
        keep = os.path.join(self.tmp.name, "keep")
        os.makedirs(keep)
        open(os.path.join(keep, "x"), "w").close()
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(regen.main(["v0.32.0", "--keep", keep, "--out", os.path.join(self.tmp.name, "o.json")]), 1)
        self.assertIn("not empty", err.getvalue())

    def test_check_fails_on_a_table_that_is_not_current(self):
        path = os.path.join(self.tmp.name, "cls.json")
        with open(path, "w") as f:
            json.dump(self.cls, f)
        out = os.path.join(self.tmp.name, "kernel_impls.json")
        args = [self.vllm, "--upstream", "v9.9.9", "--classification", path,
                "--source", f"fastlib={self.fastlib}", "--source", f"otherlib={self.other}"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(gen.main(args), 0)
        with open(out, "w") as f:
            f.write(buf.getvalue())
        with redirect_stderr(io.StringIO()):
            self.assertEqual(gen.main(args + ["--check", out]), 0)
            with open(out, "a") as f:
                f.write(" ")
            self.assertEqual(gen.main(args + ["--check", out]), 1)


if __name__ == "__main__":
    unittest.main()
