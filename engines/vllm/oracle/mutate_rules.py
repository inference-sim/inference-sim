# Original to blis-vllm.
"""Mutation sweep for the tier-1 rules (rules/v0_32): apply one mutant at a time to a scratch copy
of the module, run the rule's unit tests and its MatchSource checks against the pinned vLLM source,
and report whether each is caught.

Each mutant breaks one clause of a rule a supported row of docs/feature-matrix.md names, or one
piece of the data, rule or parser that v0.32.0 added over v0.31.0 (the 0.32.0 pack's commit), so
those mutants exist and run in CI rather than only on a contributor's machine. The mutant IDs are
C1-C19; D1-D11 are oracle/mutate_t3.py's.

Usage: python3 oracle/mutate_rules.py <blis-vllm checkout> <scratch dir> <vLLM v0.32.0 source> [C1,...]
Exits non-zero if the unmutated module fails, or a mutant survives or breaks the build without a
test failure.
"""
import os
import shutil
import subprocess
import sys

PACK = "rules/v0_32/pack.go"
SOURCE = "rules/v0_32/source_test.go"

DCP = "TestDCPDividesTP"
BIND = "TestMoEEPBackendBindsPassthrough"

# (id, the clause it breaks, file, exact old text, new text, the tests that must catch it)
MUTANTS = [
    # dcp-divides-tp (vllm/config/parallel.py:590-599@5df63a72).
    ("C1", "dcp-divides-tp: with pcp off, tp % dcp must be 0", PACK,
     "case pcp == 1 && tp%dcp != 0:", "case pcp == 1 && tp%dcp != 0 && false:", DCP),
    ("C2", "dcp-divides-tp: the modulus is tp by dcp, not dcp by tp", PACK,
     "case pcp == 1 && tp%dcp != 0:", "case pcp == 1 && dcp%tp != 0:", DCP),
    ("C3", "dcp-divides-tp: with pcp on, dcp may span the pcp axis", PACK,
     "case pcp > 1 && dcp != 1 && dcp != pcp && dcp != tp*pcp:",
     "case pcp > 1 && dcp != 1 && dcp != tp*pcp:", DCP),
    ("C4", "dcp-divides-tp: with pcp on, dcp may span tp x pcp", PACK,
     "case pcp > 1 && dcp != 1 && dcp != pcp && dcp != tp*pcp:",
     "case pcp > 1 && dcp != 1 && dcp != pcp:", DCP),
    ("C5", "dcp-divides-tp: an unstated dcp is one, not zero", PACK,
     "tp, pcp, dcp := pl.TP, max(pl.PCP, 1), max(pl.DCP, 1)",
     "tp, pcp, dcp := pl.TP, max(pl.PCP, 1), max(pl.DCP, 0)", DCP),
    # The data v0.32.0 changed (vllm/config/parallel.py:44-59, cache.py:39-63, kernel.py:122-169@5df63a72).
    ("C6", "all2all_backend gains passthrough", PACK,
     '"mori_low_latency", "moonep", "nixl_ep", "passthrough"),',
     '"mori_low_latency", "moonep", "nixl_ep"),',
     "TestAll2AllBackendMembership|TestAll2AllBackendsMatchSource"),
    ("C7", "passthrough joins the sequence-parallel MoE subset", PACK,
     '"mori_high_throughput", "mori_low_latency", "nixl_ep", "passthrough"),',
     '"mori_high_throughput", "mori_low_latency", "nixl_ep"),',
     "TestSPMoEBackendMembership|TestSPMoEBackendsMatchSource"),
    ("C8", "the cache dtypes gain ultraquant_4bit", PACK,
     '"ultraquant_4bit", "int4_per_token_head",', '"int4_per_token_head",',
     "TestSmallVocabularies|TestCacheDTypesMatchSource"),
    ("C9", "mamba_cache_mode loses all", PACK,
     'MambaCacheModes: set("none", "align"),', 'MambaCacheModes: set("none", "align", "all"),',
     "TestSmallVocabularies|TestMambaCacheModesMatchSource"),
    ("C10", "the MoE backends gain aiter_mega_moe", PACK,
     '"aiter", "aiter_mega_moe", "aiter_triton_mxfp4_bf16",', '"aiter", "aiter_triton_mxfp4_bf16",',
     "TestMoEBackendsMatchSource"),
    ("C11", "MEGA_MOE_BACKENDS is NATIVE | FLASHINFER_MOE_EP", PACK,
     '\t\t\t"flashinfer_moe_ep_mega_deep_gemm"),', '\t\t\t"flashinfer_moe_ep_mega_deep_gemm", "aiter_mega_moe"),',
     "TestMoEBackendsMatchSource"),
    # The new rule moe-ep-backend-binds-passthrough (vllm/config/kernel.py:194-227@5df63a72).
    ("C12", "moe-ep-backend-binds-passthrough: both FlashInfer MoE-EP backends communicate themselves", PACK,
     'selfCommunicating := moe == "flashinfer_moe_ep_cutedsl" || moe == "flashinfer_moe_ep_mega_deep_gemm"',
     'selfCommunicating := moe == "flashinfer_moe_ep_cutedsl"', BIND),
    ("C13", "moe-ep-backend-binds-passthrough: an EP megakernel refuses another explicit backend", PACK,
     "case selfCommunicating && a2a != \"passthrough\" && a2a != p.DefaultAll2AllBackend:",
     "case selfCommunicating && a2a != \"passthrough\" && a2a != p.DefaultAll2AllBackend && false:", BIND),
    ("C14", "moe-ep-backend-binds-passthrough: the default backend is rebound, not refused", PACK,
     "case selfCommunicating && a2a != \"passthrough\" && a2a != p.DefaultAll2AllBackend:",
     "case selfCommunicating && a2a != \"passthrough\":", BIND),
    ("C15", "moe-ep-backend-binds-passthrough: passthrough is refused beside any other MoE backend", PACK,
     'case !selfCommunicating && e.All2AllBackend == "passthrough":',
     'case !selfCommunicating && e.All2AllBackend == "passthrough" && moe != "":', BIND),
    ("C16", "moe-ep-backend-binds-passthrough: the binding reads the backend after the pplx/naive rename", PACK,
     "a2a := p.EffectiveAll2AllBackend(e.All2AllBackend)", "a2a := e.All2AllBackend", BIND),
    # The parser: MEGA_MOE_BACKENDS' module constants, and the sequence-parallel variable.
    ("C17", "the source check resolves FLASHINFER_MOE_EP_BACKENDS through its module constants", SOURCE,
     "\t\tconstants[m[1]] = m[2]\n", "\t\tconstants[m[1]] = m[1]\n", "TestMoEBackendsMatchSource"),
    ("C18", "DeepSeek-V4 folds data_parallel_size > 1 into moe_needs_token_sharded_input", SOURCE,
     'case clause == "moe_needs_token_sharded_input" && folded:',
     'case clause == "moe_needs_token_sharded_input" && !folded:', "TestModelSequenceParallelRulesMatchSource"),
    ("C19", "both DeepSeek-V4 releases keep the mega-MoE-or-dp clause", PACK,
     "rules.ArchDeepseekV4: rules.SequenceParallelCondition{PipelineParallelOne: true, ExpertParallel: true,\n"
     "\t\t\t\tTensorParallelAboveOne: true, MegaMoEOrDataParallelAboveOne: true},",
     "rules.ArchDeepseekV4: rules.SequenceParallelCondition{PipelineParallelOne: true, ExpertParallel: true,\n"
     "\t\t\t\tTensorParallelAboveOne: true},", "TestModelSequenceParallelRulesMatchSource"),
]

TESTS = "|".join(sorted({t for m in MUTANTS for t in m[5].split("|")}))


def run(scratch: str, vllm: str, tests: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, VLLM_SOURCE=vllm, VLLM_SOURCE_REQUIRED="1")
    return subprocess.run(["go", "test", "-count=1", "-run", tests, "./v0_32/"],
                          cwd=os.path.join(scratch, "rules"), capture_output=True, text=True, env=env)


def copy(src: str, scratch: str) -> None:
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))


def main() -> int:
    src, scratch, vllm = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    # The unmutated module must pass the same tests, with the source checks on, or a "caught"
    # below would mean nothing.
    copy(src, scratch)
    r = run(scratch, vllm, TESTS)
    if r.returncode != 0:
        print(f"BASELINE the unmutated rules fail {TESTS}:\n{(r.stdout + r.stderr)[-2000:]}")
        return 1
    print(f"BASELINE passes {TESTS} against {vllm}")
    bad = 0
    for mid, row, path, old, new, tests in MUTANTS:
        if only and mid not in only:
            continue
        copy(src, scratch)
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid} {row}: the mutation site is not unique in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new, 1))
        r = run(scratch, vllm, tests)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"SURVIVED {mid} {row}")
            bad += 1
        elif "build failed" in out or "[setup failed]" in out:
            print(f"BROKEN   {mid} {row}: the mutant does not compile")
            bad += 1
        else:
            print(f"CAUGHT   {mid} {row}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
