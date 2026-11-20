# Original to blis-vllm.
"""Platform replacements for the block-size oracle (design FID-1, §7.4).

vLLM's attention-backend modules import GPU-only wheels and compiled extensions, and a few
backends ask the platform what the machine can do. The oracle runs on a CPU, so each of those is
replaced here. Nothing replaced here decides a block size by itself: the priority list, the
validity filter and each backend's preferred and supported sizes are vLLM's own code, run as
written.

Every replacement is listed in REPLACEMENTS and printed in the table's header. Two of them are
FACTS: things the CPU cannot discover, which a combination's answer must not depend on to be
resolved. The oracle runs every combination under every setting of every fact.

  fa3_available   whether vLLM's FlashAttention wheel ships FA3 (flash_attn_interface.py:
                  _is_fa3_supported, :67-75); it needs CUDA 12.3 or later at build time.
  fa4_available   whether vLLM's FlashAttention wheel ships FA4 (vllm_flash_attn/
                  flash_attn_interface.py:_is_fa4_supported). It changes which FlashAttention
                  version is chosen (fa_utils.py:get_flash_attn_version, :74-180), and with it
                  FlashAttention's sink support (fa_utils.py:319-322) and its block sizes
                  (flash_attn.py:298-377).
  flashmla_built  whether vLLM's wheel carries the FlashMLA extensions
                  (v1/attention/ops/flashmla.py:_is_flashmla_available, :37-52). The FLASHMLA
                  backend is valid only if it does (mla/flashmla.py:74-95).
  trtllm_usable   whether FlashInfer's TRTLLM-gen kernels are usable
                  (utils/flashinfer.py:can_use_trtllm_attention :686-694, which needs NVIDIA's
                  artifactory, :620-645). It decides FlashInfer's sink support
                  (flashinfer.py:581-610) and whether it advertises large pages
                  (flashinfer.py:459-483).

Usage: imported by oracle/blocksize/oracle.py.
"""
import importlib.machinery
import sys
import types
from unittest import mock

REPLACEMENTS: list[tuple[str, str]] = []
FACTS = {"fa3_available": False, "fa4_available": False, "trtllm_usable": False, "flashmla_built": False}
FACT_DOC = {
    "fa3_available": "vLLM's FlashAttention wheel ships FA3, built for SM9x (flash_attn_interface.py:_is_fa3_supported)",
    "flashmla_built": "vLLM's wheel was built with FlashMLA (v1/attention/ops/flashmla.py:_is_flashmla_available, :37-52)",
    "fa4_available": "vLLM's FlashAttention wheel ships FA4 (flash_attn_interface.py:_is_fa4_supported)",
    "trtllm_usable": "FlashInfer's TRTLLM-gen kernels are usable (utils/flashinfer.py:can_use_trtllm_attention)",
}


def _spec(name):
    return importlib.machinery.ModuleSpec(name, None)


def stub_module(name: str, why: str):
    m = mock.MagicMock(name=name)
    m.__spec__ = _spec(name)
    m.__path__ = []
    sys.modules[name] = m
    REPLACEMENTS.append((f"module {name}", why))
    return m


def install_early():
    """Before vllm.platforms.cuda is imported: its compiled extensions and NVML."""
    for n in ("vllm._C_stable_libtorch", "vllm._C", "pynvml"):
        stub_module(n, "compiled extension or NVML, imported by vllm.platforms.cuda; decides no block size")


def install_late():
    """After the platform is replaced: the GPU-only wheels the backends import."""
    for n in ("flashinfer", "flashinfer.decode", "flashinfer.prefill", "flashinfer.utils", "flashinfer.page",
              "flashinfer.mla", "flashinfer.trtllm_low_latency_gemm", "flashinfer.fused_moe", "flashinfer.comm"):
        stub_module(n, "GPU-only wheel imported by the FlashInfer backends; its kernels are not run. ASSUMED: "
                    "flashinfer-python is installed, as requirements/cuda.txt:13 pins it")

    # vllm.vllm_flash_attn raises ImportError without the CUDA extensions (__init__.py:33-37).
    pkg = types.ModuleType("vllm.vllm_flash_attn")
    pkg.__spec__ = _spec("vllm.vllm_flash_attn")
    pkg.__path__ = []
    iface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    iface.__spec__ = _spec("vllm.vllm_flash_attn.flash_attn_interface")

    def is_fa_version_supported(v):
        from vllm.platforms import current_platform as cp

        if v == 2:
            return True  # ASSUMED: the CUDA wheel ships FA2 (flash_attn_interface.py:_is_fa2_supported)
        if v == 3:
            # FACT fa3_available; FA3 runs on SM9x only (:_is_fa3_supported)
            return FACTS["fa3_available"] and bool(cp.is_device_capability_family(90))
        if v == 4:
            # FACT fa4_available; FA4 runs on SM9x, 10x and 11x only (:_is_fa4_supported)
            return FACTS["fa4_available"] and any(cp.is_device_capability_family(f) for f in (90, 100, 110))
        raise ValueError(v)

    iface.is_fa_version_supported = is_fa_version_supported
    iface.fa_version_unsupported_reason = lambda v: None if is_fa_version_supported(v) else "unavailable (oracle)"
    iface.FA2_AVAILABLE = iface.FA3_AVAILABLE = True
    for n in ("flash_attn_varlen_func", "flash_attn_with_kvcache", "get_scheduler_metadata",
              "compile_flash_attn_varlen_func_from_specs"):
        setattr(iface, n, lambda *a, **k: None)
        setattr(pkg, n, lambda *a, **k: None)
    pkg.is_fa_version_supported = iface.is_fa_version_supported
    pkg.fa_version_unsupported_reason = iface.fa_version_unsupported_reason
    pkg.FA2_AVAILABLE = pkg.FA3_AVAILABLE = True
    sys.modules["vllm.vllm_flash_attn"] = pkg
    sys.modules["vllm.vllm_flash_attn.flash_attn_interface"] = iface
    REPLACEMENTS.append(("module vllm.vllm_flash_attn",
                         "CUDA flash-attention wheel; FA2 is assumed present, FA3 and FA4 are the facts "
                         "fa3_available and fa4_available"))

    # cutlass_mla.py:99 allocates a 128 MB CUDA workspace when the module is imported
    # (SM100Workspace, :70-72). It is never used for selection.
    import torch

    real_empty = torch.empty

    def empty(*a, **k):
        if k.get("device") == "cuda":
            k["device"] = "cpu"
        return real_empty(*a, **k)

    torch.empty = empty
    REPLACEMENTS.append(("torch.empty(device='cuda')",
                         "allocates on the CPU instead, so cutlass_mla.py:99's import-time workspace "
                         "(a module global, never read for selection) can be built"))
    import vllm.utils.platform_utils as pu

    pu.num_compute_units = lambda device_id=0: 132
    REPLACEMENTS.append(("vllm.utils.platform_utils.num_compute_units",
                         "a fixed SM count for cutlass_mla.py:73, read only to size that workspace"))


def patch_flashmla():
    """The fact flashmla_built replaces _is_flashmla_available (v1/attention/ops/flashmla.py:37-52)."""
    import vllm.v1.attention.ops.flashmla as fm

    def available():
        if FACTS["flashmla_built"]:
            return True, None
        return False, "vllm._flashmla_C is not available (oracle fact flashmla_built=False)"

    fm._is_flashmla_available = available
    REPLACEMENTS.append(("vllm.v1.attention.ops.flashmla._is_flashmla_available",
                         "the fact flashmla_built replaces the probe of the compiled extensions; "
                         "the capability rules in is_flashmla_dense_supported (:51-58) and "
                         "is_flashmla_sparse_supported (:61-74) run as written"))


def patch_trtllm():
    """The fact trtllm_usable replaces can_use_trtllm_attention and supports_trtllm_attention."""
    import vllm.utils.flashinfer as fi

    def can_use(num_qo_heads, num_kv_heads, is_prefill=False):
        return FACTS["trtllm_usable"] and num_qo_heads % max(num_kv_heads, 1) == 0

    def supports(is_prefill=False):
        from vllm.platforms import current_platform as cp

        if not FACTS["trtllm_usable"]:
            return False
        if cp.is_device_capability(90) or cp.is_device_capability_family(120):
            return not is_prefill  # utils/flashinfer.py:663-667
        return cp.is_device_capability_family(100)  # :669-670

    fi.can_use_trtllm_attention = can_use
    fi.supports_trtllm_attention = supports
    REPLACEMENTS.append(("vllm.utils.flashinfer.can_use_trtllm_attention / supports_trtllm_attention",
                         "the fact trtllm_usable replaces them; the capability rules (:663-670) are copied "
                         "and the artifactory and cubin probes (:620-645) are what the fact stands for"))
    return fi
