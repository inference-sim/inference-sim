"""InferenceX model slugs and GPU slugs, mapped to blis-catalog model and hardware names.

The InferenceX extractors (extract_inferencex_direct.py, extract_inferencex_absolutes.py) name a
deployment by these slugs. The tables came from extract_aisimulate_e2e.py, whose artifact used
the same slugs until AISimulate's e2e artifact began naming models by Hugging Face checkpoint.
That extractor now reads the catalog instead; these tables stay here, unchanged, for the
InferenceX rows that still carry slugs.
"""

# Artifact model slug -> blis-catalog model name. A slug absent here is skipped: the catalog
# has no graph for it, so it cannot be priced.
MODELS = {
    "gptoss120b": "gpt-oss-120b",
    "minimaxm2.5": "minimax-m2.5",
    "qwen3.5": "qwen3.5-397b-a17b",
    "glm5": "glm-5",
    # The snapshot measures meta-llama/Meta-Llama-3.1-70B; the catalog carries the Instruct
    # variant. The mapping is VERIFIED rather than assumed on the grounds that instruction
    # tuning changes no shape: the base config is gated on HuggingFace, so it was read from
    # two independent mirrors (unsloth/Meta-Llama-3.1-70B and
    # NousResearch/Meta-Llama-3.1-70B) and compared field by field against the committed
    # Instruct config. All thirteen shape fields agree -- 80 layers, 8192 hidden, 28672
    # intermediate, 64 q heads over 8 kv, 128256 vocab, bfloat16, rope_theta 500000,
    # 131072 positions, silu, rms_norm_eps 1e-05, tie_word_embeddings false,
    # LlamaForCausalLM, no head_dim -- with zero differences.
    #
    # It is the corpus's largest model at 408 points and its only DENSE one; the other four
    # are all MoE, so without it the comparison says nothing about dense architectures.
    "llama70b": "llama-3.1-70b-instruct",
    # Added to blis-catalog this cycle, each with its graph DERIVED by the catalog's own
    # scripts/derive_graph.py rather than written: deepseek-v3 prices as MLA plus routed
    # experts with a three-layer dense prologue, and kimi-k2.5 as the same family -- vLLM
    # instantiates its text decoder as DeepseekV2ForCausalLM over config.text_config, and the
    # text_config names DeepseekV3ForCausalLM itself.
    #
    # DeepSeek-V4-Pro and MiniMax-M3 are deliberately absent. Both carry a learned
    # block-sparse indexer (index_n_heads / sparse_attention_config), which the schema names
    # as sparse_mla but blis-registry has no coefficients for; the kernel would fall back to
    # the generic attention rate and misprice them silently.
    "dsr1": "deepseek-v3",
    "kimik2.5": "kimi-k2.5",
    # MiniMax-M3 is MiniMaxM3SparseForConditionalGeneration, a vision-language checkpoint
    # whose text decoder the artifact measures. Its attention is block-sparse: the read is
    # bounded by sparse_topk_blocks * sparse_block_size and a separate indexer scores which
    # blocks to read. The catalog derives it from the published config at revision
    # f0e1c1e0, with every shape cross-checked against the safetensors headers.
    "minimaxm3": "minimax-m3",
    # DeepSeek-V4-Pro: compressed sparse attention at two ratios with a top-k indexer.
    # The catalog derives it from the published config at revision b5968e91, with every
    # bound read from vLLM's own compressor and sparse_mla rather than from field names.
    "dsv4": "deepseek-v4-pro",
}

# Artifact GPU slug -> blis-catalog hardware name.
CHIPS = {"h100": "h100", "h200": "h200", "b200": "b200", "b300": "b300"}
