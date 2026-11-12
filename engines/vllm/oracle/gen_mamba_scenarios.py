# Original to blis-vllm.
"""Writes the hand-designed hybrid Mamba scenarios under oracle/scenarios (mamba-*.json).

Each scenario is a model with one full-attention (or MLA) layer type and one Mamba (MambaSpec)
layer type, run through vLLM v0.32.0's real Scheduler by oracle/run_scheduler.py. The numbers are
the post-alignment ones the engine runs with: the attention block size is the engine's block size,
the Mamba block size equals it in "align" mode (mamba_cache_mode with prefix caching on) and is
max_model_len in "none" mode, and the Mamba page is padded to the attention page
(vllm/platforms/interface.py:759-1126@5df63a72). The prompts and outputs are drawn from a
generator seeded by the scenario's name, so the files are reproducible:

    python3 -I oracle/gen_mamba_scenarios.py oracle/scenarios

Usage: python3 -I oracle/gen_mamba_scenarios.py <out dir>
"""

import json
import os
import random
import sys

# The toy model. A full-attention layer of one 4-element head holds keys and values of float32,
# so its page is 2 * block * 4 * 4 = 32 * block bytes: 256 bytes at block 8, 512 at block 16.
TOY_HEAD = 4


def attn(count, heads=1, head=TOY_HEAD, elem=None, spec="full"):
    run = {"count": count, "spec": spec, "num_kv_heads": heads, "head_size": head}
    if elem:
        run["elem_bytes"] = elem
    return run


def mamba(count, state, padded, block, mode, speculative=0, ckpt=0, align=None):
    run = {"count": count, "spec": "mamba", "state_bytes": state, "page_size_padded": padded,
           "block_size": block, "mamba_cache_mode": mode, "num_speculative_blocks": speculative,
           "num_prefill_checkpoint_blocks": ckpt}
    if ckpt:
        run["prefill_checkpoint_alignment"] = align
    return run


class Builder:
    def __init__(self, name):
        self.name = name
        self.rng = random.Random("mamba:" + name)
        self.requests = []

    def tokens(self, n):
        return [self.rng.randint(1, 50000) for _ in range(n)]

    def add(self, rid, arrival, prompt, out, **extra):
        r = {"id": rid, "arrival_step": arrival, "prompt": prompt, "output": self.tokens(out)}
        r.update(extra)
        self.requests.append(r)
        return prompt


def config(budget, seqs, block, layers, blocks=200, max_len=128, caching=True, threshold=0, **extra):
    c = {"max_num_seqs": seqs, "max_num_batched_tokens": budget, "block_size": block,
         "num_blocks": blocks, "max_model_len": max_len, "long_prefill_token_threshold": threshold,
         "enable_prefix_caching": caching, "scheduling_policy": "fcfs", "kv_layers": layers}
    c.update(extra)
    return c


def toy_layers(block, mode, speculative=0, ckpt=0, align=None, mamba_layers=1, mamba_block=None):
    page = 32 * block
    return [attn(1), mamba(mamba_layers, page * 3 // 4, page, mamba_block or block, mode,
                           speculative, ckpt, align)]


def write(out, sc, description, max_steps=300):
    sc = {"name": sc.name, "description": description, **sc.body, "max_steps": max_steps}
    check(sc)
    with open(os.path.join(out, sc["name"] + ".json"), "w") as f:
        json.dump(sc, f, indent=1)
        f.write("\n")


class Scenario:
    def __init__(self, b, cfg):
        self.name, self.body = b.name, {"config": cfg, "requests": b.requests}


def check(sc):
    """The scenario schema the runners and the Go diff test read; a bad file fails here, not in CI."""
    c = sc["config"]
    assert c["max_num_batched_tokens"] >= c["max_num_seqs"], sc["name"]
    assert c["max_model_len"] % c["block_size"] == 0, sc["name"]
    ids = [r["id"] for r in sc["requests"]]
    assert len(ids) == len(set(ids)), sc["name"]
    runs = [r for r in c["kv_layers"] if r["spec"] == "mamba"]
    assert runs, sc["name"]
    for m in runs:
        assert m["state_bytes"] <= m["page_size_padded"], sc["name"]
        if m["mamba_cache_mode"] == "align":
            assert c["enable_prefix_caching"] and m["block_size"] == c["block_size"], sc["name"]
        else:
            assert not c["enable_prefix_caching"] and m["block_size"] == c["max_model_len"], sc["name"]
        assert m["mamba_cache_mode"] in ("align", "none")
    for r in sc["requests"]:
        assert len(r["prompt"]) + len(r["output"]) < c["max_model_len"], (sc["name"], r["id"])
        assert all(1 <= t < 50256 for t in r["prompt"] + r["output"])


def make(name, cfg_args, build, description, max_steps=300, **cfg_kw):
    b = Builder(name)
    build(b)
    cfg = config(**cfg_args, **cfg_kw)
    return Scenario(b, cfg), description, max_steps


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    scenarios = []

    def add(*a, **k):
        scenarios.append(make(*a, **k))

    # (a) align mode, one full-attention group and one Mamba group at the same block size 8.
    def prefix_hit(b):
        a = b.add("A", 0, b.tokens(40), 6)
        b.add("B", 9, a[:32] + b.tokens(7), 5, max_tokens=3)
        b.add("C", 9, list(a), 4)
        b.add("D", 10, a[:20] + b.tokens(10), 4)
        b.add("E", 11, b.tokens(12), 3)
    add("mamba-align-prefix-hit", dict(budget=64, seqs=4, block=8, layers=toy_layers(8, "align")), prefix_hit,
        "Align mode, prefix caching on, one full-attention group and one Mamba group at block size 8. A's "
        "40-token prompt is split at 32 so the Mamba state at token 32 is cached. B shares 32 tokens (a hit "
        "through both groups) and stops at a client budget of 3; C repeats A whole (the hit stops a token "
        "short); D shares only 20 tokens, so the attention hit has no Mamba state to pair with and nothing "
        "is reused.")

    def chunk_below(b):
        x = b.add("X", 0, b.tokens(21), 5)
        b.add("Y", 1, b.tokens(30), 5)
        b.add("Z", 6, x[:16] + b.tokens(9), 4)
    add("mamba-align-chunk-below-block", dict(budget=5, seqs=2, block=8, layers=toy_layers(8, "align")),
        chunk_below,
        "Align mode with a token budget of 5, below the block size 8: a prefill chunk cannot reach a block "
        "boundary, so it advances inside a block and re-aligns at the next boundary.", max_steps=400)

    def chunk_mid(b):
        x = b.add("X", 0, b.tokens(37), 5)
        b.add("Y", 0, b.tokens(26), 5)
        b.add("Z", 8, x[:32] + b.tokens(5), 4)
    add("mamba-align-chunk-mid-block", dict(budget=12, seqs=3, block=8, layers=toy_layers(8, "align")),
        chunk_mid,
        "Align mode with a token budget of 12, between one and two blocks of 8: a chunk ends at the last "
        "block boundary inside the budget.")

    def chunk_above(b):
        x = b.add("X", 0, b.tokens(50), 6)
        b.add("Y", 0, b.tokens(19), 6)
        b.add("Z", 2, x[:32] + b.tokens(12), 5)
        b.add("W", 14, list(x), 3)
    add("mamba-align-chunk-above-block", dict(budget=20, seqs=3, block=8, layers=toy_layers(8, "align")),
        chunk_above,
        "Align mode with a token budget of 20, above two blocks: chunks of 16 tokens, two requests sharing "
        "the budget, and a later request that repeats a finished prompt.")

    def long_prompt(b):
        a = b.add("A", 0, b.tokens(110), 6)
        b.add("B", 3, a[:96] + b.tokens(4), 5)
    add("mamba-align-long-prompt", dict(budget=32, seqs=2, block=8, layers=toy_layers(8, "align")),
        long_prompt,
        "Align mode with a 110-token prompt (max_model_len 128) prefilled in 32-token chunks, and a second "
        "request sharing its first 96 tokens.")

    def threshold(b):
        a = b.add("A", 0, b.tokens(45), 5)
        b.add("B", 1, b.tokens(30), 5)
        b.add("C", 12, a[:40] + b.tokens(3), 4)
    add("mamba-align-long-prefill-threshold",
        dict(budget=64, seqs=3, block=8, layers=toy_layers(8, "align"), threshold=12), threshold,
        "Align mode with long_prefill_token_threshold 12 under a budget of 64: the threshold caps a chunk "
        "before the Mamba split aligns it to a block boundary.")

    def retention(b):
        a = b.add("A", 0, b.tokens(70), 4)
        b.add("B", 9, a[:40] + b.tokens(6), 4)
        b.add("C", 9, a[:66] + b.tokens(4), 4)
    for suffix, extra, what in [
        ("0", {"prefix_cache_retention_interval": 0},
         "prefix_cache_retention_interval 0, vLLM's default: only the prompt's last Mamba state is kept"),
        ("dense", {"prefix_cache_retention_interval": None},
         "prefix_cache_retention_interval null (dense): every state a prefill chunk ends on is kept"),
        ("32", {"prefix_cache_retention_interval": 32},
         "prefix_cache_retention_interval 32: a periodic state every 32 tokens beside the prompt's last")]:
        add(f"mamba-align-retention-{suffix}",
            dict(budget=32, seqs=3, block=8, layers=toy_layers(8, "align"), cache_config=extra), retention,
            f"Align mode, block size 8, token budget 32 so A's 70-token prompt ends chunks at 32 and 64; "
            f"{what}. B shares 40 tokens with A and C shares 66, so what each resumes from shows which "
            f"states were kept.")

    def match_unit(b):
        a = b.add("A", 0, b.tokens(70), 5)
        b.add("B", 8, a[:40] + b.tokens(9), 4)
        b.add("C", 8, a[:56] + b.tokens(3), 4)
    add("mamba-align-prefix-match-unit",
        dict(budget=32, seqs=3, block=16, layers=toy_layers(16, "align"), max_len=128,
             cache_config={"prefix_match_unit": 8}),
        match_unit,
        "Align mode, block size 16 with prefix_match_unit 8: hashes every 8 tokens, below the Mamba block, "
        "so a hit may land inside a block (the scheduler's mamba_partial_cache_hit path).")

    def three_groups(b):
        a = b.add("A", 0, b.tokens(41), 6)
        b.add("B", 4, a[:40] + b.tokens(4), 5)
        b.add("C", 4, b.tokens(18), 5)
    add("mamba-align-three-mamba-groups",
        dict(budget=24, seqs=3, block=8, layers=toy_layers(8, "align", mamba_layers=3)), three_groups,
        "Align mode with one full-attention layer and three Mamba layers: vLLM makes one attention group "
        "and three Mamba groups (as Nemotron-like models with more Mamba than attention layers), and a "
        "hit needs all of them.")

    # (b) the prefill checkpoint, block size 16.
    def ckpt(b):
        r1 = b.add("R1", 0, b.tokens(70), 5)
        b.add("R2", 2, r1[:50] + b.tokens(20), 5)
        b.add("R3", 12, list(r1), 4)
        b.add("R4", 12, b.tokens(33), 4)
    add("mamba-ckpt-1-1", dict(budget=40, seqs=4, block=16, layers=toy_layers(16, "align", ckpt=1, align=1)),
        ckpt,
        "Align mode, block size 16, one prefill checkpoint block with alignment 1 (Mamba-2's): the prompt's "
        "last chunk exports its state mid-chunk, so the split does not stop at the last block boundary.")
    add("mamba-ckpt-1-16", dict(budget=40, seqs=4, block=16, layers=toy_layers(16, "align", ckpt=1, align=16)),
        ckpt,
        "As mamba-ckpt-1-1 with alignment 16 (KDA with the flashkda backend): the checkpoint is valid only "
        "16 tokens from a chunk's start.")
    add("mamba-ckpt-1-1-small-budget",
        dict(budget=12, seqs=4, block=16, layers=toy_layers(16, "align", ckpt=1, align=1)), ckpt,
        "As mamba-ckpt-1-1 with a token budget of 12, below the block size 16.", max_steps=400)

    # (c) preemption and recomputation under a small pool: three requests that fit alone and not together.
    def preempt(prompt, out):
        def build(b):
            for i in range(3):
                b.add(f"P{i}", 0, b.tokens(prompt), out + 2 * i)
        return build
    add("mamba-align-preempt", dict(budget=24, seqs=3, block=8, layers=toy_layers(8, "align"), blocks=17),
        preempt(24, 24),
        "Align mode, a pool of 16 usable blocks: three requests of 24-token prompts grow past the pool "
        "during decode, so the scheduler preempts and a resumed request replays its outputs "
        "(prefill_end beyond the prompt).", max_steps=500)
    add("mamba-ckpt-preempt",
        dict(budget=32, seqs=3, block=16, layers=toy_layers(16, "align", ckpt=1, align=1), blocks=17),
        preempt(40, 40),
        "As mamba-align-preempt with a prefill checkpoint block (1, alignment 1) at block size 16.",
        max_steps=500)

    # (d) none mode: prefix caching off, the Mamba block is max_model_len.
    def none_basic(b):
        b.add("A", 0, b.tokens(30), 10)
        b.add("B", 1, b.tokens(50), 10)
        b.add("C", 2, b.tokens(20), 10, max_tokens=6)
    add("mamba-none-spec0",
        dict(budget=16, seqs=3, block=8, layers=toy_layers(8, "none", mamba_block=128), caching=False, blocks=60),
        none_basic,
        "None mode (prefix caching off): the Mamba block is max_model_len (128 tokens), so a request holds "
        "one Mamba block however long it grows, and no chunk is split for the Mamba state.")

    def none_spec(b):
        for i, (arr, p, o) in enumerate([(0, 30, 12), (1, 50, 12), (2, 20, 12)]):
            b.add(f"S{i}", arr, b.tokens(p), o)
            b.requests[-1]["accepted"] = [b.rng.randint(0, 2) for _ in range(o + 2)]
    add("mamba-none-spec2",
        dict(budget=16, seqs=3, block=8, layers=toy_layers(8, "none", speculative=2, mamba_block=128),
             caching=False, blocks=80, speculative={"method": "ngram", "num_speculative_tokens": 2}),
        none_spec,
        "None mode with n-gram speculation of 2 tokens and num_speculative_blocks 2: each request holds "
        "its Mamba state block plus 2 speculative ones.")
    add("mamba-none-preempt",
        dict(budget=24, seqs=3, block=8, layers=toy_layers(8, "none", mamba_block=128), caching=False, blocks=17),
        preempt(24, 24),
        "None mode, a pool of 16 usable blocks: preemption and recomputation without prefix caching.",
        max_steps=500)

    # (e) pool sizing and the start-up check, with the toy model: a request of max_model_len 128 needs
    # 16 full-attention pages of 256 bytes and the Mamba group's pages (align: 2 + checkpoint blocks;
    # none: 1 + speculative blocks).
    def tiny(b):
        b.add("A", 0, b.tokens(20), 4)
        b.add("B", 2, b.tokens(14), 3)
    add("mamba-pool-fits",
        dict(budget=32, seqs=2, block=8, layers=toy_layers(8, "align"), kv_available_bytes=8192,
             kv_startup_check=True), tiny,
        "Align mode, 8192 bytes of KV memory: the pool vLLM sizes over a full-attention and a Mamba group, "
        "and a start-up check that passes (one 128-token request needs 16 attention pages and 3 Mamba "
        "pages of 256 bytes).", max_steps=60)
    add("mamba-pool-refused",
        dict(budget=32, seqs=2, block=8, layers=toy_layers(8, "align"), kv_available_bytes=4096,
             kv_startup_check=True), tiny,
        "As mamba-pool-fits with 4096 bytes, less than one 128-token request needs: the start-up check "
        "refuses.", max_steps=60)
    add("mamba-none-pool-fits",
        dict(budget=32, seqs=2, block=8, layers=toy_layers(8, "none", speculative=0, mamba_block=128),
             caching=False, kv_available_bytes=8192, kv_startup_check=True), tiny,
        "None mode, 8192 bytes: the Mamba group needs one page per request, not the align mode's three.",
        max_steps=60)
    add("mamba-none-pool-refused",
        dict(budget=32, seqs=2, block=8, layers=toy_layers(8, "none", speculative=0, mamba_block=128),
             caching=False, kv_available_bytes=4096, kv_startup_check=True), tiny,
        "None mode, 4096 bytes: refused.", max_steps=60)

    # (f) page ratios of two real hybrid shapes. The sizes are examples with the models' structure,
    # not read from their checkpoints.
    # Kimi-K3-like: one MLA layer (576 elements of bfloat16 a token, block 64: a page of 73728 bytes)
    # for three KDA layers (a prefill checkpoint of 1 block, alignment 16, flashkda).
    kimi_layers = [attn(1, head=576, elem=2, spec="mla") | {"block_size": 64},
                   mamba(3, 65536, 73728, 64, "align", 0, 1, 16)]
    nemo_layers = [attn(1, heads=8, head=128, elem=2),
                   mamba(3, 400000, 458752, 112, "align", 0, 1, 1)]

    def big(prompts):
        def build(b):
            ids = []
            for i, (arr, p, o) in enumerate(prompts):
                base = ids[0] if ids and i == 2 else None
                prompt = base[: p // 2] + b.tokens(p - p // 2) if base else b.tokens(p)
                ids.append(b.add(f"R{i}", arr, prompt, o))
        return build
    reqs = [(0, 300, 6), (1, 120, 6), (6, 300, 5)]
    add("mamba-kimi-k3-like",
        dict(budget=256, seqs=4, block=64, layers=kimi_layers, blocks=300, max_len=1024,
             kv_available_bytes=10_000_000, kv_startup_check=True), big(reqs),
        "MLA beside KDA, Kimi-K3-like: one MLA layer (page 73728 bytes at block 64) and three Mamba layers "
        "whose state (65536 bytes) is padded to that page, with a prefill checkpoint (1 block, alignment "
        "16). One attention group and three Mamba groups; the pool and a passing start-up check.",
        max_steps=200)
    add("mamba-kimi-k3-like-startup-refused",
        dict(budget=256, seqs=4, block=64, layers=kimi_layers, blocks=300, max_len=1024,
             kv_available_bytes=1_000_000, kv_startup_check=True), big(reqs[:1]),
        "The Kimi-K3-like model with 1000000 bytes of KV memory, less than the MLA group needs for one "
        "1024-token request: refused.", max_steps=60)
    add("mamba-nemotron-like",
        dict(budget=336, seqs=4, block=112, layers=nemo_layers, blocks=300, max_len=1008,
             kv_available_bytes=100_000_000, kv_startup_check=True), big(reqs),
        "GQA beside Mamba-2, Nemotron-like: one attention layer of 8 heads x 128 in bfloat16 at block 112 "
        "(page 458752 bytes) and three Mamba layers whose 400000-byte state is padded to it, with a "
        "prefill checkpoint (1 block, alignment 1). The block size 112 is what aligning the attention "
        "page to a large Mamba state gives.", max_steps=200)
    add("mamba-nemotron-like-startup-refused",
        dict(budget=336, seqs=4, block=112, layers=nemo_layers, blocks=300, max_len=1008,
             kv_available_bytes=4_000_000, kv_startup_check=True), big(reqs[:1]),
        "The Nemotron-like model with 4000000 bytes of KV memory, less than the attention group needs for "
        "one 1008-token request: refused.", max_steps=60)

    for sc, description, max_steps in scenarios:
        write(out, sc, description, max_steps)
    print(f"wrote {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()
