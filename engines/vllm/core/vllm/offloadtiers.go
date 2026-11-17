// Original to blis-vllm.

package vllm

// modelledTiers are the secondary tier types TieringOffloadingSpec registers
// (SecondaryTierFactory.register_tier, vllm/v1/kv_offload/tiering/factory.py:146-174@4ba787ec)
// that the engine builds beside the CPU tier, each run by the connector's TieringOffloadingManager
// (offloading.Config.Manager). buildOffload refuses every other registered type by name; a type
// vLLM does not register is refused first, as code_not_visible. surface.json marks exactly these
// modelled. None is yet: each joins with its oracle evidence.
var modelledTiers = map[string]bool{}
