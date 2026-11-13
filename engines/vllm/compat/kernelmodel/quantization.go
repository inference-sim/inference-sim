// Original to blis-vllm.

package kernelmodel

import (
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/quantization"
)

// lowerQuantization applies vLLM's resolution of `quantization` (package quantization) to the
// settings the kernel prices, for a checkpoint that package covers: the stated value is cleared,
// because the layout vLLM serves is the checkpoint's own recipe and not a weight width the flag
// names, and the layout is stated (Settings.ServedWeights) for the kernel to prove the graph stores.
// A value vLLM refuses is refused here with vLLM's message.
//
// A checkpoint the package does not cover keeps its stated value, which the kernel prices as it did.
// When vLLM replaces the stated value with the checkpoint's own method, the returned substitution
// records the replacement.
func lowerQuantization(g *model.Graph, stated string, settings *enginesettings.Settings) (*sim.Substitution, error) {
	if g == nil {
		return nil, nil
	}
	low, err := quantization.Lower(g.DerivedFrom.SHA256, stated)
	if err != nil || low == nil {
		return nil, err
	}
	settings.Quantization = ""
	served := low.Served
	settings.ServedWeights = &served
	if low.Requested == "" {
		return nil, nil
	}
	return &sim.Substitution{Setting: "quantization", Stated: low.Requested, Priced: low.Method,
		Reason: "vLLM v0.32.0 serves this checkpoint with the quantization method its own recipe selects " +
			"whatever --quantization names (ModelConfig._verify_quantization, vllm/config/model.py:1416-1437@5df63a72)"}, nil
}
