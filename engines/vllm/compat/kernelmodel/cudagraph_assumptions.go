// Original to blis-vllm.

package kernelmodel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// cudagraphAssumptions is the divergence a deployment that states cudagraph_mode or the capture
// sizes carries: the kernel prices the stated value under blis-substrate's vLLM v0.31.0 policy (the
// update to v0.32.0's is tracked in blis-vllm#78), and the two agree on the default mode (FULL_AND_PIECEWISE
// at optimization level 2, vllm/config/vllm.py:303,324), on the DeepEP high-throughput and MoonEP
// forced-off cases (compilation.py:1237-1253; vllm.py:2067-2076; parallel.py:784-789) and on the
// capture sizes (VllmConfig._set_cudagraph_sizes; TestKernelCaptureTokensAreVLLMs). They differ in one
// way the kernel cannot see: v0.32.0 lowers a stated mode where the attention backend cannot capture a
// full graph for a mixed batch, or refuses a mode with full decode graphs where it can capture none
// (CompilationConfig.resolve_cudagraph_mode_and_sizes, compilation.py:1369-1420). The same function
// is reached for an encoder-decoder model (vllm.py:1888-1899), a pooling model (vllm.py:1879-1885) and
// a KV connector that needs PIECEWISE (LMCache with use_layerwise, vllm.py:1900-1925); the catalog has no
// such model and the engine block cannot state that connector option (blis-vllm#89), so those
// are not part of the divergence here.
//
// The text names the field and the stated value, as the language_model_only assumption does, so a
// point's substitution list can copy it.
func cudagraphAssumptions(e engineconfig.Engine) []string {
	var out []string
	if m := e.CUDAGraphMode; m != "" && !strings.EqualFold(m, "NONE") {
		out = append(out, fmt.Sprintf("compilation_config.cudagraph_mode = %s: priced under the kernel's vLLM 0.31.0 policy "+
			"(blis-vllm#78); vLLM 0.32.0 resolves the mode against the attention backend's cudagraph support and may run a "+
			"lower mode or refuse this one, which the kernel does not see; the default mode and the capture sizes agree with 0.32.0", m))
	}
	if len(e.CUDAGraphCaptureSizes) > 0 {
		out = append(out, fmt.Sprintf("compilation_config.cudagraph_capture_sizes = %v: priced under the kernel's vLLM 0.31.0 "+
			"policy (blis-vllm#78); the sizes that run equal vLLM 0.32.0's resolution, but the mode they are captured under "+
			"can differ as cudagraph_mode's assumption says", e.CUDAGraphCaptureSizes))
	}
	return out
}
