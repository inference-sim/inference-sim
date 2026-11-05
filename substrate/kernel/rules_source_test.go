package latencykernel

import (
	"os/exec"
	"strings"
	"testing"
)

// The library takes its rules from its caller and imports no engine model: only this
// repository's commands, its documentation generator and its tests may reach the vLLM packs
// (through internal/harness). A library import would put vLLM's rules back inside the cost
// model, which is what taking them from the caller removes.
func TestTheLibraryImportsNoEngineModel(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", ".", "./internal/price", "./internal/resolve").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(dep, "github.com/inference-sim/blis-vllm") {
			t.Errorf("the library depends on %s; it must take engine rules from its caller", dep)
		}
	}
}
