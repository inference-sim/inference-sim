// Original to blis-vllm.

package vllm

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
)

// PYTHONUNBUFFERED is proven inert (oracle/inertness/PYTHONUNBUFFERED.json): Python reads it to write
// its standard streams without a buffer, and nothing in vLLM's serving process names it. A recipe
// that states it is accepted and runs step for step as the recipe without it; any other variable
// the engine does not read is still refused by name.
func TestAnInertEnvVariableChangesNothingTheEngineDecides(t *testing.T) {
	arr := arrivals(req("a", 40, 12), req("b", 33, 9), req("c", 20, 7))
	want := enginetest.TestEngine(t, build(t, ""), arr, enginetest.Options{})
	extra := "env: {PYTHONUNBUFFERED: \"1\"}\n"
	if _, err := build(t, extra)(); err != nil {
		t.Fatalf("PYTHONUNBUFFERED is refused: %v", err)
	}
	got := enginetest.TestEngine(t, build(t, extra), arr, enginetest.Options{})
	if !reflect.DeepEqual(got.Steps, want.Steps) || !reflect.DeepEqual(got.Finished, want.Finished) {
		t.Error("stating PYTHONUNBUFFERED changes the engine's steps")
	}
	_, err := build(t, "env: {PYTHONUNBUFFERED: \"1\", VLLM_LOGGING_LEVEL: DEBUG}\n")()
	if err == nil || !strings.Contains(err.Error(), "VLLM_LOGGING_LEVEL") {
		t.Errorf("an unread variable beside it must be refused by its own name, got %v", err)
	}
}
