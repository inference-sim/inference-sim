package latencykernel

import (
	"testing"
)

// A term the kernel cannot price from its inputs is refused or recorded, never charged at
// nothing in silence.

// A model-specific MTP name is the method vLLM runs as mtp, so a deployment naming mtp
// drafts with a graph's deepseek_mtp speculator.
func TestModelSpecificMTPNamesAreMTP(t *testing.T) {
	if got := specMethod("deepseek_mtp"); got != "mtp" {
		t.Errorf("deepseek_mtp normalises to %q", got)
	}
	if got := specMethod("dspark"); got != "dspark" {
		t.Errorf("dspark normalises to %q", got)
	}
}
