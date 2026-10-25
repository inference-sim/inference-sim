package rules_test

import (
	"os"
	"strings"
	"testing"
)

// The rules module asks for no newer Go than blis-schemas does (go 1.24), because the go
// command raises a consumer's own directive to its highest requirement's: a cost model
// importing these packs would otherwise need a toolchain only this module chose
// (inference-sim/blis-latency-kernel#87). The packs are data and pure functions, which need
// nothing newer.
func TestTheModuleAsksForNoNewerGoThanBlisSchemas(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "go "); ok {
			if v != "1.24" {
				t.Errorf("go.mod declares go %s; keep it at blis-schemas' go 1.24", v)
			}
			return
		}
	}
	t.Fatal("go.mod has no go directive")
}
