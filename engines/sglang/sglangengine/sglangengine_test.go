// Original to blis-sglang.

package sglangengine

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

func TestTheFactoryRefusesAnotherReleaseAndNamesWhatItNeeds(t *testing.T) {
	if _, err := Factory().New(engine.Spec{Kind: Kind, UpstreamVersion: "0.4.0"}); err == nil || !strings.Contains(err.Error(), "0.4.0") {
		t.Fatalf("got %v, want a refusal naming the release", err)
	}
	if _, err := Factory().New(engine.Spec{Kind: Kind}); err == nil || !strings.Contains(err.Error(), "#50") || !strings.Contains(err.Error(), "#49") {
		t.Fatalf("got %v, want a refusal citing #50 and #49", err)
	}
	if Factory().Kind() != Kind {
		t.Fatal("wrong kind")
	}
}
