package validate

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// aliasBomb nests anchors depth levels deep, each level holding fan aliases of the one
// before: a few hundred bytes that expand to fan^depth nodes.
func aliasBomb(depth, fan int) string {
	var b strings.Builder
	b.WriteString("l0: &a0 [x, x, x, x, x, x, x, x, x, x]\n")
	for d := 1; d <= depth; d++ {
		fmt.Fprintf(&b, "l%d: &a%d [", d, d)
		for i := 0; i < fan; i++ {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "*a%d", d-1)
		}
		b.WriteString("]\n")
	}
	return b.String()
}

func TestCheckNodeRefusesAnAliasBombQuickly(t *testing.T) {
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte(aliasBomb(8, 10)), &doc); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	err := CheckNode(&doc, nil)
	if err == nil || !strings.Contains(err.Error(), "MaxNodes") {
		t.Fatalf("err = %v, want a refusal naming MaxNodes", err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("the bomb took %v to refuse", d)
	}
}

func TestCheckNodeAcceptsModerateAliasing(t *testing.T) {
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte(aliasBomb(3, 10)), &doc); err != nil {
		t.Fatal(err)
	}
	if err := CheckNode(&doc, nil); err != nil {
		t.Fatalf("a document of about 10^4 nodes was refused: %v", err)
	}
}

func TestCheckSize(t *testing.T) {
	if err := CheckSize("a.yaml", MaxDocumentBytes); err != nil {
		t.Errorf("a document at the limit was refused: %v", err)
	}
	if err := CheckSize("a.yaml", MaxDocumentBytes+1); err == nil || !strings.Contains(err.Error(), "MaxDocumentBytes") {
		t.Errorf("err = %v, want a refusal naming MaxDocumentBytes", err)
	}
}
