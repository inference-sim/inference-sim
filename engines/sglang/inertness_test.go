// Original to blis-sglang.

package blissglang

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// A proven_inert field names a proof file, oracle/inertness/<field>.json; the file is for that
// field, lists the probe values the value classes were computed from, and names a Go test that
// exists. The test is the proof (it resolves and lowers a pool at each probe and finds nothing
// changed); this holds the list to those files.
func TestEveryProvenInertFieldNamesItsProof(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range l.Fields {
		if f.Status != fields.StatusProvenInert {
			continue
		}
		b, err := os.ReadFile(f.Evidence.Proof)
		if err != nil {
			t.Errorf("%s: %v", f.Name, err)
			continue
		}
		var p struct {
			Field  string `json:"field"`
			Test   struct{ File, Func string }
			Probes []struct{ Value any } `json:"probes"`
		}
		if err := json.Unmarshal(b, &p); err != nil {
			t.Errorf("%s: %v", f.Name, err)
			continue
		}
		if p.Field != f.Name || len(p.Probes) == 0 {
			t.Errorf("%s: proof file is for %q with %d probes", f.Name, p.Field, len(p.Probes))
		}
		src, err := os.ReadFile(p.Test.File)
		if err != nil || !strings.Contains(string(src), "func "+p.Test.Func+"(") {
			t.Errorf("%s: the proof names %s %s, which does not exist", f.Name, p.Test.File, p.Test.Func)
		}
	}
}

// A modelled field names the traces that show it changing a decision; each is a file in the
// tree (committed gzipped), non-empty, so no field is classed modelled without a comparison
// to read (blis-sglang#63, X1).
func TestEveryModelledFieldNamesTracesThatExist(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, f := range l.Fields {
		if f.Status != fields.StatusModelled {
			continue
		}
		n++
		if len(f.Evidence.Traces) == 0 {
			t.Errorf("%s: modelled with no trace", f.Name)
		}
		for _, tr := range f.Evidence.Traces {
			fi, err := os.Stat(tr)
			if err != nil {
				fi, err = os.Stat(tr + ".gz")
			}
			if err != nil || fi.Size() == 0 {
				t.Errorf("%s: trace %s is missing or empty", f.Name, tr)
			}
		}
	}
	if n < 20 {
		t.Fatalf("only %d modelled fields in the list", n)
	}
}
