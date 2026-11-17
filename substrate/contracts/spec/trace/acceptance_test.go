package trace

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

func acceptanceHeader(w *acceptance.Workload) Header {
	h := sample().Header
	h.Acceptance = w
	return h
}

// A trace header carries the acceptance terms the workload stated (inference-sim#2115): written
// and read back they are equal, so a replay can resolve the profile the run did.
func TestHeaderAcceptanceRoundTrips(t *testing.T) {
	al := 2.5
	for name, w := range map[string]*acceptance.Workload{
		"category and thinking": {Category: "coding", Thinking: acceptance.ThinkingOn},
		"a value of its own":    {Value: &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: &al, Source: "measured"}},
		"all three":             {Category: "chat", Thinking: acceptance.ThinkingOff, Value: &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: &al}},
	} {
		t.Run(name, func(t *testing.T) {
			h := acceptanceHeader(w)
			var buf bytes.Buffer
			if err := WriteHeader(&buf, &h); err != nil {
				t.Fatal(err)
			}
			back, err := ReadHeader(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatalf("read back: %v\n%s", err, buf.String())
			}
			if !reflect.DeepEqual(back.Acceptance, w) {
				t.Errorf("acceptance read back as %+v, want %+v\n%s", back.Acceptance, w, buf.String())
			}
		})
	}
	// Nothing stated: nothing written, so a trace of a workload with no terms is unchanged.
	var buf bytes.Buffer
	h := acceptanceHeader(nil)
	if err := WriteHeader(&buf, &h); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), "acceptance") {
		t.Errorf("a header with no acceptance terms wrote some:\n%s", buf.String())
	}
}

func TestHeaderAcceptanceIsValidated(t *testing.T) {
	for name, tc := range map[string]struct {
		w    *acceptance.Workload
		want string
	}{
		"empty terms":   {&acceptance.Workload{}, "states no category, thinking mode or value"},
		"bad thinking":  {&acceptance.Workload{Thinking: "sometimes"}, "acceptance.thinking"},
		"invalid value": {&acceptance.Workload{Value: &acceptance.Value{}}, "acceptance.value"},
	} {
		t.Run(name, func(t *testing.T) {
			h := acceptanceHeader(tc.w)
			p := h.Validate()
			if p.OK() || !strings.Contains(p.Error(), tc.want) {
				t.Fatalf("Validate() = %v, want a problem naming %q", p, tc.want)
			}
		})
	}
}
