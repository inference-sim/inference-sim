// Original to blis-vllm.

package tiering

import (
	"encoding/json"
	"testing"
)

// ParseTierFilter answers as vLLM's _parse_tier_filter does (offloading/scheduler.py:496-537@4ba787ec)
// on a table of kv_load_tiers values: each row is the value as JSON and, for the nine pairs of a
// tier's medium (none, CPU, STORAGE) and locality (none, LOCAL, REMOTE), whether vLLM's TierFilter
// allows a tier of that medium and locality ("1") or not ("0"). The rows were produced by running
// vLLM's function and classes, copied unchanged, on these values.
func TestParseTierFilterAnswersAsVLLMsDoes(t *testing.T) {
	for _, c := range []struct{ raw, allows string }{
		{`null`, "111111111"},
		{`0`, "111111111"},
		{`""`, "111111111"},
		{`"cpu"`, "111111111"},
		{`{}`, "111111111"},
		{`[]`, "000000000"},
		{`[{}]`, "111111111"},
		{`[null]`, "111111111"},
		{`["cpu"]`, "111111111"},
		{`[{"medium":"cpu"}]`, "111111000"},
		{`[{"medium":"CPU"}]`, "111111000"},
		{`[{"medium":"Storage"}]`, "111000111"},
		{`[{"medium":"disk"}]`, "111111111"},
		{`[{"medium":3}]`, "111111111"},
		{`[{"medium":null}]`, "111111111"},
		{`[{"medium":""}]`, "111111111"},
		{`[{"locality":"local"}]`, "110110110"},
		{`[{"locality":"Remote"}]`, "101101101"},
		{`[{"locality":"near"}]`, "111111111"},
		{`[{"locality":1}]`, "111111111"},
		{`[{"medium":"cpu","locality":"remote"}]`, "101101000"},
		{`[{"medium":"storage","locality":"local"}]`, "110000110"},
		{`[{"medium":"cpu"},{"medium":"bogus"}]`, "111111000"},
		{`[{"medium":"bogus"}]`, "111111111"},
		{`[{"medium":"bogus"},{"locality":"x"}]`, "111111111"},
		{`[{"medium":"cpu"},"x"]`, "111111000"},
		{`["x",3]`, "111111111"},
		{`[{"medium":"cpu"},{"medium":"storage","locality":"remote"}]`, "111111101"},
		{`[{"other":1}]`, "111111111"},
		{`[{"medium":"cpu","extra":true}]`, "111111000"},
		{`[{"medium":"bogus","locality":"local"}]`, "111111111"},
		{`[{"medium":"cpu","locality":"bogus"}]`, "111111111"},
	} {
		var raw any
		if err := json.Unmarshal([]byte(c.raw), &raw); err != nil {
			t.Fatal(err)
		}
		f := ParseTierFilter(raw)
		i := 0
		for _, med := range []Medium{MediumNone, MediumCPU, MediumStorage} {
			for _, loc := range []Locality{LocalityNone, LocalityLocal, LocalityRemote} {
				want := c.allows[i] == '1'
				i++
				if got := f.allows(med, loc); got != want {
					t.Errorf("kv_load_tiers %s: allows(medium %q, locality %q) = %v, vLLM %v", c.raw, med, loc, got, want)
				}
			}
		}
	}
}
