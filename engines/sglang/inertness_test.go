// Original to blis-sglang.

package blissglang

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// A proven_inert field names a proof file, oracle/inertness/<field>.json, of kind observed_reads: runs
// of SGLang's own scheduler with the field at its default and at each probe, with a record of which
// of the field's reads (oracle/inertness.py; the scan follows the resolved-config bags) each run
// executed. The field is inert when the runs agree and every read, and every line that uses what a
// read stored, was executed (design 7.4, VLM-3; inference-sim/inference-sim#2350). The file is
// written by `oracle/inertness.py --write` and held to the source by `--check`; this holds the field
// list to the file.
type observedProof struct {
	Field  string `json:"field"`
	Kind   string `json:"kind"`
	Probes []struct {
		Value any `json:"value"`
	} `json:"probes"`
	Scenarios []struct {
		Name, Path, Golden string
	} `json:"scenarios"`
	Consumers []struct {
		At string `json:"at"`
	} `json:"consumers"`
	Reads []struct {
		At string `json:"at"`
		To int    `json:"to"`
	} `json:"reads"`
	Observed struct {
		Golden  map[string]string `json:"golden"`
		Runs    []observedRun     `json:"runs"`
		Verdict string            `json:"verdict"`
	} `json:"observed"`
}

type observedRun struct {
	Scenario  string   `json:"scenario"`
	Stream    bool     `json:"stream"`
	Value     any      `json:"value"`
	Digest    string   `json:"digest"`
	Executed  []string `json:"executed"`
	Consumers []string `json:"consumers"`
}

func contains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// proofProblems is why p does not prove field inert: empty when it does.
func proofProblems(field string, p observedProof) []string {
	var out []string
	if p.Field != field || p.Kind != "observed_reads" {
		out = append(out, fmt.Sprintf("the proof file is of kind %q for %q", p.Kind, p.Field))
	}
	if p.Observed.Verdict != "inert" {
		out = append(out, "the verdict is "+p.Observed.Verdict)
	}
	if len(p.Probes) == 0 || len(p.Reads) == 0 || len(p.Consumers) == 0 || len(p.Scenarios) == 0 {
		out = append(out, "the proof lists no probe, read, consumer or scenario")
	}
	type group struct {
		scenario string
		stream   bool
	}
	groups := map[group][]observedRun{}
	for _, r := range p.Observed.Runs {
		groups[group{r.Scenario, r.Stream}] = append(groups[group{r.Scenario, r.Stream}], r)
	}
	for _, sc := range p.Scenarios {
		for _, stream := range []bool{false, true} {
			rs := groups[group{sc.Name, stream}]
			want := map[string]bool{fmt.Sprint(nil): true}
			for _, pr := range p.Probes {
				want[fmt.Sprint(pr.Value)] = true
			}
			got := map[string]bool{}
			for _, r := range rs {
				got[fmt.Sprint(r.Value)] = true
				if r.Digest != rs[0].Digest {
					out = append(out, fmt.Sprintf("%s (stream %v): the result at %v differs from the default's", sc.Name, stream, r.Value))
				}
				if !stream && r.Digest != p.Observed.Golden[sc.Name] {
					out = append(out, fmt.Sprintf("%s: the non-streaming run at %v is not the scenario's golden", sc.Name, r.Value))
				}
			}
			if len(got) != len(want) || len(rs) != len(want) {
				out = append(out, fmt.Sprintf("%s (stream %v): ran at %d values, want the default and %d probes", sc.Name, stream, len(rs), len(p.Probes)))
			}
		}
	}
	for _, rd := range p.Reads {
		ran := false
		for _, r := range p.Observed.Runs {
			ran = ran || contains(r.Executed, rd.At)
		}
		if !ran {
			out = append(out, "no run executed the read at "+rd.At)
		}
	}
	for _, pr := range p.Probes {
		ran := false
		for _, r := range p.Observed.Runs {
			ran = ran || (fmt.Sprint(r.Value) == fmt.Sprint(pr.Value) && len(r.Executed) > 0)
		}
		if !ran {
			out = append(out, fmt.Sprintf("no run at the probe %v executed a read", pr.Value))
		}
	}
	for _, c := range p.Consumers {
		ran := false
		for _, r := range p.Observed.Runs {
			ran = ran || (r.Stream && contains(r.Consumers, c.At))
		}
		if !ran {
			out = append(out, "no streaming run executed the consumer at "+c.At)
		}
	}
	return out
}

func readObservedProof(t *testing.T, path string) observedProof {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var p observedProof
	if err := json.Unmarshal(b, &p); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return p
}

func TestEveryProvenInertFieldHasAnObservedProof(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, f := range l.Fields {
		if f.Status != fields.StatusProvenInert {
			continue
		}
		n++
		if want := "oracle/inertness/" + f.Name + ".json"; f.Evidence.Proof != want {
			t.Errorf("%s: the proof is %q, want %q", f.Name, f.Evidence.Proof, want)
			continue
		}
		p := readObservedProof(t, f.Evidence.Proof)
		for _, pr := range proofProblems(f.Name, p) {
			t.Errorf("%s: %s", f.Name, pr)
		}
		for _, sc := range p.Scenarios {
			for _, path := range []string{sc.Path, sc.Golden} {
				if _, err := os.Stat(path); err != nil {
					if _, err := os.Stat(path + ".gz"); err != nil {
						t.Errorf("%s: scenario %s names %s, which does not exist", f.Name, sc.Name, path)
					}
				}
			}
		}
	}
	if n == 0 {
		t.Fatal("no field is proven inert; the list has lost its proofs")
	}
}

// A proof that stops being one must fail: each change below is one a run, a dropped coverage read or
// a field that starts to matter would make to the observed record.
func TestAProofThatIsNotOneIsRefused(t *testing.T) {
	good := func() observedProof { return readObservedProof(t, "oracle/inertness/stream_interval.json") }
	if got := proofProblems("stream_interval", good()); len(got) != 0 {
		t.Fatalf("the committed proof has problems: %v", got)
	}
	cases := map[string]func(p *observedProof){
		"the field affects the result at a probe": func(p *observedProof) {
			for i := range p.Observed.Runs {
				if p.Observed.Runs[i].Stream && p.Observed.Runs[i].Value != nil {
					p.Observed.Runs[i].Digest += "x"
					return
				}
			}
		},
		"the field affects a non-streaming result": func(p *observedProof) {
			p.Observed.Golden[p.Scenarios[0].Name] += "x"
		},
		"a coverage read is dropped": func(p *observedProof) {
			for i := range p.Observed.Runs {
				p.Observed.Runs[i].Executed = nil
			}
		},
		"one read is never executed": func(p *observedProof) {
			at := p.Reads[len(p.Reads)-1].At
			for i := range p.Observed.Runs {
				var keep []string
				for _, e := range p.Observed.Runs[i].Executed {
					if e != at {
						keep = append(keep, e)
					}
				}
				p.Observed.Runs[i].Executed = keep
			}
		},
		"a consumer is never executed": func(p *observedProof) {
			for i := range p.Observed.Runs {
				p.Observed.Runs[i].Consumers = nil
			}
		},
		"a probe is never run": func(p *observedProof) {
			var keep []observedRun
			for _, r := range p.Observed.Runs {
				if fmt.Sprint(r.Value) != fmt.Sprint(p.Probes[0].Value) {
					keep = append(keep, r)
				}
			}
			p.Observed.Runs = keep
		},
		"the verdict is not inert":        func(p *observedProof) { p.Observed.Verdict = "not proven" },
		"the proof is a reviewed reading": func(p *observedProof) { p.Kind = "reviewed_reading_and_test" },
	}
	for name, mutate := range cases {
		p := good()
		mutate(&p)
		if len(proofProblems("stream_interval", p)) == 0 {
			t.Errorf("%s: the proof was not refused", name)
		}
	}
}

// A modelled field names the traces that show it changing a decision; each is a file in the
// tree (committed gzipped), non-empty, so no field is classed modelled without a comparison
// to read (inference-sim/inference-sim#2350, X1).
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
