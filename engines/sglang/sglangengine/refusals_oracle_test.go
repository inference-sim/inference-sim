// Original to blis-sglang.

package sglangengine

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
)

// SGLang's own refusals, as SGLang v0.5.23 itself raises them: sglang_refusals.json is what
// oracle/run_refusals.py records by calling SGLang's code (the oracle workflow runs it again and
// fails unless the file is what SGLang gives, oracle/compare_refusals.py). The port's refusal is
// compared with that recorded message, not a copy typed into a test (SGL-6, SIM-9).

type refusalCaseSpec struct {
	Name             string         `json:"name"`
	EngineArgs       map[string]any `json:"engine_args"`
	EngineCacheClass string         `json:"engine_cache_class"`
	CacheClass       string         `json:"cache_class"`
	EngineRefuses    bool           `json:"engine_refuses"`
}

type refusalAnswer struct {
	Name    string `json:"name"`
	Refused *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"refused"`
	Accepted string `json:"accepted"`
}

type refusalGolden struct {
	Provenance struct {
		SGLangCommit string `json:"sglang_commit"`
		CasesSHA256  string `json:"cases_sha256"`
	} `json:"provenance"`
	Cases []refusalAnswer `json:"cases"`
}

// readRefusals reads the recorded answers and the case file. It fails if the answers are missing, are
// for another SGLang commit, or were recorded for another case file than the one in the tree (a stale
// golden), or if a case has no answer.
func readRefusals(goldenPath, casesPath, commit string) ([]refusalCaseSpec, map[string]refusalAnswer, error) {
	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		return nil, nil, fmt.Errorf("the recorded SGLang refusals are missing (%v): record them with the oracle workflow (oracle/run_refusals.py)", err)
	}
	blob, err := os.ReadFile(casesPath)
	if err != nil {
		return nil, nil, err
	}
	var g refusalGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		return nil, nil, fmt.Errorf("%s: %v", goldenPath, err)
	}
	if g.Provenance.SGLangCommit != commit {
		return nil, nil, fmt.Errorf("the recorded refusals are SGLang %s's, this build models %s: record them again", g.Provenance.SGLangCommit, commit)
	}
	h := sha256.Sum256(blob)
	sum := hex.EncodeToString(h[:])
	if g.Provenance.CasesSHA256 != sum {
		return nil, nil, fmt.Errorf("the recorded refusals are for another oracle/refusals/cases.json (sha256 %s, the file is %s): record them again", g.Provenance.CasesSHA256, sum)
	}
	var spec struct {
		Cases []refusalCaseSpec `json:"cases"`
	}
	if err := json.Unmarshal(blob, &spec); err != nil {
		return nil, nil, err
	}
	if len(spec.Cases) == 0 {
		return nil, nil, errors.New("oracle/refusals/cases.json has no case")
	}
	answers := map[string]refusalAnswer{}
	for _, a := range g.Cases {
		answers[a.Name] = a
	}
	for _, c := range spec.Cases {
		a, ok := answers[c.Name]
		if !ok {
			return nil, nil, fmt.Errorf("the case %s has no recorded answer: record them again", c.Name)
		}
		if (a.Refused == nil) == (a.Accepted == "") {
			return nil, nil, fmt.Errorf("the case %s records neither a refusal nor an acceptance, or both", c.Name)
		}
	}
	return spec.Cases, answers, nil
}

func TestRefusalsAreSGLangsOwn(t *testing.T) {
	cases, answers, err := readRefusals("testdata/sglang_refusals.json", "../oracle/refusals/cases.json", blissglang.Upstream.Commit)
	if err != nil {
		t.Fatal(err)
	}
	refused, accepted := 0, 0
	for _, c := range cases {
		a := answers[c.Name]
		err := startUp(t, c.EngineArgs, nil)
		switch {
		case a.Refused != nil && c.EngineRefuses:
			refused++
			// SGLang names the class of the tree cache it was given; the port names its own (FlexKV).
			want, got := a.Refused.Message, ""
			if err != nil {
				got = err.Error()
			}
			if c.EngineCacheClass != "" {
				want = strings.ReplaceAll(want, "tree_cache is "+c.CacheClass+".", "tree_cache is <tree cache>.")
				got = strings.ReplaceAll(got, "tree_cache is "+c.EngineCacheClass+".", "tree_cache is <tree cache>.")
			}
			if err == nil || !strings.Contains(got, want) {
				t.Errorf("%s: SGLang raises %q, the port gave %v", c.Name, want, err)
			}
			if err != nil && !isRefusal(err) {
				t.Errorf("%s: the refusal is not SGLang's own: %v", c.Name, err)
			}
		case a.Refused == nil && !c.EngineRefuses:
			accepted++
			// SGLang accepts the input: the port starts, or declines it for what it does not model
			// (an issue-cited NotModelled), never with a refusal SGLang does not make (SIM-9).
			if err != nil && strings.Contains(err.Error(), "SGLang refuses this configuration") {
				t.Errorf("%s: SGLang accepts this and the port refuses it as SGLang's: %v", c.Name, err)
			}
		default:
			t.Errorf("%s: the case file says the port refuses=%v but SGLang refuses=%v: they are one fact", c.Name, c.EngineRefuses, a.Refused != nil)
		}
	}
	if refused < 5 || accepted < 5 {
		t.Fatalf("held %d refusals and %d accepted partners, want every quoted refusal (5) and its partner (5)", refused, accepted)
	}
}

// The loader fails on a golden that is missing, is another commit's, is for another case file, or
// lacks an answer: a refusal test that read a stale golden would pass against the wrong thing.
func TestRefusalGoldenLoaderFailsWhenMissingOrStale(t *testing.T) {
	dir := t.TempDir()
	cases := filepath.Join(dir, "cases.json")
	golden := filepath.Join(dir, "golden.json")
	const blob = `{"cases":[{"name":"a","engine_refuses":true}]}`
	sum := sha256.Sum256([]byte(blob))
	good := `{"provenance":{"sglang_commit":"c1","cases_sha256":"` + hex.EncodeToString(sum[:]) + `"},` +
		`"cases":[{"name":"a","refused":{"type":"ValueError","message":"m"}}]}`
	if err := os.WriteFile(cases, []byte(blob), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, golden, commit string
		write                bool
		fails                bool
	}{
		{"current", good, "c1", true, false},
		{"missing", "", "c1", false, true},
		{"another commit", good, "c2", true, true},
		{"another case file", strings.Replace(good, hex.EncodeToString(sum[:]), strings.Repeat("0", 64), 1), "c1", true, true},
		{"no answer for a case", strings.Replace(good, `"name":"a"`, `"name":"b"`, 1), "c1", true, true},
		{"neither a refusal nor an acceptance", strings.Replace(good, `"refused":{"type":"ValueError","message":"m"}`, `"x":1`, 1), "c1", true, true},
	} {
		if err := os.Remove(golden); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if tc.write {
			if err := os.WriteFile(golden, []byte(tc.golden), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		_, _, err := readRefusals(golden, cases, tc.commit)
		if failed := err != nil; failed != tc.fails {
			t.Errorf("%s: the loader failed=%v, want %v", tc.name, failed, tc.fails)
		}
	}
}
