// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/lora/worker_manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// LRUCacheWorkerLoRAManager is the class.

package lora

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// trace is what oracle/run_lora_cache.py records of vLLM's own LRUCacheWorkerLoRAManager.
type trace struct {
	Config struct {
		MaxLoRAs    int `json:"max_loras"`
		MaxCPULoRAs int `json:"max_cpu_loras"`
	} `json:"config"`
	Steps []struct {
		Requested  []int   `json:"requested"`
		Events     [][]any `json:"events"`
		Error      *string `json:"error"`
		Registered []int   `json:"registered"`
		Active     []int   `json:"active"`
		Slots      []int   `json:"slots"`
	} `json:"steps"`
}

func eventsOf(t *testing.T, raw [][]any) []Event {
	t.Helper()
	var out []Event
	for _, r := range raw {
		out = append(out, Event{Kind(r[0].(string)), int(r[1].(float64))})
	}
	return out
}

// replay drives the Go cache with the trace's requests and returns the first difference from
// vLLM's recorded events, caches and slots, or "".
func replay(t *testing.T, tr trace) string {
	t.Helper()
	c, err := New(tr.Config.MaxLoRAs, tr.Config.MaxCPULoRAs)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range tr.Steps {
		ev, err := c.SetActive(s.Requested)
		if (err != nil) != (s.Error != nil) {
			return "step " + itoa(i) + ": error " + errString(err) + ", vLLM's " + ptrString(s.Error)
		}
		if s.Error != nil {
			// vLLM refuses before touching either cache.
			ev = nil
		}
		if want := eventsOf(t, s.Events); !reflect.DeepEqual(ev, want) && (len(ev) != 0 || len(want) != 0) {
			return "step " + itoa(i) + ": events " + fmtEvents(ev) + ", vLLM's " + fmtEvents(want)
		}
		if got := c.Registered(); !reflect.DeepEqual(got, s.Registered) && (len(got) != 0 || len(s.Registered) != 0) {
			return "step " + itoa(i) + ": CPU cache order differs"
		}
		if got := c.Active(); !reflect.DeepEqual(got, s.Active) && (len(got) != 0 || len(s.Active) != 0) {
			return "step " + itoa(i) + ": GPU cache order differs"
		}
		if got := c.Slots(); !reflect.DeepEqual(got, zeroNil(s.Slots)) {
			return "step " + itoa(i) + ": slots differ"
		}
	}
	return ""
}

func zeroNil(s []int) []int { return s }

func loadTraces(t *testing.T, dir string) map[string]trace {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no traces in %s (%v)", dir, err)
	}
	out := map[string]trace{}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var tr trace
		if err := json.Unmarshal(raw, &tr); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		out[strings.TrimSuffix(filepath.Base(f), ".json")] = tr
	}
	return out
}

// TestCacheReplaysVLLMsAdapterLRU holds the cache to vLLM's own LRUCacheWorkerLoRAManager, step by
// step: the adapters it loads, evicts, activates and deactivates in order, and the two caches' LRU
// order and the slots after each batch. The scenarios include a batch that evicts an adapter it
// then needs again (it is loaded twice) and a batch with more adapters than slots (refused).
func TestCacheReplaysVLLMsAdapterLRU(t *testing.T) {
	for name, tr := range loadTraces(t, "testdata/golden") {
		t.Run(name, func(t *testing.T) {
			if d := replay(t, tr); d != "" {
				t.Error(d)
			}
		})
	}
}

// TestCacheAgainstFuzzCorpus replays a corpus of random scenarios run through vLLM
// (oracle/gen_fuzz_lora.py), when BLIS_VLLM_LORA_FUZZ_DIR names one.
func TestCacheAgainstFuzzCorpus(t *testing.T) {
	dir := os.Getenv("BLIS_VLLM_LORA_FUZZ_DIR")
	if dir == "" {
		t.Skip("BLIS_VLLM_LORA_FUZZ_DIR is unset: no fuzz corpus to replay")
	}
	bad := 0
	all := loadTraces(t, dir)
	for name, tr := range all {
		if d := replay(t, tr); d != "" {
			bad++
			t.Errorf("%s: %s", name, d)
		}
	}
	t.Logf("%d fuzz traces, %d mismatches", len(all), bad)
}

func itoa(i int) string { b, _ := json.Marshal(i); return string(b) }
func errString(e error) string {
	if e == nil {
		return "none"
	}
	return e.Error()
}
func ptrString(s *string) string {
	if s == nil {
		return "none"
	}
	return *s
}
func fmtEvents(ev []Event) string { b, _ := json.Marshal(ev); return string(b) }
